"""HTML report generation for ClinMRI-QC.

Two public functions:

    generate_report(result, nifti_path, output_path=None)
        Single-scan HTML report from a detect_artifacts() result dict.

    generate_html_from_csv(csv_path, output_path=None)
        Multi-patient HTML report from a CSV produced by append_csv_record().
        Adapts automatically to single vs. multi-patient input.

Artifact severity scores in both reports are expected to be scaled [0,1]
values as stored by build_qc_record() (pulled from artifact_severity_scaled).
"""

import base64
import csv
import io
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clinmriqc.schema import (
    ALL_COLUMNS, ARTIFACT_CLASSES, IQM_RANGES, RECOMMENDATIONS, severity_label,
)

_FLAG_COLOURS = {'GREEN': '#4ade80', 'YELLOW': '#fbbf24', 'RED': '#f87171'}

_STATUS_COLOURS = {'pass': '#4ade80', 'warning': '#fbbf24', 'fail': '#f87171'}
_STATUS_CLS     = {'pass': 'pass',    'warning': 'warn',    'fail': 'fail'}


def _safe_float(v, fmt='.3f', fallback='—'):
    """Format v as a float string, returning fallback if v is empty or non-numeric."""
    try:
        return f'{float(v):{fmt}}'
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _val(row: dict, col: str, default=''):
    v = row.get(col, default)
    return default if v == '' else v


def _plot_slice(nifti_path: str) -> str:
    vol = nib.load(nifti_path).get_fdata()
    z   = vol.shape[2] // 2
    sl  = np.rot90(vol[:, :, z].astype(np.float32))
    p2, p98 = np.percentile(sl, 2), np.percentile(sl, 98)
    sl  = np.clip((sl - p2) / max(p98 - p2, 1e-6), 0, 1)
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor='#0f172a')
    ax.imshow(sl, cmap='gray', aspect='equal')
    ax.axis('off')
    ax.set_title(f'Axial slice {z}', color='#94a3b8', fontsize=8, pad=4)
    fig.tight_layout(pad=0.2)
    return _fig_to_b64(fig)


def _plot_severity_bars(severity: dict, detected: list) -> str:
    """Horizontal bar chart of scaled [0,1] severity scores for a single scan."""
    try:
        from clinmriqc.artifacts import SCALED_THRESHOLDS
    except ImportError:
        SCALED_THRESHOLDS = {}

    classes = ARTIFACT_CLASSES
    scores  = [float(severity.get(c, 0)) for c in classes]
    labels  = [c.replace('_', ' ').title() for c in classes]
    colours = ['#ef4444' if c in detected else '#475569' for c in classes]

    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    bars = ax.barh(labels, scores, color=colours, height=0.55, edgecolor='none')

    # Per-class threshold markers
    for i, cls in enumerate(classes):
        thr = SCALED_THRESHOLDS.get(cls)
        if thr is not None:
            ax.axvline(thr, ymin=(i / len(classes)) + 0.02,
                       ymax=((i + 1) / len(classes)) - 0.02,
                       color='#facc15', linewidth=1.5, linestyle='--', alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Scaled severity [0 – 1]', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#cbd5e1', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, score in zip(bars, scores):
        ax.text(min(score + 0.02, 0.96), bar.get_y() + bar.get_height() / 2,
                f'{score:.2f}', va='center', color='#f1f5f9', fontsize=8.5)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.5)
    return _fig_to_b64(fig)


def _iqm_gauge_html(key: str, value: float) -> str:
    info    = IQM_RANGES[key]
    lo, hi  = info['normal']
    warning = info['warning']
    max_val = hi * 1.6
    lo_pct  = lo  / max_val * 100
    hi_pct  = hi  / max_val * 100
    val_pct = min(value / max_val * 100, 99)
    colour  = '#22c55e' if value >= warning else '#ef4444'
    status  = ('✓ normal' if lo <= value <= hi
                else ('⚠ below normal' if value < lo else '⚠ above normal'))
    return f"""
    <div style="position:relative;background:#334155;height:10px;border-radius:6px;margin:10px 0 6px">
      <div style="position:absolute;left:{lo_pct:.1f}%;width:{hi_pct-lo_pct:.1f}%;
                  background:#166534;height:100%;border-radius:4px"></div>
      <div style="position:absolute;left:{val_pct:.1f}%;transform:translateX(-50%);
                  top:-4px;width:18px;height:18px;background:{colour};
                  border-radius:50%;border:2px solid #1e293b;z-index:2"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:2px">
      <span style="color:#475569">0</span>
      <span style="color:{colour};font-weight:700">{value:.4f}
        <span style="font-weight:400;opacity:0.8">{status}</span>
      </span>
      <span style="color:#475569">{max_val:.2f}</span>
    </div>
    <div style="font-size:10px;color:#334155;margin-top:3px">
      ◼ Normal range: {lo}–{hi}
    </div>"""


# ---------------------------------------------------------------------------
# Single-scan CSS + generate_report()
# ---------------------------------------------------------------------------

_CSS_SINGLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f172a; color: #e2e8f0;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       font-size: 14px; padding: 24px; }
.card { background: #1e293b; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
h1 { font-size: 20px; font-weight: 700; color: #f1f5f9; }
h2 { font-size: 13px; font-weight: 600; text-transform: uppercase;
     letter-spacing: 0.08em; color: #64748b; margin-bottom: 14px; }
.meta { color: #64748b; font-size: 12px; margin-top: 4px; }
.verdict { display: inline-block; padding: 6px 18px; border-radius: 99px;
           font-size: 15px; font-weight: 700; margin-top: 12px; }
.pass { background: #14532d; color: #4ade80; }
.fail { background: #450a0a; color: #f87171; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.artifact-tag { display: inline-block; padding: 3px 10px; border-radius: 99px;
                font-size: 12px; font-weight: 600; margin: 3px 3px 0 0; }
.rec { border-left: 3px solid; padding: 10px 14px; margin-bottom: 10px;
       border-radius: 0 8px 8px 0; background: #0f172a; }
.rec-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.rec-body  { color: #94a3b8; font-size: 12px; line-height: 1.5; }
.disclaimer { font-size: 11px; color: #475569; border-top: 1px solid #1e293b;
              padding-top: 12px; margin-top: 4px; }
img.chart { max-width: 100%; border-radius: 8px; }
"""


def generate_report(
    result: dict,
    nifti_path: str,
    output_path: str = None,
) -> str:
    """Generate a self-contained single-scan HTML QC report.

    Parameters
    ----------
    result      : dict returned by detect_artifacts().
    nifti_path  : path to the original NIfTI (used for the slice preview).
    output_path : where to save the HTML.  Defaults to
                  <nifti_stem>_qc_report.html next to the input.

    Returns
    -------
    str — path to the saved HTML file.
    """
    from clinmriqc.generate_csv import build_qc_record
    from clinmriqc.append_csv   import append_csv_record

    nifti_path = Path(nifti_path)
    if output_path is None:
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        output_path = nifti_path.parent / f'{stem}_qc_report.html'

    detected = result.get('artifacts_detected', [])
    passed   = result.get('quality_passed', True)
    iqms     = result.get('iqms', {})
    # Use scaled [0,1] severity for display; fall back to raw or old softmax schema.
    severity = (
        result.get('artifact_severity_scaled')
        or result.get('artifact_severity')
        or result.get('artifact_probabilities', {})
    )
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    severity_chart = _plot_severity_bars(severity, detected)
    slice_img      = _plot_slice(str(nifti_path))

    verdict_cls  = 'pass' if passed else 'fail'
    verdict_text = ('✓ PASS — no artifacts detected' if passed
                    else f'✗ FAIL — {", ".join(detected)} detected')

    tag_html = ''
    if detected:
        for art in detected:
            lbl, col = severity_label(float(severity.get(art, 0)))
            tag_html += (f'<span class="artifact-tag" style="background:{col}22;color:{col}">'
                         f'{art.replace("_"," ")} ({lbl})</span>')
    else:
        tag_html = '<span style="color:#4ade80;font-size:13px">None detected</span>'

    iqm_html = ''
    for key, value in iqms.items():
        if key not in IQM_RANGES:
            continue
        info  = IQM_RANGES[key]
        iqm_html += f"""
        <div style="margin-bottom:18px">
          <div style="font-size:12px;color:#94a3b8;margin-bottom:2px">{info['label']}</div>
          {_iqm_gauge_html(key, value)}
          <div style="font-size:11px;color:#475569;margin-top:4px">{info['note']}</div>
        </div>"""

    rec_html = ''
    if detected:
        for art in detected:
            lbl, col = severity_label(float(severity.get(art, 0)))
            title, body = RECOMMENDATIONS.get(art, (art, ''))
            rec_html += f"""
            <div class="rec" style="border-color:{col}">
              <div class="rec-title" style="color:{col}">
                ⚠ {title} &nbsp;
                <span style="font-weight:400;font-size:11px">
                  {lbl} — score {severity.get(art, 0):.2f}
                </span>
              </div>
              <div class="rec-body">{body}</div>
            </div>"""
    else:
        rec_html = '<p style="color:#4ade80;font-size:13px">No action required. Scan appears clean.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClinMRI-QC Report — {nifti_path.name}</title>
<style>{_CSS_SINGLE}</style>
</head>
<body>

<div class="card">
  <h1>ClinMRI-QC Report</h1>
  <div class="meta">Scan: <strong style="color:#cbd5e1">{nifti_path.name}</strong>
    &nbsp;|&nbsp; Generated: {now} &nbsp;|&nbsp; KCL BMEIS
  </div>
  <div class="verdict {verdict_cls}">{verdict_text}</div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Artifact Severity</h2>
    <img class="chart" src="data:image/png;base64,{severity_chart}">
    <div style="font-size:11px;color:#475569;margin-top:8px">
      Scores scaled 0–1 relative to worst observed / synthetic artifact.
      Dashed yellow lines = per-class detection thresholds.
    </div>
    <div style="margin-top:12px">
      <div style="font-size:12px;color:#64748b;margin-bottom:6px">Detected artifacts</div>
      {tag_html}
    </div>
  </div>
  <div class="card">
    <h2>Representative Slice</h2>
    <img class="chart" src="data:image/png;base64,{slice_img}">
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Image Quality Metrics</h2>
    {iqm_html or '<span style="color:#334155;font-size:12px">Not available</span>'}
    <div style="font-size:11px;color:#334155;margin-top:8px">
      ◼ Green band = normal range (derived from 30 clean T1w scans)
    </div>
  </div>
  <div class="card">
    <h2>Recommendations</h2>
    {rec_html}
    <div class="disclaimer" style="margin-top:16px">
      Results are classifier-based estimates. Review alongside visual inspection.
    </div>
  </div>
</div>

</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    csv_path = Path(str(output_path).replace('.html', '.csv'))
    record = build_qc_record(image_path=str(nifti_path), artifacts=result, timestamp=now)
    append_csv_record(record, str(csv_path))

    print(f'Report saved to {output_path}')
    print(f'CSV    saved to {csv_path}')
    return str(output_path)


# ---------------------------------------------------------------------------
# Multi-patient CSS
# ---------------------------------------------------------------------------

_CSS_REPORT = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f172a; color: #e2e8f0;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       font-size: 14px; line-height: 1.5; }
.page { max-width: 1080px; margin: 0 auto; padding: 28px 24px; }
.card { background: #1e293b; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
h1 { font-size: 20px; font-weight: 700; color: #f1f5f9; }
h2 { font-size: 11px; font-weight: 700; text-transform: uppercase;
     letter-spacing: 0.09em; color: #475569; margin-bottom: 14px; }
.meta { color: #64748b; font-size: 12px; margin-top: 6px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.stat-row { display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap; }
.stat-val { font-size: 28px; font-weight: 700; line-height: 1; }
.stat-lbl { font-size: 11px; color: #64748b; margin-top: 4px; }
.verdict { display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px;
           border-radius: 99px; font-size: 13px; font-weight: 700; margin-top: 12px; }
.pass { background: #052e16; color: #4ade80; }
.fail { background: #450a0a; color: #f87171; }
.warn { background: #451a03; color: #fbbf24; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 99px;
         font-size: 11px; font-weight: 600; margin: 2px; }
.badge-on  { background: #052e1644; color: #4ade80; border: 1px solid #14532d; }
.badge-off { background: transparent; color: #475569; border: 1px solid #334155; }
.mod { border-left: 3px solid; padding: 14px 0 4px 16px; margin-bottom: 20px; }
.mod.art  { border-color: #7c3aed; }
.mod.con  { border-color: #0284c7; }
.mod.reg  { border-color: #0f766e; }
.mod.meta { border-color: #d97706; }
.mod-title { font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
             text-transform: uppercase; color: #475569; margin-bottom: 10px; }
.prob-row  { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
.prob-lbl  { width: 76px; font-size: 12px; color: #94a3b8; flex-shrink: 0;
             text-transform: capitalize; }
.prob-bg   { flex: 1; background: #0f172a; border-radius: 3px; height: 7px;
             overflow: visible; position: relative; }
.prob-fill { height: 100%; border-radius: 3px; }
.prob-thr  { position: absolute; top: -3px; bottom: -3px; width: 2px;
             background: #facc15; border-radius: 1px; pointer-events: none; }
.prob-val  { width: 34px; font-size: 12px; font-weight: 500; text-align: right;
             flex-shrink: 0; }
.art-tag  { display: inline-block; padding: 3px 10px; border-radius: 99px;
            font-size: 12px; font-weight: 600; margin: 2px; }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
               margin-bottom: 14px; }
.metric-box  { background: #0f172a; border-radius: 8px; padding: 12px 14px; }
.metric-val  { font-size: 20px; font-weight: 700; color: #f1f5f9; }
.metric-lbl  { font-size: 11px; color: #64748b; margin-top: 2px; }
.metric-ok   { color: #4ade80; font-size: 11px; margin-top: 5px; }
.metric-warn { color: #fbbf24; font-size: 11px; margin-top: 5px; }
.metric-bad  { color: #f87171; font-size: 11px; margin-top: 5px; }
.iqm-note    { font-size: 11px; color: #334155; margin-top: 4px; }
.rec { border-left: 3px solid; padding: 10px 14px; margin-bottom: 10px;
       border-radius: 0 8px 8px 0; background: #0f172a; }
.rec-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.rec-body  { color: #94a3b8; font-size: 12px; line-height: 1.6; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { padding: 8px 10px; color: #475569; font-weight: 700; text-transform: uppercase;
     font-size: 10px; letter-spacing: 0.06em; border-bottom: 1px solid #334155;
     text-align: left; white-space: nowrap; background: #0f172a; }
td { padding: 9px 10px; border-bottom: 1px solid #1e293b; vertical-align: middle;
     white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #1e293b99; }
details { background: #1e293b; border-radius: 10px; margin-bottom: 8px;
          overflow: hidden; }
summary { cursor: pointer; padding: 15px 18px; list-style: none;
          display: flex; align-items: center; gap: 12px; user-select: none; }
summary::-webkit-details-marker { display: none; }
.chev { font-size: 10px; color: #475569; display: inline-block;
        transition: transform 0.15s; flex-shrink: 0; }
details[open] .chev { transform: rotate(90deg); }
details[open] { border: 1px solid #334155; }
.dbody { padding: 4px 18px 20px; }
.nav-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.npill { padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 500;
         text-decoration: none; border: 1px solid #334155; color: #94a3b8;
         background: #1e293b; }
.npill.ok  { border-color: #14532d; color: #4ade80; }
.npill.bad { border-color: #7f1d1d; color: #f87171; }
.divider { border: none; border-top: 1px solid #334155; margin: 16px 0; }
img.chart { max-width: 100%; border-radius: 8px; }
.disclaimer { font-size: 11px; color: #334155; text-align: center;
              padding: 16px 0 8px; }
"""


# ---------------------------------------------------------------------------
# Per-patient section builders (multi-patient report)
# ---------------------------------------------------------------------------

def _prob_bars_html(row: dict) -> str:
    try:
        from clinmriqc.artifacts import SCALED_THRESHOLDS
    except ImportError:
        SCALED_THRESHOLDS = {}

    detected = set(d for d in (row.get('artifacts_detected') or '').split('|') if d)
    html = ''
    for cls in ARTIFACT_CLASSES:
        raw = _val(row, f'prob_{cls}')
        if raw == '':
            continue
        p      = float(raw)
        is_det = cls in detected
        colour = '#ef4444' if is_det else ('#f97316' if p >= 0.5 else '#475569')
        thr    = SCALED_THRESHOLDS.get(cls)
        thr_marker = (
            f'<div class="prob-thr" style="left:{thr*100:.1f}%"'
            f' title="threshold {thr:.2f}"></div>'
            if thr is not None else ''
        )
        html += f'''
        <div class="prob-row">
          <span class="prob-lbl">{cls.replace("_"," ")}</span>
          <div class="prob-bg">
            <div class="prob-fill" style="width:{p*100:.1f}%;background:{colour}"></div>
            {thr_marker}
          </div>
          <span class="prob-val" style="color:{colour}">{p:.2f}</span>
        </div>'''
    return html


def _iqm_boxes_html(row: dict) -> str:
    efc_raw = _val(row, 'iqm_motion_blur_score')
    snr_raw = _val(row, 'iqm_snr')
    if efc_raw == '' and snr_raw == '':
        return ''

    html = '<div class="metric-grid">'
    for key, raw in [('motion_blur_score', efc_raw), ('snr', snr_raw)]:
        if raw == '':
            continue
        v        = float(raw)
        info     = IQM_RANGES[key]
        lo, hi   = info['normal']
        warn_thr = info['warning']
        if lo <= v <= hi:
            scls, slbl = 'metric-ok',   '✓ Within normal range'
        elif v >= warn_thr:
            scls, slbl = 'metric-warn', '⚠ Outside normal range'
        else:
            scls, slbl = 'metric-bad',  '✗ Below warning threshold'
        html += f'''
        <div class="metric-box">
          <div class="metric-val">{v:.4f}</div>
          <div class="metric-lbl">{info['label']}</div>
          <div class="{scls}" style="margin-bottom:8px">{slbl}</div>
          {_iqm_gauge_html(key, v)}
          <div class="iqm-note">{info['note']}</div>
        </div>'''
    html += '</div>'
    return html


def _artifacts_section_html(row: dict) -> str:
    if _val(row, 'artifacts_quality_passed') == '':
        return ''

    try:
        from clinmriqc.artifacts import SCALED_THRESHOLDS
    except ImportError:
        SCALED_THRESHOLDS = {}

    passed   = str(row['artifacts_quality_passed']).lower() in ('true', '1')
    detected = [d for d in (row.get('artifacts_detected') or '').split('|') if d]
    scores   = {cls: float(_val(row, f'prob_{cls}', '0'))
                for cls in ARTIFACT_CLASSES if _val(row, f'prob_{cls}') != ''}

    verdict_cls  = 'pass' if passed else 'fail'
    verdict_text = ('✓ No artifacts detected' if passed
                    else f'✗ {len(detected)} artifact{"s" if len(detected)>1 else ""} detected')

    tag_html = ''
    for art in detected:
        p = scores.get(art, 0)
        lbl, col = severity_label(p)
        tag_html += (f'<span class="art-tag" style="background:{col}22;color:{col}">'
                     f'{art.replace("_"," ")} · {lbl} ({p:.2f})</span>')

    rec_html = ''
    if detected:
        rec_html = '<div class="divider"></div>'
        rec_html += ('<div style="font-size:10px;font-weight:700;letter-spacing:.1em;'
                     'text-transform:uppercase;color:#475569;margin-bottom:10px">'
                     'Recommended actions</div>')
        for art in detected:
            p = scores.get(art, 0)
            lbl, col = severity_label(p)
            title, body = RECOMMENDATIONS.get(art, (art, ''))
            rec_html += f'''
            <div class="rec" style="border-color:{col}">
              <div class="rec-title" style="color:{col}">
                {title}
                <span style="font-weight:400;font-size:11px;margin-left:8px;opacity:.8">
                  {lbl} · score {p:.2f}
                </span>
              </div>
              <div class="rec-body">{body}</div>
            </div>'''

    thr_note = ''
    if SCALED_THRESHOLDS:
        thr_note = ('<div style="font-size:11px;color:#334155;margin-bottom:10px">'
                    'Yellow mark = per-class detection threshold (P99 of 937-scan dataset)</div>')

    iqm_html = _iqm_boxes_html(row)

    return f'''
    <div class="mod art">
      <div class="mod-title">Artifact Detection</div>
      <span class="verdict {verdict_cls}">{verdict_text}</span>
      {"<div style='margin-top:10px'>" + tag_html + "</div>" if tag_html else ""}
      <div class="divider"></div>
      <div class="grid2" style="gap:24px">
        <div>
          <div style="font-size:10px;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;color:#475569;margin-bottom:10px">
            Scaled severity per class
          </div>
          {thr_note}
          {_prob_bars_html(row)}
        </div>
        <div>
          <div style="font-size:10px;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;color:#475569;margin-bottom:10px">
            Image quality metrics
          </div>
          {iqm_html or '<span style="color:#334155;font-size:12px">Not available</span>'}
        </div>
      </div>
      {rec_html}
    </div>'''


def _contrast_section_html(row: dict) -> str:
    if _val(row, 'contrast_enhanced') == '':
        return ''

    enhanced = str(row['contrast_enhanced']).lower() in ('true', '1')
    vr       = _val(row, 'contrast_vessel_ratio')
    bvf      = _val(row, 'contrast_bright_voxel_fraction')
    vcls     = 'fail' if enhanced else 'pass'
    vtext    = '⚠ Contrast enhancement detected' if enhanced else '✓ No contrast enhancement'

    alert = ''
    if enhanced:
        alert = '''<div style="margin-top:12px;font-size:12px;color:#f87171;
                   background:#450a0a33;padding:10px 14px;border-radius:8px;line-height:1.6">
          Gadolinium enhancement is present. Ensure downstream pipelines account for this,
          or use a pre-contrast scan for structural analysis.
        </div>'''

    return f'''
    <div class="mod con">
      <div class="mod-title">Contrast Enhancement</div>
      <span class="verdict {vcls}">{vtext}</span>
      <div class="metric-grid" style="margin-top:14px">
        <div class="metric-box">
          <div class="metric-val">{_safe_float(vr, ".3f")}</div>
          <div class="metric-lbl">Vessel Intensity Ratio (P99.9 / P50)</div>
          <div style="font-size:11px;color:#475569;margin-top:6px">
            Native T1w: 1.2–1.4 &nbsp;·&nbsp; Post-gadolinium: 1.6–2.0
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{_safe_float(bvf, ".4f")}</div>
          <div class="metric-lbl">Bright Voxel Fraction</div>
          <div style="font-size:11px;color:#475569;margin-top:6px">
            Vessels occupy ~0.3–1% of brain volume after contrast
          </div>
        </div>
      </div>
      {alert}
    </div>'''


def _coreg_section_html(row: dict) -> str:
    if _val(row, 'coreg_flag') == '':
        return ''

    flag        = row['coreg_flag']
    col         = _FLAG_COLOURS.get(flag, '#94a3b8')
    ssim        = _val(row, 'coreg_ssim')
    ncc         = _val(row, 'coreg_ncc')
    ssim_passed = str(_val(row, 'coreg_ssim_passed', 'False')).lower() in ('true', '1')
    ncc_passed  = str(_val(row, 'coreg_ncc_passed',  'False')).lower() in ('true', '1')

    flag_text = {
        'GREEN':  '✓ Registration passed — both metrics above threshold',
        'YELLOW': '⚠ Registration marginal — one metric below threshold',
        'RED':    '✗ Registration failed — review required',
    }.get(flag, flag)
    vcls = {'GREEN': 'pass', 'YELLOW': 'warn', 'RED': 'fail'}.get(flag, 'warn')

    alert = ''
    if flag == 'RED':
        alert = '''<div style="margin-top:12px;font-size:12px;color:#f87171;
                   background:#450a0a33;padding:10px 14px;border-radius:8px;line-height:1.6">
          Both metrics below threshold. Re-run with a more robust algorithm or
          check source images for significant pathology.
        </div>'''
    elif flag == 'YELLOW':
        alert = '''<div style="margin-top:12px;font-size:12px;color:#fbbf24;
                   background:#451a0333;padding:10px 14px;border-radius:8px;line-height:1.6">
          One metric is below threshold. Visually inspect before downstream analysis.
        </div>'''

    return f'''
    <div class="mod reg">
      <div class="mod-title">Registration QC</div>
      <span class="verdict {vcls}">{flag_text}</span>
      <div class="metric-grid" style="margin-top:14px">
        <div class="metric-box">
          <div class="metric-val">{_safe_float(ssim, ".4f")}</div>
          <div class="metric-lbl">SSIM &nbsp;(threshold ≥ 0.70)</div>
          <div class="{"metric-ok" if ssim_passed else "metric-bad"}" style="margin-top:6px">
            {"✓ Pass" if ssim_passed else "✗ Fail"}
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{_safe_float(ncc, ".4f")}</div>
          <div class="metric-lbl">NCC &nbsp;(threshold ≥ 0.80)</div>
          <div class="{"metric-ok" if ncc_passed else "metric-bad"}" style="margin-top:6px">
            {"✓ Pass" if ncc_passed else "✗ Fail"}
          </div>
        </div>
      </div>
      {alert}
    </div>'''


def _metaqc_section_html(row: dict) -> str:
    status = _val(row, 'metaqc_status')
    if status == '':
        return ''

    col  = _STATUS_COLOURS.get(status, '#94a3b8')
    vcls = _STATUS_CLS.get(status, 'warn')
    status_text = {'pass': '✓ Metadata & features OK',
                   'warning': '⚠ Warnings detected',
                   'fail': '✗ One or more checks failed'}.get(status, status)

    ff_raw  = _val(row, 'metaqc_foreground_fraction')
    mean_raw = _val(row, 'metaqc_intensity_mean')
    std_raw  = _val(row, 'metaqc_intensity_std')
    com_raw  = _val(row, 'metaqc_centroid_offset_mm')
    meta_st  = _val(row, 'metaqc_metadata_status')

    def _ff_cls(ff_str):
        try:
            ff = float(ff_str)
            if ff < 0.05: return 'metric-bad',  f'✗ Very low ({ff:.1%}) — possible empty volume'
            if ff < 0.10: return 'metric-warn', f'⚠ Low ({ff:.1%}) — check coverage'
            return 'metric-ok', f'✓ {ff:.1%}'
        except (TypeError, ValueError):
            return 'metric-warn', '—'

    def _com_cls(com_str):
        try:
            v = float(com_str)
            if v > 30: return 'metric-warn', f'⚠ {v:.1f} mm — brain off-centre?'
            return 'metric-ok', f'✓ {v:.1f} mm'
        except (TypeError, ValueError):
            return 'metric-warn', '—'

    ff_cls_name, ff_msg   = _ff_cls(ff_raw)
    com_cls_name, com_msg = _com_cls(com_raw)

    meta_badge = ''
    if meta_st:
        mc  = _STATUS_COLOURS.get(meta_st, '#94a3b8')
        meta_badge = (f'<div class="metric-ok" style="color:{mc};margin-top:4px">'
                      f'{meta_st.upper()}</div>')

    reasons_raw = _val(row, 'metaqc_reasons')
    reasons_html = ''
    if reasons_raw and reasons_raw != '':
        items = [r.strip() for r in reasons_raw.split('|') if r.strip()]
        if items:
            li = ''.join(f'<li style="color:#94a3b8;margin-bottom:4px">{r}</li>' for r in items)
            reasons_html = f'''
            <div class="divider"></div>
            <div style="font-size:10px;font-weight:700;letter-spacing:.1em;
                        text-transform:uppercase;color:#475569;margin-bottom:8px">
              Flagged checks
            </div>
            <ul style="padding-left:16px;font-size:12px;line-height:1.6">{li}</ul>'''

    return f'''
    <div class="mod meta">
      <div class="mod-title">Metadata &amp; Image Features</div>
      <span class="verdict {vcls}">{status_text}</span>
      <div class="metric-grid" style="margin-top:14px">
        <div class="metric-box">
          <div class="metric-val">{_safe_float(ff_raw, ".3f")}</div>
          <div class="metric-lbl">Foreground Fraction</div>
          <div class="{ff_cls_name}" style="margin-top:5px">{ff_msg}</div>
          <div style="font-size:11px;color:#334155;margin-top:4px">
            Normal ≥ 0.10 · Warn &lt; 0.10 · Fail &lt; 0.05
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{_safe_float(mean_raw, ".1f")}</div>
          <div class="metric-lbl">Mean Intensity (brain)</div>
          <div style="font-size:11px;color:#64748b;margin-top:4px">
            std {_safe_float(std_raw, ".1f")}
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{_safe_float(com_raw, ".1f")} mm</div>
          <div class="metric-lbl">Centroid Offset</div>
          <div class="{com_cls_name}" style="margin-top:5px">{com_msg}</div>
          <div style="font-size:11px;color:#334155;margin-top:4px">
            Distance from intensity centroid to geometric centre
          </div>
        </div>
        <div class="metric-box">
          <div class="metric-val" style="font-size:14px;color:{_STATUS_COLOURS.get(meta_st, "#94a3b8")}">
            Header QC
          </div>
          <div class="metric-lbl">Voxel size · Anisotropy</div>
          {meta_badge}
        </div>
      </div>
      {reasons_html}
    </div>'''


def _patient_full_html(row: dict, collapsed: bool = True) -> str:
    pid      = row.get('patient_id', 'unknown')
    ts       = row.get('timestamp', '')
    has_art  = _val(row, 'artifacts_quality_passed') != ''
    has_con  = _val(row, 'contrast_enhanced') != ''
    has_reg  = _val(row, 'coreg_flag') != ''
    has_meta = _val(row, 'metaqc_status') != ''
    passed   = str(row.get('artifacts_quality_passed', '')).lower() in ('true', '1')
    detected = [d for d in (row.get('artifacts_detected') or '').split('|') if d]

    if has_art:
        scls, stxt = ('pass', 'PASS') if passed else ('fail', 'FAIL')
    elif has_meta:
        ms = _val(row, 'metaqc_status')
        scls = _STATUS_CLS.get(ms, 'warn')
        stxt = ms.upper() if ms else 'N/A'
    elif has_reg:
        flag = row.get('coreg_flag', '')
        scls = {'GREEN': 'pass', 'YELLOW': 'warn', 'RED': 'fail'}.get(flag, 'warn')
        stxt = flag
    else:
        scls, stxt = 'warn', 'N/A'

    mod_html = ''.join(
        f'<span class="badge {"badge-on" if active else "badge-off"}">{lbl}</span>'
        for lbl, active in [('Artifact detection', has_art),
                             ('Contrast enhancement', has_con),
                             ('Registration QC', has_reg),
                             ('Metadata QC', has_meta)]
    )

    body = f'''
    <div style="margin-bottom:16px">{mod_html}</div>
    {_artifacts_section_html(row)}
    {_contrast_section_html(row)}
    {_coreg_section_html(row)}
    {_metaqc_section_html(row)}
    <div style="font-size:11px;color:#334155;margin-top:12px">
      Results are classifier-based estimates. Review alongside visual inspection.
    </div>'''

    if collapsed:
        det_str = (', '.join(detected)
                   if detected else '')
        return f'''
    <div id="pt-{pid}">
      <details>
        <summary>
          <span class="chev">▶</span>
          <span style="font-size:14px;font-weight:600;color:#f1f5f9;flex:1">{pid}</span>
          <span class="verdict {scls}" style="margin-top:0;padding:3px 12px;font-size:12px">{stxt}</span>
          {"<span style='font-size:12px;color:#f87171;margin-left:8px'>" + det_str + "</span>" if det_str else ""}
          <span style="font-size:11px;color:#334155;margin-left:12px">{ts}</span>
        </summary>
        <div class="dbody">{body}</div>
      </details>
    </div>'''
    else:
        scan = row.get('scan_path', '')
        return f'''
    <div id="pt-{pid}" class="card">
      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
        <h1>{pid}</h1>
        <span class="verdict {scls}" style="margin-top:0">{stxt}</span>
      </div>
      <div class="meta" style="margin-top:6px">{scan} &nbsp;·&nbsp; {ts}</div>
      <div style="margin-top:12px;margin-bottom:4px">{mod_html}</div>
      <hr class="divider">
      {body}
    </div>'''


# ---------------------------------------------------------------------------
# Aggregate charts
# ---------------------------------------------------------------------------

def _plot_artifact_prevalence(rows: list) -> str:
    counts = {cls: 0 for cls in ARTIFACT_CLASSES}
    n = len(rows)
    for row in rows:
        detected = set(d for d in (row.get('artifacts_detected') or '').split('|') if d)
        for cls in ARTIFACT_CLASSES:
            if cls in detected:
                counts[cls] += 1

    labels  = [c.replace('_', ' ').title() for c in ARTIFACT_CLASSES]
    fracs   = [counts[c] / n if n else 0 for c in ARTIFACT_CLASSES]
    colours = ['#ef4444' if f > 0.3 else '#f97316' if f > 0.1 else '#475569' for f in fracs]

    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    bars = ax.bar(labels, fracs, color=colours, width=0.55, edgecolor='none')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of patients', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#cbd5e1', labelsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, frac in zip(bars, fracs):
        if frac > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.02,
                    f'{frac:.0%}', ha='center', color='#f1f5f9', fontsize=8.5)
    fig.tight_layout(pad=0.6)
    return _fig_to_b64(fig)


def _plot_iqm_scatter(rows: list) -> str:
    pts = []
    for row in rows:
        efc_v = _val(row, 'iqm_motion_blur_score')
        snr_v = _val(row, 'iqm_snr')
        if efc_v == '' or snr_v == '':
            continue
        passed = str(row.get('artifacts_quality_passed', '')).lower() in ('true', '1')
        pts.append((float(efc_v), float(snr_v), passed, row.get('patient_id', '')[:8]))

    if not pts:
        return ''

    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    lo_efc, hi_efc = IQM_RANGES['motion_blur_score']['normal']
    lo_snr, hi_snr = IQM_RANGES['snr']['normal']
    ax.axvspan(lo_efc, hi_efc, alpha=0.10, color='#22c55e')
    ax.axhspan(lo_snr, hi_snr, alpha=0.10, color='#3b82f6')

    for efc_v, snr_v, passed, pid in pts:
        col = '#4ade80' if passed else '#f87171'
        ax.scatter(efc_v, snr_v, color=col, s=55, zorder=3, edgecolors='none')
        ax.text(efc_v, snr_v + 0.015, pid, fontsize=6.5, color='#94a3b8',
                ha='center', va='bottom')

    ax.set_xlabel('Motion / Blur Score (EFC)', color='#94a3b8', fontsize=9)
    ax.set_ylabel('SNR', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#cbd5e1', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ade80',
               markersize=7, label='Pass'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f87171',
               markersize=7, label='Fail'),
    ], fontsize=8, framealpha=0.15, labelcolor='#cbd5e1',
       facecolor='#1e293b', edgecolor='#334155')
    fig.tight_layout(pad=0.5)
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Multi-patient HTML entry point
# ---------------------------------------------------------------------------

def generate_html_from_csv(csv_path: str, output_path: str = None) -> str:
    """Generate a professional HTML QC report from a multi-patient CSV.

    Single patient → expanded per-patient report.
    Multiple patients → summary dashboard + collapsible per-patient sections.

    Parameters
    ----------
    csv_path    : path to the CSV produced by append_csv_record().
    output_path : where to write the HTML.  Defaults to csv_path with .html.

    Returns
    -------
    str — path to the saved HTML file.
    """
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix('.html')

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f'No rows found in {csv_path}')

    n_total = len(rows)
    n_pass  = sum(1 for r in rows
                  if str(r.get('artifacts_quality_passed', '')).lower() in ('true', '1'))
    n_fail  = n_total - n_pass
    now     = datetime.now().strftime('%Y-%m-%d %H:%M')
    single  = n_total == 1

    has_artifacts = any(_val(r, 'artifacts_quality_passed') != '' for r in rows)
    has_contrast  = any(_val(r, 'contrast_enhanced') != '' for r in rows)
    has_coreg     = any(_val(r, 'coreg_flag') != '' for r in rows)
    has_meta      = any(_val(r, 'metaqc_status') != '' for r in rows)

    mod_badges = ''.join(
        f'<span class="badge {"badge-on" if active else "badge-off"}"'
        f' style="margin-right:6px">{lbl}</span>'
        for lbl, active in [('Artifact detection', has_artifacts),
                             ('Contrast enhancement', has_contrast),
                             ('Registration QC', has_coreg),
                             ('Metadata QC', has_meta)]
    )

    stats_html = '<div class="stat-row">'
    stats_html += (f'<div><div class="stat-val">{n_total}</div>'
                   f'<div class="stat-lbl">Total scans</div></div>')
    if has_artifacts:
        stats_html += (f'<div><div class="stat-val" style="color:#4ade80">{n_pass}</div>'
                       f'<div class="stat-lbl">Passed</div></div>')
        stats_html += (f'<div><div class="stat-val" style="color:#f87171">{n_fail}</div>'
                       f'<div class="stat-lbl">Failed</div></div>')
        if not single:
            pct = f'{n_pass / n_total:.0%}'
            stats_html += (f'<div><div class="stat-val" style="color:#4ade80">{pct}</div>'
                           f'<div class="stat-lbl">Pass rate</div></div>')
    stats_html += '</div>'

    aggregate_html = ''
    if not single and has_artifacts:
        prev_b64 = _plot_artifact_prevalence(rows)
        iqm_b64  = _plot_iqm_scatter(rows)
        aggregate_html = '<div class="grid2">'
        if prev_b64:
            aggregate_html += f'''
            <div class="card">
              <h2>Artifact Prevalence Across Cohort</h2>
              <img class="chart" src="data:image/png;base64,{prev_b64}"
                   alt="Artifact prevalence bar chart">
              <div style="font-size:11px;color:#334155;margin-top:8px">
                Fraction of patients where each artifact class was detected.
                Red ≥ 30% · Orange ≥ 10% · Grey below 10%.
              </div>
            </div>'''
        if iqm_b64:
            lo_efc, hi_efc = IQM_RANGES['motion_blur_score']['normal']
            lo_snr, hi_snr = IQM_RANGES['snr']['normal']
            aggregate_html += f'''
            <div class="card">
              <h2>IQM Distribution</h2>
              <img class="chart" src="data:image/png;base64,{iqm_b64}"
                   alt="EFC vs SNR scatter">
              <div style="font-size:11px;color:#334155;margin-top:8px">
                Each point = one patient. Shaded bands = normal ranges
                (EFC {lo_efc}–{hi_efc}, SNR {lo_snr}–{hi_snr}).
              </div>
            </div>'''
        aggregate_html += '</div>'

    summary_table_html = ''
    if not single:
        cols = [('patient_id', 'Patient'), ('timestamp', 'Timestamp')]
        if has_artifacts:
            cols += [('artifacts_quality_passed', 'Status'),
                     ('artifacts_detected',       'Artifacts detected'),
                     ('iqm_motion_blur_score',    'EFC'),
                     ('iqm_snr',                  'SNR')]
        if has_contrast:
            cols += [('contrast_enhanced',    'Contrast'),
                     ('contrast_vessel_ratio', 'Vessel ratio')]
        if has_coreg:
            cols += [('coreg_flag', 'Reg. flag'),
                     ('coreg_ssim', 'SSIM'),
                     ('coreg_ncc',  'NCC')]
        if has_meta:
            cols += [('metaqc_status',              'Meta QC'),
                     ('metaqc_foreground_fraction', 'Foreground')]

        th_html = ''.join(f'<th>{lbl}</th>' for _, lbl in cols)
        tr_html = ''
        for row in rows:
            passed = str(row.get('artifacts_quality_passed', '')).lower() in ('true', '1')
            cells  = ''
            for col, _ in cols:
                v = row.get(col, '')
                if col == 'artifacts_quality_passed':
                    c = '#4ade80' if passed else '#f87171'
                    v = f'<strong style="color:{c}">{"PASS" if passed else "FAIL"}</strong>'
                elif col == 'coreg_flag':
                    c = _FLAG_COLOURS.get(v, '#94a3b8')
                    v = f'<span style="color:{c};font-weight:600">{v}</span>'
                elif col == 'contrast_enhanced':
                    c = '#f87171' if str(v).lower() in ('true', '1') else '#4ade80'
                    v = f'<span style="color:{c}">{"Yes" if str(v).lower() in ("true","1") else "No"}</span>'
                elif col == 'metaqc_status':
                    c = _STATUS_COLOURS.get(str(v).lower(), '#94a3b8')
                    v = f'<span style="color:{c};font-weight:600">{str(v).upper()}</span>'
                elif col == 'metaqc_foreground_fraction':
                    v = _safe_float(v, '.3f') if v else '—'
                elif col == 'patient_id':
                    v = f'<a href="#pt-{v}" style="color:#7c3aed;text-decoration:none">{v}</a>'
                cells += f'<td>{v}</td>'
            tr_html += f'<tr>{cells}</tr>'

        summary_table_html = f'''
        <div class="card">
          <h2>Overview — all patients</h2>
          <div style="overflow-x:auto">
            <table><thead><tr>{th_html}</tr></thead>
                   <tbody>{tr_html}</tbody></table>
          </div>
        </div>'''

    patients_header_html = ''
    if not single:
        pills = ''.join(
            f'<a class="npill {"ok" if str(r.get("artifacts_quality_passed","")).lower() in ("true","1") else "bad"}"'
            f' href="#pt-{r.get("patient_id","")}">{r.get("patient_id","")}</a>'
            for r in rows
        )
        patients_header_html = f'''
        <div class="card">
          <h2>Per-patient details</h2>
          <div style="font-size:12px;color:#64748b;margin-bottom:12px">
            Click a patient ID to expand their full QC report.
          </div>
          <div class="nav-pills">{pills}</div>
        </div>'''

    patient_sections = '\n'.join(
        _patient_full_html(row, collapsed=not single) for row in rows
    )

    title = (rows[0].get('patient_id', csv_path.stem) if single
             else f'ClinMRI-QC Batch Report — {n_total} patients')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{_CSS_REPORT}</style>
</head>
<body>
<div class="page">

<div class="card">
  <h1>ClinMRI-QC {"Report" if single else "Batch Report"}</h1>
  <div class="meta">
    {"Patient: <strong style='color:#cbd5e1'>" + rows[0].get("patient_id","") + "</strong> &nbsp;|&nbsp; " if single else ""}
    Generated: {now} &nbsp;|&nbsp; Source: {csv_path.name} &nbsp;|&nbsp; KCL BMEIS
  </div>
  {stats_html}
  <div style="margin-top:14px">{mod_badges}</div>
</div>

{aggregate_html}

{summary_table_html}

{patients_header_html}

{patient_sections}

<div class="disclaimer">
  ClinMRI-QC · KCL BMEIS &nbsp;·&nbsp;
  Results are classifier-based estimates trained on simulated data.
  All findings should be reviewed alongside visual inspection of the source images.
</div>

</div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    n_str = f'{n_total} patient{"s" if n_total != 1 else ""}'
    print(f'Report saved to {output_path}  ({n_str}, {n_pass} pass / {n_fail} fail)')
    return str(output_path)
