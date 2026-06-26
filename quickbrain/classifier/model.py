"""Artifact severity regression model — ResNet50 backbone, 6 independent sigmoid outputs.

Replaces the 7-class softmax classifier with 6 independent regression heads,
one per artifact type. Each head predicts severity in [0, 1]:
    0.0  = absent
    0.33 = mild
    0.67 = moderate
    1.0  = severe

Classes: motion, noise, ghosting, bias_field, gibbs, zipper
         (no 'clean' class — absence is indicated by all severities near 0)
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

REGRESSION_CLASSES = ['motion', 'noise', 'ghosting', 'bias_field', 'gibbs', 'zipper']
N_ARTIFACTS        = len(REGRESSION_CLASSES)
RESIZE_HW          = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_regression_model() -> nn.Module:
    """ResNet50 with 6 independent linear outputs (sigmoid applied at inference).

    BCEWithLogitsLoss is used during training, so sigmoid is NOT in forward().
    At inference, torch.sigmoid() is applied to convert logits to severity scores.
    """
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, N_ARTIFACTS)
    return m


def load_regression_model(checkpoint_path: str, device: str = None) -> nn.Module:
    """Load a trained regression checkpoint and set to eval mode."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_regression_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_volume(
    model: nn.Module,
    volume: np.ndarray,
    class_thresholds: dict = None,
    device: str = None,
) -> dict:
    """Return per-artifact severity scores via sigmoid averaging across axial slices.

    Samples slices from the central 20-80% of the volume, runs each through the
    model, averages sigmoid(logits) across slices. A class is flagged when its
    mean severity exceeds its threshold.

    Parameters
    ----------
    model            : loaded regression model (from load_regression_model)
    volume           : np.ndarray, shape (H, W, D), float — raw voxel intensities
    class_thresholds : per-class severity thresholds. Defaults to 0.5 for any class
                       not specified. Callers should pass DEFAULT_THRESHOLDS from
                       artifacts.py for calibrated behaviour.
    device           : 'cpu' or 'cuda'. Auto-detected if not specified.

    Returns
    -------
    dict with keys:
        artifact_severity  : dict {class_name: mean_severity_score [0,1]}
        artifacts_detected : list of class names whose severity >= threshold
        quality_passed     : bool — True if no artifacts detected
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    effective_thresholds = dict(class_thresholds or {})

    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    volume = np.asarray(volume, dtype=np.float32)
    vol_min, vol_max = volume.min(), volume.max()
    if vol_max > vol_min:
        volume = (volume - vol_min) / (vol_max - vol_min)

    D = volume.shape[2]
    start = int(D * 0.20)
    end   = int(D * 0.80)
    slice_step = max(1, (end - start) // 15)

    severity_accumulator = np.zeros(N_ARTIFACTS, dtype=np.float64)
    n_slices = 0

    with torch.no_grad():
        for i in range(start, end, slice_step):
            sl = volume[:, :, i]
            if sl.shape[0] != RESIZE_HW or sl.shape[1] != RESIZE_HW:
                import torch.nn.functional as F
                sl_t = torch.tensor(sl).unsqueeze(0).unsqueeze(0)
                sl = F.interpolate(sl_t, size=(RESIZE_HW, RESIZE_HW),
                                   mode='bilinear', align_corners=False).squeeze().numpy()
            t = torch.tensor(sl[None]).repeat(3, 1, 1)
            t = norm(t).unsqueeze(0).to(device)
            logits = model(t)                               # (1, 6)
            severity = torch.sigmoid(logits).cpu().numpy()  # (1, 6) in [0, 1]
            severity_accumulator += severity[0]
            n_slices += 1

    mean_severity = severity_accumulator / max(n_slices, 1)

    artifact_severity = {
        cls: float(round(float(s), 4))
        for cls, s in zip(REGRESSION_CLASSES, mean_severity)
    }

    artifacts_detected = [
        cls for cls, s in artifact_severity.items()
        if s >= effective_thresholds.get(cls, 0.5)
    ]

    return {
        "artifact_severity":  artifact_severity,
        "artifacts_detected": artifacts_detected,
        "quality_passed":     len(artifacts_detected) == 0,
    }
