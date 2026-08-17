"""
Depth estimation using Depth Anything V2.
"""

import os
import sys
import numpy as np
import torch

# Add Depth-Anything-V2 to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DAV2_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), 'Depth-Anything-V2')
if _DAV2_PATH not in sys.path:
    sys.path.insert(0, _DAV2_PATH)

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

_CKPT_DIR = os.environ.get(
    'PANODREAMER_DAV2_CHECKPOINT_DIR',
    os.path.join(os.path.dirname(_SCRIPT_DIR), 'checkpoints'),
)

_model = None
_metric_model = None
_encoder = 'vitl'


def _get_model():
    """Load relative depth model on first use."""
    global _model
    if _model is None:
        checkpoint_path = os.path.join(_CKPT_DIR, f'depth_anything_v2_{_encoder}.pth')
        print(f'[INFO] Loading Depth Anything V2 ({_encoder}) from {checkpoint_path}')
        _model = DepthAnythingV2(**MODEL_CONFIGS[_encoder])
        _model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        _model = _model.to(DEVICE).eval()
    return _model


class _MetricDepthAnythingV2(DepthAnythingV2):
    """Metric variant: Sigmoid head (not ReLU), forward scales by max_depth."""

    def __init__(self, max_depth=20.0, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        head_seq = self.depth_head.scratch.output_conv2
        self.depth_head.scratch.output_conv2 = torch.nn.Sequential(
            head_seq[0], head_seq[1], head_seq[2],
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        return depth.squeeze(1)


def _get_metric_model(dataset='vkitti'):
    """Load metric depth model on first use. dataset: 'vkitti' (outdoor, 80m) or 'hypersim' (indoor, 20m)."""
    global _metric_model

    max_depth = {'vkitti': 80.0, 'hypersim': 20.0}[dataset]
    checkpoint_path = os.path.join(_CKPT_DIR, f'depth_anything_v2_metric_{dataset}_{_encoder}.pth')

    if _metric_model is None or getattr(_metric_model, '_dataset', None) != dataset:
        print(f'[INFO] Loading Depth Anything V2 metric ({dataset}, max_depth={max_depth}) from {checkpoint_path}')
        model = _MetricDepthAnythingV2(max_depth=max_depth, **MODEL_CONFIGS[_encoder])
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(DEVICE).eval()
        model._dataset = dataset
        _metric_model = model
    return _metric_model


def estimate_depth(img, mode='test'):
    """
    Estimate relative depth from an image.

    Args:
        img: PIL Image or numpy array (RGB)
        mode: Unused, kept for compatibility

    Returns:
        depth: HxW numpy array of relative depth values (larger = farther)
    """
    model = _get_model()
    img = np.asarray(img)[:, :, [2, 1, 0]]  # RGB to BGR

    with torch.no_grad():
        depth = model.infer_image(img)

    return depth


def estimate_metric_depth(img, dataset='vkitti'):
    """
    Estimate metric depth from an image.

    Args:
        img: PIL Image or numpy array (RGB)
        dataset: 'vkitti' (outdoor, max 80m) or 'hypersim' (indoor, max 20m)

    Returns:
        depth: HxW numpy array of metric depth in meters (larger = farther)
    """
    model = _get_metric_model(dataset)
    img = np.asarray(img)[:, :, [2, 1, 0]]

    with torch.no_grad():
        depth = model.infer_image(img)

    return depth


_moge_model = None


def _get_moge_model():
    """Load MoGe V2 model on first use."""
    global _moge_model
    if _moge_model is None:
        from moge.model.v2 import MoGeModel
        print('[INFO] Loading MoGe V2 (vitl-normal)')
        _moge_model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').to(DEVICE).eval()
    return _moge_model


def estimate_depth_moge(img):
    """
    Estimate metric depth using MoGe V2.

    Args:
        img: PIL Image or numpy array (RGB, uint8 or float32 0-1)

    Returns:
        depth: HxW numpy array of metric depth in meters
    """
    model = _get_moge_model()
    img = np.asarray(img).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

    with torch.no_grad():
        output = model.infer(img_tensor)

    depth = output['depth'].cpu().numpy()
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return depth


def calibrate_relative_depth(relative, metric, percentile=2):
    """
    Compute scale and bias to map relative depth to metric depth.
    Uses near-far alignment with percentile-based robustness.

    Args:
        relative: HxW relative depth (from estimate_depth)
        metric: HxW metric depth (from estimate_metric_depth), same image
        percentile: percentile for near/far bounds (default 2 = 2nd/98th)

    Returns:
        scale, bias: such that calibrated = scale * relative + bias
    """
    valid = (relative > 0) & (metric > 0) & np.isfinite(relative) & np.isfinite(metric)
    rel_valid = relative[valid]
    met_valid = metric[valid]
    rel_near = np.percentile(rel_valid, percentile)
    rel_far = np.percentile(rel_valid, 100 - percentile)
    met_near = np.percentile(met_valid, percentile)
    met_far = np.percentile(met_valid, 100 - percentile)

    scale = (met_far - met_near) / (rel_far - rel_near + 1e-8)
    bias = met_near - scale * rel_near

    print(f'[INFO] Depth calibration: scale={scale:.4f}, bias={bias:.4f}')
    print(f'[INFO]   relative range: [{rel_near:.2f}, {rel_far:.2f}] -> metric range: [{met_near:.2f}m, {met_far:.2f}m]')

    return scale, bias
