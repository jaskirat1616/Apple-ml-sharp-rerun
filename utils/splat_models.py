"""
Splat model backend registry and SHARP conversion helpers.

Splatline's downstream tools expect per-frame 3D Gaussian `.ply` files in an
OpenCV-like camera coordinate system.  Apple SHARP is currently the only
implemented backend here because it provides that contract directly from a
single image on CPU/CUDA/MPS.  Newer multi-view research models are tracked in
the registry so they can be integrated behind the same interface when their
runtime and output contracts are made compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SHARP_DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@dataclass(frozen=True)
class SplatBackendInfo:
    """Metadata for a splat reconstruction backend."""

    key: str
    name: str
    status: str
    license: str
    commercial_ok: bool
    input_mode: str
    output_contract: str
    recommended_for: str
    notes: str
    source_url: str


SPLAT_BACKENDS: Dict[str, SplatBackendInfo] = {
    "sharp": SplatBackendInfo(
        key="sharp",
        name="Apple SHARP",
        status="implemented",
        license="apple-amlr (non-commercial research only)",
        commercial_ok=False,
        input_mode="single image / extracted video frames",
        output_contract="3DGS PLY, OpenCV coordinates, metric scale",
        recommended_for="research default — fast monocular 3DGS from single image",
        notes="Fast feed-forward monocular 3DGS; direct PLY output. Non-commercial license.",
        source_url="https://github.com/apple/ml-sharp",
    ),
    "triposplat": SplatBackendInfo(
        key="triposplat",
        name="TripoSplat",
        status="implemented",
        license="MIT (fully open-source, commercial OK)",
        commercial_ok=True,
        input_mode="single image",
        output_contract="3DGS PLY, variable Gaussian count (up to 262K)",
        recommended_for="open-source default — MIT-licensed SHARP alternative",
        notes="SIGGRAPH 2026. Learned density control, ~2000 LOC, pip-installable.",
        source_url="https://github.com/VAST-AI-Research/TripoSplat",
    ),
    "depthsplat": SplatBackendInfo(
        key="depthsplat",
        name="DepthSplat",
        status="research_reference",
        license="MIT (fully open-source, commercial OK)",
        commercial_ok=True,
        input_mode="sparse multi-view images (up to 12 views)",
        output_contract="3DGS PLY possible through project configs",
        recommended_for="multi-view keyframe groups with overlapping coverage",
        notes="CVPR 2025. Connects depth estimation and Gaussian splatting. MIT license.",
        source_url="https://github.com/cvg/depthsplat",
    ),
    "longsplat": SplatBackendInfo(
        key="longsplat",
        name="LongSplat",
        status="research_reference",
        license="NVlabs (check terms)",
        commercial_ok=False,
        input_mode="casual long videos (unposed)",
        output_contract="3DGS PLY via converter script",
        recommended_for="video-native temporal coherence — single coherent scene from full video",
        notes="ICCV 2025. MASt3R pose estimation + adaptive octree anchoring. Memory-efficient.",
        source_url="https://github.com/NVlabs/LongSplat",
    ),
    "splinegs": SplatBackendInfo(
        key="splinegs",
        name="SplineGS",
        status="research_reference",
        license="MIT",
        commercial_ok=True,
        input_mode="monocular video",
        output_contract="dynamic 3D Gaussians",
        recommended_for="real-time dynamic 3DGS from monocular video",
        notes="CVPR 2025. Robust motion-adaptive spline for dynamic scenes.",
        source_url="https://github.com/KAIST-VICLab/SplineGS",
    ),
    "vggt": SplatBackendInfo(
        key="vggt",
        name="VGGT",
        status="research_reference",
        license="Custom (commercial checkpoint available via application)",
        commercial_ok=True,
        input_mode="one to hundreds of views",
        output_contract="camera parameters, depth, point maps, tracks; not native 3DGS PLY",
        recommended_for="geometry bootstrap before Gaussian export",
        notes="CVPR 2025 Best Paper. Feed-forward all 3D attributes in <1 second.",
        source_url="https://github.com/facebookresearch/vggt",
    ),
    "noposplat": SplatBackendInfo(
        key="noposplat",
        name="NoPoSplat",
        status="research_reference",
        license="Not specified",
        commercial_ok=False,
        input_mode="sparse unposed multi-view images",
        output_contract="3D Gaussians in canonical space",
        recommended_for="future pose-free sparse video frame groups",
        notes="ICLR 2025 Oral. Promising for unposed sports clips.",
        source_url="https://noposplat.github.io/",
    ),
    "anysplat": SplatBackendInfo(
        key="anysplat",
        name="AnySplat",
        status="research_reference",
        license="Not specified",
        commercial_ok=False,
        input_mode="unconstrained image collections",
        output_contract="Gaussian, depth, and camera pose heads",
        recommended_for="future uncalibrated multi-view scene reconstruction",
        notes="SIGGRAPH Asia 2025. Uses VGGT as geometry prior.",
        source_url="https://github.com/InternRobotics/AnySplat",
    ),
    "volsplat": SplatBackendInfo(
        key="volsplat",
        name="VolSplat",
        status="research_reference",
        license="Not specified",
        commercial_ok=False,
        input_mode="multi-view images",
        output_contract="voxel-aligned 3D Gaussian predictions",
        recommended_for="future robust multi-view reconstruction research",
        notes="ECCV 2026. Voxel-aligned prediction replacing pixel-aligned paradigm.",
        source_url="https://github.com/ziplab/VolSplat",
    ),
}


def list_splat_backends() -> List[SplatBackendInfo]:
    """Return all known splat backends."""
    return list(SPLAT_BACKENDS.values())


def get_splat_backend(key: str) -> SplatBackendInfo:
    """Return backend metadata by key."""
    try:
        return SPLAT_BACKENDS[key]
    except KeyError as exc:
        known = ", ".join(sorted(SPLAT_BACKENDS))
        raise ValueError(f"Unknown splat backend '{key}'. Known backends: {known}") from exc


def resolve_torch_device(device: str):
    """Resolve default/cuda/mps/cpu into a torch.device."""
    import torch

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device), device


def load_sharp_predictor(
    device: str = "default",
    model_url: str = SHARP_DEFAULT_MODEL_URL,
    checkpoint_path: Optional[Path] = None,
):
    """Load a SHARP predictor and return `(predictor, torch_device, device_name)`."""
    import torch
    from sharp.models import PredictorParams, create_predictor

    device_obj, device_name = resolve_torch_device(device)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device_obj)
    else:
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device_obj)
    return predictor, device_obj, device_name


def predict_sharp_image(
    predictor,
    image: np.ndarray,
    f_px: float,
    device_obj,
    internal_size: int = 1536,
):
    """Predict metric SHARP Gaussians for one RGB image."""
    import torch
    import torch.nn.functional as F
    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (internal_size, internal_size)
    img_pt = torch.from_numpy(image.copy()).float().to(device_obj).permute(2, 0, 1) / 255.0
    _, height, width = img_pt.shape
    disp_factor = torch.tensor([f_px / width]).float().to(device_obj)

    img_resized = F.interpolate(
        img_pt[None],
        size=internal_shape,
        mode="bilinear",
        align_corners=True,
    )
    gaussians_ndc = predictor(img_resized, disp_factor)

    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).float().to(device_obj)
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    return unproject_gaussians(
        gaussians_ndc,
        torch.eye(4).to(device_obj),
        intrinsics_resized,
        internal_shape,
    )


def convert_frames_with_sharp(
    frames_dir: Path,
    output_dir: Path,
    device: str = "default",
    model_url: str = SHARP_DEFAULT_MODEL_URL,
    checkpoint_path: Optional[Path] = None,
    internal_size: int = 1536,
    skip_existing: bool = True,
) -> Path:
    """
    Convert images in `frames_dir` to SHARP 3D Gaussian PLY files.

    Returns the output `gaussians/` directory.
    """
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply

    gaussians_dir = output_dir / "gaussians"
    gaussians_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading SHARP model...")
    if checkpoint_path is not None:
        print(f"  Checkpoint: {checkpoint_path}")
    else:
        print(f"  Model URL: {model_url}")
    predictor, device_obj, device_name = load_sharp_predictor(device, model_url, checkpoint_path)
    print(f"  Device: {device_name}")

    image_paths = []
    for ext in io.get_supported_image_extensions():
        image_paths.extend(frames_dir.glob(f"*{ext}"))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(f"No supported images found in {frames_dir}")

    print(f"  Converting {len(image_paths)} frame(s) with SHARP...")
    for idx, img_path in enumerate(image_paths):
        out_ply = gaussians_dir / f"{img_path.stem}.ply"
        if skip_existing and out_ply.exists():
            continue

        print(f"  [{idx + 1}/{len(image_paths)}] {img_path.name}", end="  ")
        image, _, f_px = io.load_rgb(img_path)
        height, width = image.shape[:2]
        gaussians = predict_sharp_image(
            predictor,
            image,
            f_px,
            device_obj,
            internal_size=internal_size,
        )
        save_ply(gaussians, f_px, (height, width), out_ply)
        print("done")

    return gaussians_dir


def convert_frames_to_splats(
    frames_dir: Path,
    output_dir: Path,
    backend: str = "sharp",
    device: str = "default",
    model_url: str = SHARP_DEFAULT_MODEL_URL,
    checkpoint_path: Optional[Path] = None,
    internal_size: int = 1536,
    skip_existing: bool = True,
) -> Path:
    """Convert frames to per-frame splat PLY files with a supported backend."""
    info = get_splat_backend(backend)
    if info.status != "implemented":
        raise NotImplementedError(
            f"Backend '{backend}' is tracked but not implemented in Splatline yet. "
            f"Reason: {info.notes}"
        )

    if backend == "sharp":
        return convert_frames_with_sharp(
            frames_dir=frames_dir,
            output_dir=output_dir,
            device=device,
            model_url=model_url,
            checkpoint_path=checkpoint_path,
            internal_size=internal_size,
            skip_existing=skip_existing,
        )

    if backend == "triposplat":
        return convert_frames_with_triposplat(
            frames_dir=frames_dir,
            output_dir=output_dir,
            device=device,
            skip_existing=skip_existing,
        )

    raise NotImplementedError(f"No converter registered for backend '{backend}'")


def convert_frames_with_triposplat(
    frames_dir: Path,
    output_dir: Path,
    device: str = "default",
    skip_existing: bool = True,
) -> Path:
    """
    Convert images to 3D Gaussian PLY files using TripoSplat (MIT license).

    TripoSplat is a SIGGRAPH 2026 model that regresses 3D Gaussians from a
    single image with learned density control. Install from:
        git clone https://github.com/VAST-AI-Research/TripoSplat
        pip install -e TripoSplat
    """
    import numpy as np
    from sharp.utils.gaussians import save_ply  # reuse PLY writer for compat

    gaussians_dir = output_dir / "gaussians"
    gaussians_dir.mkdir(parents=True, exist_ok=True)

    device_obj, device_name = resolve_torch_device(device)
    print(f"  Loading TripoSplat model on {device_name}...")

    try:
        import torch
        from triposplat import TripoSplatModel
    except ImportError as exc:
        raise ImportError(
            "TripoSplat is not installed. Install with:\n"
            "  git clone https://github.com/VAST-AI-Research/TripoSplat\n"
            "  pip install -e TripoSplat\n"
            f"Error: {exc}"
        ) from exc

    model = TripoSplatModel.from_pretrained("VAST-AI-Research/TripoSplat")
    model.eval().to(device_obj)

    image_paths = sorted(
        p for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        for p in frames_dir.glob(f"*{ext}")
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {frames_dir}")

    print(f"  Converting {len(image_paths)} frame(s) with TripoSplat...")
    for idx, img_path in enumerate(image_paths):
        out_ply = gaussians_dir / f"{img_path.stem}.ply"
        if skip_existing and out_ply.exists():
            continue

        print(f"  [{idx + 1}/{len(image_paths)}] {img_path.name}", end="  ")

        import cv2
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image_rgb).float().to(device_obj).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            gaussians = model(img_tensor)

        positions = gaussians["positions"].cpu().numpy()
        colors = gaussians["colors"].cpu().numpy()
        scales = gaussians["scales"].cpu().numpy()
        opacities = gaussians["opacities"].cpu().numpy()
        rotations = gaussians["rotations"].cpu().numpy()

        f_px = float(max(image.shape[1], image.shape[0]))
        save_ply(
            {
                "positions": positions,
                "colors": colors,
                "scales": scales,
                "opacities": opacities,
                "rotations": rotations,
            },
            f_px,
            (image.shape[0], image.shape[1]),
            out_ply,
        )
        print("done")

    return gaussians_dir

