"""
Splatline v2 configuration and first-run backend selection.

Persists user choices (reconstruction backend, human pipeline tier, device)
to ~/.splatline/config.json so they survive across sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path.home() / ".splatline"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Human pipeline tiers
TIER_SKELETON = "skeleton"  # Fast: 2D pose → MotionBERT 3D lift + depth fusion
TIER_MESH = "mesh"          # Detailed: HMR 2.0 SMPL mesh + 3D tracking
TIER_BOTH = "both"          # Run both tiers

VALID_TIERS = {TIER_SKELETON, TIER_MESH, TIER_BOTH}


@dataclass
class SplatlineConfig:
    """Persistent user configuration for Splatline v2."""

    splat_backend: str = "sharp"
    human_tier: str = TIER_SKELETON
    device: str = "default"
    pose_model: str = "yolo26x-pose.pt"
    athlete_height_m: float = 1.75
    sharp_internal_size: int = 1536
    first_run_complete: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SplatlineConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def load_config() -> SplatlineConfig:
    """Load config from disk, or return defaults if not found."""
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return SplatlineConfig.from_dict(data)
        except Exception:
            pass
    return SplatlineConfig()


def save_config(config: SplatlineConfig) -> None:
    """Persist config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )


def get_backend_choice(config: Optional[SplatlineConfig] = None) -> str:
    """Return the configured splat backend, prompting on first run if needed."""
    if config is None:
        config = load_config()
    return config.splat_backend


def set_backend_choice(backend: str, config: Optional[SplatlineConfig] = None) -> SplatlineConfig:
    """Set and persist the splat backend choice."""
    if config is None:
        config = load_config()
    config.splat_backend = backend
    config.first_run_complete = True
    save_config(config)
    return config


# --- Backend metadata for UI/CLI display ---

BACKEND_CHOICES = [
    {
        "key": "sharp",
        "name": "Apple SHARP",
        "license": "apple-amlr (non-commercial research only)",
        "commercial_ok": False,
        "description": "Fast feed-forward monocular 3DGS from a single image. "
                       "Metric scale, OpenCV coordinates, direct .ply output. "
                       "~2.5GB model download.",
        "source_url": "https://github.com/apple/ml-sharp",
    },
    {
        "key": "triposplat",
        "name": "TripoSplat",
        "license": "MIT (fully open-source, commercial OK)",
        "commercial_ok": True,
        "description": "Single-image 3D Gaussians with learned density control. "
                       "SIGGRAPH 2026. Lightweight (~2000 LOC).",
        "source_url": "https://github.com/VAST-AI-Research/TripoSplat",
    },
    {
        "key": "depthsplat",
        "name": "DepthSplat",
        "license": "MIT (fully open-source, commercial OK)",
        "commercial_ok": True,
        "description": "Multi-view depth + Gaussian splatting. CVPR 2025. "
                       "Best for keyframe groups with overlapping views.",
        "source_url": "https://github.com/cvg/depthsplat",
    },
    {
        "key": "longsplat",
        "name": "LongSplat",
        "license": "NVlabs (check terms)",
        "commercial_ok": False,
        "description": "Video-native: coherent 3DGS from full video with MASt3R "
                       "pose estimation. ICCV 2025. Temporal coherence.",
        "source_url": "https://github.com/NVlabs/LongSplat",
    },
]

HUMAN_TIER_CHOICES = [
    {
        "key": TIER_SKELETON,
        "name": "Skeleton (fast)",
        "description": "YOLO26-pose 2D → MotionBERT temporal 3D lifting + "
                       "Gaussian depth fusion. Smooth, metric-scale 3D skeleton.",
    },
    {
        "key": TIER_MESH,
        "name": "SMPL mesh (detailed)",
        "description": "HMR 2.0 / 4DHumans SMPL body mesh recovery + PHALP "
                       "3D tracking. Textured 3D human body in scene.",
    },
    {
        "key": TIER_BOTH,
        "name": "Both (tiered)",
        "description": "Run skeleton first, then SMPL mesh. Most complete "
                       "but slower.",
    },
]
