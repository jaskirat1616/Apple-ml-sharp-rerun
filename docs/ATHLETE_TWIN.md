# Splatline Athlete Twin

Splatline Athlete Twin turns sport video into a scene-aware 3D movement review:

- 3D athlete skeleton lifted from video pose into Gaussian scene depth
- Frame-by-frame sport-science metrics
- Automatic jump/landing, high-speed, cut, and deceleration review events
- JSON, CSV, event CSV, and Markdown report exports
- Compact movement-evidence JSON with review frames and quality warnings
- Optional Rerun dashboard with 3D scene, skeleton, video overlay, and metric timelines

This feature is designed for coaching, sports science review, and longitudinal athlete comparison. It is not a medical diagnosis tool.

## v2 Tiered Human Pipeline

v2 replaces the v1 per-frame depth-lifting with a tiered human reconstruction pipeline. The tier is selected with `--human-tier skeleton|mesh|both`. v1 scripts keep working unchanged; the v2 pipeline is opt-in.

### Tier 1 — Fast skeleton (MotionBERT)

`--human-tier skeleton` (default)

- **MotionBERT** (MIT, ICCV 2023) for temporal 3D pose lifting. A dual-stream spatio-temporal transformer looks at up to **243 frames at once**, producing smooth, temporally consistent 3D motion instead of independent per-frame lifts. This is the biggest quality upgrade in v2.
- The lifted 3D skeleton is **fused with Gaussian splat depth** for metric scale recovery — the key advantage over pure monocular 3D lifting, which has no scene scale.
- Output: 3D joints and bones inside the reconstructed Gaussian scene, same exports as v1.

### Tier 2 — Full SMPL mesh (HMR 2.0)

`--human-tier mesh`

- **HMR 2.0 / 4DHumans** (MIT) for SMPL body mesh recovery and 3D tracking from video. Produces a textured 3D human body in the scene, not just a skeleton. Supports multi-person reconstruction with identity tracking.
- **PHALP** (MIT) for 3D-aware person tracking — maintains consistent person IDs through occlusion events.
- Requires the SMPL body model (register at [smplify.is.tue.mpg.de](https://smplify.is.tue.mpg.de) and place `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in the HMR 2.0 data directory). See [NOTICE.md](../NOTICE.md).

### Both tiers

`--human-tier both` runs the fast skeleton first, then the SMPL mesh, so you get the quick skeleton review plus the detailed mesh in one pass.

### 2D pose detection

Both tiers start from a 2D pose detector. v2 defaults to **YOLO26-pose** (Ultralytics, 2026) — NMS-free, 72% COCO AP. **RTMPose** (Apache-2.0, MMPose) is available as an AGPL-free alternative for commercial deployments. See [Pose Model Choice](#pose-model-choice) below.

## Quick Start

Analyze a new video end to end (v2 defaults: SHARP backend, skeleton tier):

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 --device mps --pose-model yolo26x-pose.pt --view
```

Analyze with the v2 tiered pipeline — full SMPL mesh via HMR 2.0:

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend sharp \
  --human-tier mesh \
  --device mps \
  --view
```

Run both tiers (skeleton + mesh) in one pass:

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --human-tier both \
  --view
```

Use the MIT-licensed TripoSplat backend (commercial-safe) with the MotionBERT skeleton tier:

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend triposplat \
  --human-tier skeleton \
  --device mps
```

Analyze an existing Splatline conversion:

```bash
python scripts/sports/analyze_athlete_twin.py \
  --frames-dir output_training/frames \
  --gaussians-dir output_training/gaussians \
  --fps 30 \
  --view
```

The command writes:

```text
athlete_twin.json   full frame-by-frame 3D joints, metrics, events, metadata
evidence_summary.json compact event evidence, quality warnings, and key metrics
metrics.csv         one row per analyzed frame
events.csv          movement events and review notes
report.md           coach/scientist summary
```

## Pose Model Choice

Athlete Twin defaults to Ultralytics `yolo26x-pose.pt`, the highest-accuracy YOLO26 COCO-Pose model. Use a smaller YOLO26 pose model when speed or memory matters:

```bash
# Fast edge/dev run
python scripts/sports/analyze_athlete_twin.py training.mp4 --pose-model yolo26n-pose.pt

# Balanced
python scripts/sports/analyze_athlete_twin.py training.mp4 --pose-model yolo26m-pose.pt

# Maximum accuracy
python scripts/sports/analyze_athlete_twin.py training.mp4 --pose-model yolo26x-pose.pt --pose-imgsz 960
```

YOLO26 support requires `ultralytics>=8.4.0`.

## v2 CLI Flags

The v2 pipeline adds two main flags. All v1 flags still work unchanged.

### `--splat-backend {sharp,triposplat,depthsplat,longsplat}`

Selects the 3D reconstruction backend from the [backend registry](SPLAT_MODELS.md). Defaults to `sharp` (non-commercial research). Use `triposplat` for the MIT-licensed, commercial-safe backend. On first run without a saved choice, the backend selector prompts you to pick one; the choice persists in `~/.splatline/config.json`.

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 --splat-backend triposplat
```

The legacy `--sharp-model-url`, `--sharp-checkpoint`, and `--sharp-internal-size` flags still apply when `--splat-backend=sharp`.

### `--human-tier {skeleton,mesh,both}`

Selects the human reconstruction tier. Defaults to `skeleton`.

| Tier | Model | Output | Speed |
| --- | --- | --- | --- |
| `skeleton` | MotionBERT (MIT, ICCV 2023) | 3D joints + bones fused with splat depth | fast |
| `mesh` | HMR 2.0 / 4DHumans + PHALP (MIT) | Textured SMPL body mesh with identity tracking | slower |
| `both` | MotionBERT then HMR 2.0 | Skeleton + mesh in one pass | slowest |

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 --human-tier mesh
```

Run `python scripts/sports/analyze_athlete_twin.py --help` for the full flag list.

## Best Initial Use Cases

Athlete Twin is strongest for:

- Change-of-direction and deceleration review
- Sprint mechanics from side or three-quarter video
- Jump takeoff and landing review
- Return-to-play movement comparison
- Repeated sessions where the same camera setup is used

It is intentionally not specialized for one sport. The first wedge is field and court sport movement: soccer, basketball, football, rugby, lacrosse, tennis, volleyball, and rehab testing.

## Metrics

Frame metrics include:

- Left/right knee angle and flexion proxy
- Left/right hip flexion proxy
- Left/right elbow flexion
- Trunk lean
- Knee alignment deviation proxy
- Pelvis position, horizontal speed, and vertical velocity
- Mean pose confidence
- Frame-level technique flags

Event detectors include:

- `jump_landing`
- `change_of_direction`
- `hard_deceleration`
- `high_speed_window`

## Scaling

The SHARP scene scale is not guaranteed to be metric. By default, Athlete Twin scales the skeleton to `--athlete-height-m 1.75` so speed and distance outputs are more interpretable. For a real athlete, pass their approximate height:

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 --athlete-height-m 1.88
```

To leave native scene units untouched:

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 --no-height-scale
```

## Practical Capture Guidance

Use a stable camera. Keep the full athlete visible through the movement. Avoid heavy occlusion. For cuts and deceleration, film from a three-quarter angle that shows both travel direction and body alignment. For jump landings, make sure feet and hips remain visible.

## Example Review Workflow

1. Run Athlete Twin on a training clip.
2. Open `report.md` for the high-level summary.
3. Sort `events.csv` by event type and review peak frames.
4. Open the Rerun dashboard with `--view` to inspect the 3D skeleton inside the reconstructed scene.
5. Compare `metrics.csv` across sessions for progression or asymmetry changes.
