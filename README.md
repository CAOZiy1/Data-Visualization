<div align="center">

# 2023 Tropical Cyclone Best Track – Artistic Multi-Version Visualizations

Multi-version artistic animations of 2023 tropical cyclone (typhoon) tracks using Hong Kong Observatory (HKO) Best Track CSV data. Each generation iterates on a different visual concept: classic track lines, dark gradients, fractal accents, pure abstract forms, energetic spike/spiral fields, and finally a parameterized spiral growth engine (v6).

</div>

---

## Table of Contents
1. Overview  
2. Repository Layout  
3. Installation  
4. Data & Column Specification  
5. Version Overview (v1–v6)  
6. Detailed Version Guides  
7. v6 API & Parameters (`animate_typhoons`)  
8. Performance Tuning  
9. Troubleshooting  
10. Extensibility & Future Ideas  
11. Changelog / Evolution Summary  
12. Example Programmatic Calls  
13. Preview Media  
14. License (Placeholder)  
15. Attribution & Data Notice  

---

## 1. Overview

This project chronicles iterative experimentation with rendering typhoon “best track” data as evolving art. Rather than only plotting geographic paths, each version explores visual storytelling through color theory, fractal perturbation, motion layering, or procedural spiral geometry to express intensity (wind speed) and temporal evolution.

The latest stable feature set resides in **v6**, which exposes a reusable function:  
`animate_typhoons(...)` – allowing external reuse without modifying internal logic (frame pacing, style toggles, track overlay, end-frame holding, etc.).

### Motivation & Evolution Arc
- **v1** – Baseline geographic polyline animation
- **v2** – Dark theme + intensity gradients
- **v3** – Fractal miniatures as symbolic energy blooms
- **v4** – Axis-free abstract aesthetic
- **v5** – Spike/energy geometry emphasizing destructive force
- **v6** – Controlled spiral growth system (deterministic layers + wind‑scaled dynamics)

---

## 2. Repository Layout

```
HKO2023BST.csv                     # Root copy of 2023 dataset
requirements.txt
typhoon_spiral_art.png             # Static conceptual preview

typhoon_data_visualization_v1/
  main_v1.py
  typhoon_tracks_animation_v1_*.gif

typhoon_data_visualization_v2/
  main_v2.py
  typhoon_tracks_animation_v2_*.gif

typhoon_data_visualization_v3/
  main_v3.py
  typhoon_tracks_animation_v3.gif

typhoon_data_visualization_v4/
  main_v4.py
  typhoon_tracks_animation_v4.gif

typhoon_data_visualization_v5/
  main_v5.py
  HKO2023BST.csv
  typhoon_tracks_animation_v5.gif

typhoon_data_visualization_v6/
  main_v6.py
  HKO2023BST.csv
  typhoon_tracks_animation_v6.gif
```

Each subfolder includes a local copy of the CSV for fully standalone experimentation (no fragile relative imports).

---

## 3. Installation

```bash
pip install -r requirements.txt
```

Core dependencies:
- pandas
- numpy
- matplotlib
- pillow  (Pillow writer for GIF export)

Optional (future / custom forks):
- cartopy (geographic coastlines / projections)
- ffmpeg (higher quality MP4 export)
- tqdm (progress output if batch rendering)

---

## 4. Data & Column Specification

Source: Hong Kong Observatory (HKO) “Best Track” data (2023).  
Official dataset page:  
https://data.gov.hk/sc-data/dataset/hk-hko-rss-tropical-cyclone-best-track-data/resource/cb6aa1db-f0c4-4540-ae0c-a2daa37bc0bf

Format notes:
- First 3 remark lines → skipped via `skiprows=3`.
- Column names are trimmed at the first `/` for cleanliness.
- Relevant columns:
  - `Tropical Cyclone Name`
  - `Latitude (0.01 degree N)`  → divide by 100 for degrees
  - `Longitude (0.01 degree E)` → divide by 100 for degrees
  - `Estimated maximum surface winds (knot)`

Wind speed is normalized per run and mapped differently per version (line width, gradient brightness, fractal iteration depth, spike length, spiral radius / rotation acceleration).

---

## 5. Version Overview (v1–v6)

| Version | Script | Visual Focus | Data ↔ Visual Mapping Highlights |
|---------|--------|--------------|----------------------------------|
| v1 | `main_v1.py` | Basic geographic polyline | Wind → color / size (simple) |
| v2 | `main_v2.py` | Dark contrast gradients | Wind → gradient intensity |
| v3 | `main_v3.py` | Fractal overlays | Wind → fractal scale / brightness |
| v4 | `main_v4.py` | Abstract minimal canvas | Wind → layered opacity |
| v5 | `main_v5.py` | Spike / energy fields | Wind → spike count / length |
| v6 | `main_v6.py` | Spiral growth engine (API) | Wind → spiral arms, radial growth, luminosity |

---

## 6. Detailed Version Guides

(Keep brief; full deep dive can move to a `/docs` folder if it grows.)

### v1
Straightforward historical path accumulation. Uses plain `plot` lines; educational baseline.

### v2
Introduces dark background, per‑storm palette variation, more cinematic contrast.

### v3
Adds fractal micro‑glyphs (e.g., Mandelbrot-like patches) to imply energy signatures emerging along the path.

### v4
Drops axes, embraces purely symbolic swirling / glowing constructs. Moves from “map” → “art canvas.”

### v5
Radial spikes and centrifugal offsets suggest kinetic release. Emphasis on intensity-driven deformation.

### v6
Refactors into importable function `animate_typhoons`. Replaces random flicker with deterministic perturbations for smoother temporal coherence. Consolidates color normalization and reduces overdraw for performance.

---

## 7. v6 API & Parameters (`animate_typhoons`)

Signature (as of current code):

```python
def animate_typhoons(
    save_gif: bool = True,
    interval: int = 200,
    fps: int = 5,
    show_track: bool = True,
    hold_last_frames: int = 15
) -> None:
    ...
```

Parameter reference:

| Name | Type | Default | Purpose | Tune When |
|------|------|---------|---------|-----------|
| `save_gif` | bool | True | Whether to encode a GIF (`typhoon_tracks_animation.gif`) | Disable while iterating visuals interactively |
| `interval` | int (ms) | 200 | Delay between frames in interactive live window | Increase for slower preview pacing |
| `fps` | int | 5 | Output GIF frame rate (temporal density) | Raise for smoother GIF; lowers if file too large |
| `show_track` | bool | True | Draw accumulated polyline path beneath spiral forms | Disable for purely abstract look |
| `hold_last_frames` | int | 15 | Duplicate final frame to create a “pause” at end | Reduce to shorten GIF tail |

Important distinction:
- `interval` affects real-time matplotlib playback delay.
- `fps` affects how many frames per second are encoded in the saved GIF.
They do not have to match; mismatch simply changes perceived time scaling between interactive view and final artifact.

Return value: None (side effects: window display, optional GIF file).  
Output filename: `typhoon_tracks_animation.gif` in v6 folder.

---

## 8. Performance Tuning

| Strategy | Effect |
|----------|--------|
| Set `save_gif=False` | Skips encoding loop (fast iteration) |
| Lower `fps` (e.g. 4) | Smaller GIF; less temporal smoothness |
| Reduce `hold_last_frames` | Shortens duration |
| Limit window size / DPI | Current `dpi=90` chosen as compromise |
| Remove `show_track` | Slightly fewer draw calls |
| Downscale spiral detail (code) | Edit: point count / arms in `draw_artistic_typhoon` |
| Batch export MP4 (future) | Use `ffmpeg` for better compression (roadmap) |

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `[ERROR] Data file not found` | Missing `HKO2023BST.csv` in version folder | Copy root CSV into subfolder or adjust path logic |
| GIF not produced | Pillow missing / writer error | `pip install --upgrade pillow matplotlib` |
| Flat colors | Normalization dominated by outliers | Clip or rescale wind values inside code |
| Very large GIF | High `fps` + high `hold_last_frames` | Reduce one or both |
| Window blank / slow | Backend / large monitor scaling | Try smaller figure size or switch Matplotlib backend |

---

## 10. Extensibility & Future Ideas

- Add **CLI** (`argparse`) for filtering by storm name or date range.
- Provide **MP4 export** via `animation.FFMpegWriter`.
- Integrate **cartopy** basemap / coastlines toggle.
- Add **colorbar** legend for wind speed.
- Interactive slider / playback (e.g., **panel**, **ipywidgets**).
- Parameter for **palette set selection**.
- Deterministic RNG seeding for reproducible frames.
- Storm labels with fade-in/out.
- Multi-year merge mode.

---

## 11. Changelog / Evolution Summary

| Version | Key Change | Rationale |
|---------|------------|-----------|
| v1 | Initial polyline build | Establish baseline |
| v2 | Dark styling + gradients | Visual contrast |
| v3 | Fractal embellishments | Symbolic energy motifs |
| v4 | Axis removal + abstraction | Pure art direction |
| v5 | Spike/energy geometry | Emphasize intensity |
| v6 | Spiral engine + API function | Reusability & smoother motion |

---

## 12. Example Programmatic Calls

Minimal (no track overlay, quicker):
```python
from typhoon_data_visualization_v6.main_v6 import animate_typhoons
animate_typhoons(save_gif=False, show_track=False, fps=6, interval=160, hold_last_frames=5)
```

High-resolution feeling (bigger pause):
```python
animate_typhoons(save_gif=True, fps=8, interval=180, hold_last_frames=25)
```

---

## 13. Preview Media

- `typhoon_spiral_art.png` – Conceptual spiral rendering
- `typhoon_data_visualization_v6/typhoon_tracks_animation_v6.gif` – Latest full animation
- Earlier GIFs in version folders illustrate stylistic evolution.

---

## 14. License (Placeholder)

No license specified yet. Recommended to adopt **MIT** or **Apache-2.0** before broader sharing.  
(Without a license, others technically cannot reuse or modify.)

---

## 15. Attribution & Data Notice

Data structure based on Hong Kong Observatory Tropical Cyclone “Best Track” 2023 dataset.  
Users must obtain, verify, and ensure any redistribution complies with source terms.  
If used in publications or media, add formal citation per HKO guidance.

---

### Quick Start (Shortcut)

```bash
python typhoon_data_visualization_v6/main_v6.py
```

Or import programmatically (see Section 12).

---

### Disclaimer

This repository treats meteorological data through an artistic lens; visual distortions are intentional and not suitable for operational meteorology or safety decisions.

---

Happy exploring the storm aesthetics!
