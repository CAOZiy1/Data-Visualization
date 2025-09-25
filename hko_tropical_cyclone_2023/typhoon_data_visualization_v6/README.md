# Typhoon Best Track Artistic Animation

An artistic animation of 2023 tropical cyclone (typhoon) tracks using best track CSV data (e.g. HKO2023BST.csv). Each cyclone is rendered with a stylized rotating spiral whose size and brightness relate to its wind intensity. Optionally the historical track polyline is shown. A GIF can be exported.

## Features
- Reads best track CSV (skips first 3 description lines)
- Per‑cyclone color palettes (auto-assigned)
- Spiral animation with wind-based size/intensity
- Optional track lines
- Generates animated GIF (Pillow writer)
- Fully black background for visual contrast

## Output
Default: typhoon_tracks_animation.gif (saved beside the CSV).
Also opens an interactive animation window (matplotlib).

## File Requirements
Expected column names (after trimming anything after a "/"):
- Tropical Cyclone Name
- Latitude (0.01 degree N)
- Longitude (0.01 degree E)
- Estimated maximum surface winds (knot)

Latitude / Longitude are stored in hundredths of degrees (so divide by 100.0).

The script skips the first 3 lines: place any metadata there if present.

## Project Structure (relevant)
- typhoon_data_visualization_v6/
  - main_v6.py
  - HKO2023BST.csv (place here)

## Installation
```bash
pip install -r requirements.txt
```
Minimal required packages:
- pandas
- numpy
- matplotlib
- pillow (for GIF saving; usually comes with matplotlib but install if missing)

## Usage
From project root:
```bash
python typhoon_data_visualization_v6/main_v6.py
```

### Adjust Parameters
Edit the call at the bottom of main_v6.py:
```python
animate_typhoons(
    save_gif=True,
    interval=200,      # ms between frames
    fps=5,             # GIF frame rate
    show_track=True,   # toggle path polyline
    hold_last_frames=15
)
```

### Example (custom call in a Python shell)
```python
from typhoon_data_visualization_v6.main_v6 import animate_typhoons
animate_typhoons(save_gif=False, show_track=False, fps=8, interval=120)
```

## Performance Tips
- Decrease interval or increase fps cautiously (larger GIF size).
- Set save_gif=False while tweaking visuals to speed iteration.
- Reduce hold_last_frames to shorten final pause.

## Notes
- No cartopy/geographic basemap in this version; it uses raw lat/lon in a fixed extent.
- Wind intensity is normalized per full dataset for relative scaling.
- Stronger cyclones render brighter, larger spirals.

## License
Add a license section if distributing.

## Future Ideas
- Add basemap (cartopy) layer (coastlines)
- CLI argument parsing
- Legend / colorbar
- Filtering by cyclone name or date range
