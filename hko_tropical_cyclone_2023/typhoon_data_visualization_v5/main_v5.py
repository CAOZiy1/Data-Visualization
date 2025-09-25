import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# Auto-locate the CSV in the same directory as this script
CSV_FILE = Path(__file__).with_name("HKO2023BST.csv")

def animate_typhoons(save_gif=True, interval=200, fps=5):
    if not CSV_FILE.exists():
        print(f"[ERROR] Data file not found: {CSV_FILE}")
        return

    # Read data (first three rows are descriptive headers -> skip)
    df = pd.read_csv(CSV_FILE, skiprows=3)
    df.columns = [c.split('/')[0].strip() for c in df.columns]

    required = [
        'Tropical Cyclone Name',
        'Latitude (0.01 degree N)',
        'Longitude (0.01 degree E)',
        'Estimated maximum surface winds (knot)'
    ]
    for col in required:
        if col not in df.columns:
            print("[ERROR] Missing column:", col)
            return

    grouped = df.groupby('Tropical Cyclone Name')

    all_lats = df['Latitude (0.01 degree N)'] / 100.0
    all_lons = df['Longitude (0.01 degree E)'] / 100.0
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(lon_min - 1, lon_max + 1)
    ax.set_ylim(lat_min - 1, lat_max + 1)
    ax.axis('off')

    # Color palettes per cyclone
    palette_sets = [
        ['#ffb3b3', '#ff6666', '#ff3333', '#ff9999', '#ffd6d6', '#fff5f5'],
        ['#b3d1ff', '#66a3ff', '#3385ff', '#99c2ff', '#d6eaff', '#f5faff'],
        ['#b3ffb3', '#66ff66', '#33cc33', '#99ff99', '#d6ffd6', '#f5fff5'],
        ['#e0b3ff', '#c266ff', '#a633ff', '#cc99ff', '#ecd6ff', '#faf5ff'],
        ['#ffd9b3', '#ffb366', '#ff9933', '#ffcc99', '#ffe6d6', '#fffaf5'],
        ['#b3ffff', '#66ffff', '#33cccc', '#99ffff', '#d6ffff', '#f5ffff']
    ]

    names = list(grouped.groups.keys())
    cmaps = {
        name: LinearSegmentedColormap.from_list(f"ty_{i}", palette_sets[i % len(palette_sets)], N=256)
        for i, name in enumerate(names)
    }

    wind_min = df['Estimated maximum surface winds (knot)'].min()
    wind_max = df['Estimated maximum surface winds (knot)'].max()

    typhoon_tracks = []
    for name, g in grouped:
        lats = (g['Latitude (0.01 degree N)'] / 100.0).to_numpy()
        lons = (g['Longitude (0.01 degree E)'] / 100.0).to_numpy()
        winds = g['Estimated maximum surface winds (knot)'].to_numpy()
        wind_norm = (winds - wind_min) / (wind_max - wind_min + 1e-6)
        color_vals = 1.0 - wind_norm   # Reverse: stronger wind => brighter
        typhoon_tracks.append((name, lons, lats, winds, wind_norm, color_vals, cmaps[name]))

    max_len = max(len(t[1]) for t in typhoon_tracks)

    def update(frame):
        # Remove previous frame's custom artists
        for ln in list(ax.lines):
            if getattr(ln, "_typhoon_art", False):
                ln.remove()
        for p in list(ax.patches):
            if getattr(p, "_typhoon_art", False):
                p.remove()

        artists = []
        if frame == 0:
            return artists

        for (name, lons, lats, winds, wind_norm, color_vals, cmap) in typhoon_tracks:
            if frame >= len(lons):
                idx = len(lons) - 1
            else:
                idx = frame - 1
            if idx <= 0:
                direction_angle = 0.0
            else:
                dx = lons[idx] - lons[idx - 1]
                dy = lats[idx] - lats[idx - 1]
                direction_angle = np.arctan2(dy, dx)

            size = 0.12 + 0.18 * wind_norm[idx]
            draw_artistic_typhoon(
                ax,
                lons[idx],
                lats[idx],
                direction_angle,
                size,
                cmap,
                color_vals[idx],
                wind_norm[idx],
                frame=frame
            )

        # Collect newly added artists
        for ln in ax.lines:
            if getattr(ln, "_typhoon_art", False):
                artists.append(ln)
        for p in ax.patches:
            if getattr(p, "_typhoon_art", False):
                artists.append(p)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=interval, blit=True)

    if save_gif:
        out = CSV_FILE.parent / "typhoon_tracks_animation.gif"
        print(f"[INFO] Saving GIF: {out}")
        ani.save(out, writer="pillow", fps=fps)

    plt.show()

def draw_artistic_typhoon(ax, x0, y0, direction_angle, size, cmap, color_val, wind_val,
                          n_spikes=70, frame=0):
        # Central spike-like radial structure
        base_rotation = frame * (0.6 + 0.35 * wind_val)
        angles = np.linspace(0, 2 * np.pi, n_spikes, endpoint=False)
        for ang in angles:
            direction_factor = 1.0 + 4.0 * np.clip(np.cos(ang - direction_angle), 0, 1)
            length = size * (8 + 25 * wind_val) * direction_factor * (0.7 + 0.7 * np.random.rand())
            phase_jitter = np.sin(frame * 0.1 + ang * 6) * 0.5
            rot = base_rotation + phase_jitter
            x1 = x0 + length * np.cos(ang + rot)
            y1 = y0 + length * np.sin(ang + rot)
            color_factor = 0.8 + 0.2 * np.clip(np.cos(ang - direction_angle), 0, 1)
            spike_color = cmap(min(1, color_val * color_factor + 0.08 * np.random.rand()))
            alpha = 0.4 + 0.5 * direction_factor * (0.7 + 0.3 * np.random.rand())
            lw = 0.5 + 0.7 * wind_val * direction_factor
            ln, = ax.plot([x0, x1], [y0, y1],
                          color=spike_color,
                          alpha=np.clip(alpha, 0.2, 1),
                          linewidth=lw,
                          zorder=9)
            ln._typhoon_art = True

        # Outer faint rings
        for i in range(2):
            r = size * (2 + 2.5 * wind_val) * (1 + 0.15 * i)
            circ = plt.Circle(
                (x0, y0),
                r,
                color=cmap(color_val * (0.45 + 0.05 * i)),
                alpha=0.08 - 0.02 * i,
                zorder=10 + i
            )
            circ._typhoon_art = True
            ax.add_patch(circ)

        # Inner core
        inner = plt.Circle(
            (x0, y0),
            size * (1 + 2 * wind_val),
            color=cmap(0.8),
            alpha=0.18,
            zorder=15
        )
        inner._typhoon_art = True
        ax.add_patch(inner)

        # Extra energetic lines for strong storms
        if wind_val > 0.7:
            for _ in range(6):
                jitter_ang = np.random.rand() * 2 * np.pi
                jitter_len = size * (2.0 + 2 * np.random.rand())
                xj = x0 + jitter_len * np.cos(jitter_ang)
                yj = y0 + jitter_len * np.sin(jitter_ang)
                ln, = ax.plot([x0, xj], [y0, yj],
                              color=cmap(0.1 + 0.8 * np.random.rand()),
                              alpha=0.18,
                              linewidth=1.0,
                              zorder=8)
                ln._typhoon_art = True

if __name__ == "__main__":
    animate_typhoons()