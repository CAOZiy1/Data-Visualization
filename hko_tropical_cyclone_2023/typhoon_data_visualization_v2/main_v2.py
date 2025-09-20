import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

# File name for the typhoon best track data




from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
import matplotlib.cm as cm
import matplotlib as mpl

def animate_typhoons():
    CSV_FILE = 'HKO2023BST.csv'
    if not os.path.exists(CSV_FILE):
        print(f"CSV file '{CSV_FILE}' not found.")
        return

    df = pd.read_csv(CSV_FILE, skiprows=3)
    df.columns = [col.split('/')[0].strip() for col in df.columns]
    grouped = df.groupby('Tropical Cyclone Name')

    all_lats = df['Latitude (0.01 degree N)'] / 100.0
    all_lons = df['Longitude (0.01 degree E)'] / 100.0
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(True, linestyle='--', color='#888888', alpha=0.4)
    ax.set_xlim(lon_min - 1, lon_max + 1)
    ax.set_ylim(lat_min - 1, lat_max + 1)
    ax.set_xlabel('Longitude (°E)', fontsize=14, color='white')
    ax.set_ylabel('Latitude (°N)', fontsize=14, color='white')
    plt.title('Typhoon Tracks Animation (2023)', fontsize=20, fontweight='bold', color='white', pad=20)
    ax.tick_params(axis='both', colors='white')


    # Use tab20 colormap for higher color distinction
    tab_cmap = mpl.colormaps['tab20']
    typhoon_names = list(grouped.groups.keys())
    base_colors = [tab_cmap(i % tab_cmap.N) for i in range(len(typhoon_names))]

    def make_typhoon_cmap(base_rgb):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list('typhoon_cmap', [(0, 'black'), (0.5, base_rgb), (1, 'white')], N=256)

    wind_min = df['Estimated maximum surface winds (knot)'].min()
    wind_max = df['Estimated maximum surface winds (knot)'].max()

    typhoon_collections = []
    typhoon_data = []
    typhoon_cmaps = []
    for idx, (name, group) in enumerate(grouped):
        lats = (group['Latitude (0.01 degree N)'] / 100.0).to_numpy()
        lons = (group['Longitude (0.01 degree E)'] / 100.0).to_numpy()
        winds = group['Estimated maximum surface winds (knot)'].to_numpy()
        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        base_rgb = base_colors[idx]
        cmap = make_typhoon_cmap(base_rgb)
        typhoon_cmaps.append(cmap)
        lc = LineCollection([], cmap=cmap, norm=Normalize(vmin=wind_min, vmax=wind_max), linewidth=3, alpha=0.95, zorder=3)
        ax.add_collection(lc)
        typhoon_collections.append(lc)
        typhoon_data.append((segments, winds, name, lons, lats, cmap))
        # Typhoon name: offset, smaller font, more transparent, white outline
        from matplotlib.patheffects import withStroke
        offset = 0.25  # Offset in longitude/latitude direction
        ax.text(
            lons[0] + offset, lats[0] + offset,
            name,
            fontsize=9,
            fontweight='bold',
            color='white',
            alpha=0.7,
            zorder=4,
            ha='left', va='bottom',
            path_effects=[withStroke(linewidth=2, foreground='black')]
        )

    max_len = max(len(lons) for _, _, _, lons, lats, _ in typhoon_data)

    def update(frame):
        for idx, (lc, (segments, winds, _, lons, lats, cmap)) in enumerate(zip(typhoon_collections, typhoon_data)):
            if frame < 1 or frame >= len(lons):
                lc.set_segments([])
                continue
            segs = segments[:frame]
            colors = winds[:frame]
            lc.set_segments(segs)
            lc.set_array(np.array(colors))
        return typhoon_collections

    sm = cm.ScalarMappable(norm=Normalize(vmin=wind_min, vmax=wind_max), cmap=typhoon_cmaps[0])
    cbar = plt.colorbar(sm, ax=ax, label='Estimated maximum surface winds (knot)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=120, blit=True, repeat=False)
    ani.save('typhoon_tracks_animation.gif', writer='pillow', fps=10)
    plt.show()

if __name__ == '__main__':
    animate_typhoons()
