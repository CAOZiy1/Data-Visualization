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



    # Assign a main color scheme (red, blue, green, purple) to each typhoon
    from matplotlib.colors import LinearSegmentedColormap
    base_colors = [
        ('red',   ['#330000', '#ff6666', '#ffcccc']),
        ('blue',  ['#000033', '#3399ff', '#cce6ff']),
        ('green', ['#003300', '#66ff66', '#ccffcc']),
        ('purple',['#220033', '#cc66ff', '#f3e6ff'])
    ]
    typhoon_names = list(grouped.groups.keys())
    typhoon_cmaps = {}
    for i, name in enumerate(typhoon_names):
        _, color_list = base_colors[i % len(base_colors)]
        cmap = LinearSegmentedColormap.from_list(f'typhoon_{i}', color_list, N=256)
        typhoon_cmaps[name] = cmap
    # You can get the colormap for each typhoon by typhoon_cmaps[typhoon_name]

    wind_min = df['Estimated maximum surface winds (knot)'].min()
    wind_max = df['Estimated maximum surface winds (knot)'].max()

    typhoon_collections = []
    typhoon_collections = []
    typhoon_data = []
    for idx, (name, group) in enumerate(grouped):
        lats = (group['Latitude (0.01 degree N)'] / 100.0).to_numpy()
        lons = (group['Longitude (0.01 degree E)'] / 100.0).to_numpy()
        winds = group['Estimated maximum surface winds (knot)'].to_numpy()
        # Normalize wind speed: higher wind = lighter color, lower wind = darker color
        wind_norm = (winds - wind_min) / (wind_max - wind_min + 1e-6)
        # Reverse color scale: 0 is dark, 1 is light
        color_vals = 1.0 - wind_norm
        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = typhoon_cmaps[name]
        lc = LineCollection([], cmap=cmap, norm=Normalize(vmin=0, vmax=1), linewidth=3, alpha=0.0, zorder=3)  # alpha=0 hides original line
        ax.add_collection(lc)
        typhoon_collections.append(lc)
        # Pass color_vals for later fractal art
        typhoon_data.append((segments, color_vals, name, lons, lats, cmap, wind_norm))
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
    max_len = max(len(lons) for _, _, _, lons, lats, _, _ in typhoon_data)


    def draw_mandelbrot(ax, x0, y0, size, cmap, color_val, wind_val, zoom=0.15, res=18):
        # Draw a Mandelbrot fractal thumbnail at (x0, y0), color controlled by cmap and color_val
        # zoom controls fractal detail, res controls resolution
        x = np.linspace(-zoom, zoom, res) + x0
        y = np.linspace(-zoom, zoom, res) + y0
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        img = np.zeros(C.shape, dtype=float)
        max_iter = int(20 + 80 * wind_val)  # Higher wind = more iterations
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            img += mask & (img == 0) * (i + 1)
        img[img == 0] = max_iter
        # Normalize
        img = img / img.max()
        # Color mapping
        rgb_img = cmap(color_val * img)
        ax.imshow(rgb_img, extent=[x0-size, x0+size, y0-size, y0+size], origin='lower', zorder=6, alpha=0.85)

    def update(frame):
        # Remove old fractals (to avoid overlap)
        [im.remove() for im in ax.get_images() if im.get_zorder() == 6]
        for idx, (lc, (segments, color_vals, name, lons, lats, cmap, wind_norm)) in enumerate(zip(typhoon_collections, typhoon_data)):
            if frame < 1 or frame >= len(lons):
                lc.set_segments([])
                continue
            segs = segments[:frame]
            colors = color_vals[:frame]
            winds = wind_norm[:frame]
            lc.set_segments([])  # Do not draw original track line
            # Draw Mandelbrot fractal thumbnail at each track point
            for i in range(frame):
                x0, y0 = lons[i], lats[i]
                color_val = colors[i]
                wind_val = winds[i]
                size = 0.18 + 0.10 * wind_val  # Higher wind = larger fractal
                draw_mandelbrot(ax, x0, y0, size, cmap, color_val, wind_val)
        return typhoon_collections

    # Use the first typhoon's colormap for the colorbar
    first_typhoon = typhoon_names[0]
    sm = cm.ScalarMappable(norm=Normalize(vmin=wind_min, vmax=wind_max), cmap=typhoon_cmaps[first_typhoon])
    cbar = plt.colorbar(sm, ax=ax, label='Estimated maximum surface winds (knot)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=120, blit=True, repeat=False)
    ani.save('typhoon_tracks_animation.gif', writer='pillow', fps=10)
    plt.show()



if __name__ == '__main__':
    animate_typhoons()
