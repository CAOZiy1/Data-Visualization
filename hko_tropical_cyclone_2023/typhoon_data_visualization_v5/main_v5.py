import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

# File name for the typhoon best track data
CSV_FILE = 'E:/VSCode/Helloworld/hko_tropical_cyclone_2023/hko_tropical_cyclone_2023/HKO2023BST.csv'

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
import matplotlib.cm as cm
import matplotlib as mpl

def animate_typhoons():
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
    # Remove all text and axes decorations so only animation remains
    ax.axis('off')

    # Assign enhanced color schemes with more vibrant gradients
    from matplotlib.colors import LinearSegmentedColormap
    base_colors = [
        ('red',    ['#0a0000', '#660000', '#ff0000', '#ff6666', '#ffaaaa', '#ffffff']),
        ('blue',   ['#000033', '#001166', '#0044ff', '#3377ff', '#66aaff', '#ffffff']),
        ('green',  ['#001100', '#003300', '#00aa00', '#44ff44', '#88ff88', '#ffffff']),
        ('purple', ['#220033', '#550066', '#aa00aa', '#dd44dd', '#ff88ff', '#ffffff']),
        ('orange', ['#331100', '#663300', '#ff6600', '#ff9944', '#ffcc88', '#ffffff']),
        ('cyan',   ['#003333', '#006666', '#00cccc', '#44ffff', '#88ffff', '#ffffff'])
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
        # Removed text labels to avoid any displayed text

    max_len = max(len(lons) for _, _, _, lons, lats, _, _ in typhoon_data)


    # Adjust draw_artistic_typhoon for larger spikes and centrifugal offset
    def draw_artistic_typhoon(ax, x0, y0, direction_angle, size, cmap, color_val, wind_val, zoom=0.15, res=36):
        # Draw an artistic typhoon pattern combining spirals and directional effects
        
        # 1. Mandelbrot fractal core
        x = np.linspace(-zoom, zoom, res) + x0
        y = np.linspace(-zoom, zoom, res) + y0
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        img = np.zeros(C.shape, dtype=float)
        max_iter = int(20 + 80 * wind_val)
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            img += mask & (img == 0) * (i + 1)
        img[img == 0] = max_iter
        img = img / img.max()
        rgb_img = cmap(color_val * img)
        ax.imshow(rgb_img, extent=[x0-size, x0+size, y0-size, y0+size], origin='lower', zorder=6, alpha=0.7)
        
        # 2. Enhanced spiral arms with particle effects
        theta = np.linspace(0, 12*np.pi, 600)  # Further increase spiral detail
        for arm in range(10):  # Increase number of spiral arms
            arm_offset = arm * np.pi/5 + direction_angle
            spiral_r = size * (2.0 + 2.5 * np.exp(-theta * 0.05))  # Drastically increase spiral radius
            
            # Add extreme randomness for centrifugal effect
            noise = 0.5 * size * np.sin(theta * 12) * wind_val
            spiral_x = x0 + (spiral_r + noise) * np.cos(theta + arm_offset)
            spiral_y = y0 + (spiral_r + noise) * np.sin(theta + arm_offset)
            
            # Dynamic color and width variation
            intensities = np.exp(-theta * 0.05) * (0.5 + 0.5 * wind_val)
            colors = [cmap(color_val * intensity) for intensity in intensities]
            widths = size * 30 * wind_val * intensities  # Further increase spike width
            
            for i in range(len(spiral_x)-1):
                alpha = max(0.2, intensities[i] * 0.8)
                ax.plot([spiral_x[i], spiral_x[i+1]], [spiral_y[i], spiral_y[i+1]], 
                       color=colors[i], linewidth=widths[i], alpha=alpha, zorder=5)
        
        # 3. Dynamic directional tail with turbulence effect
        tail_length = size * 4 * (0.5 + wind_val)
        tail_angle = direction_angle + np.pi
        
        for j in range(7):  # More tail streamers
            offset_angle = tail_angle + (j-3) * 0.4
            turbulence = 0.2 * size * np.sin(j * 2) * wind_val
            
            tail_x = [x0, x0 + tail_length * np.cos(offset_angle) + turbulence]
            tail_y = [y0, y0 + tail_length * np.sin(offset_angle) + turbulence]
            
            tail_color = cmap(color_val * (0.2 + 0.3 * (1 - j/7)))
            tail_width = size * 20 * wind_val * (1 - j/7)
            tail_alpha = max(0.1, (0.6 - j * 0.08) * wind_val)
            
            ax.plot(tail_x, tail_y, color=tail_color, 
                   linewidth=tail_width, alpha=tail_alpha, zorder=4)
        
        # 4. Pulsating eye of the storm
        eye_size = size * (0.2 + 0.2 * np.sin(wind_val * 10))  # Pulsating effect
        # Outer eye wall
        outer_eye = plt.Circle((x0, y0), eye_size * 1.5, 
                              color=cmap(color_val * 0.6), alpha=0.4, zorder=7)
        ax.add_patch(outer_eye)
        # Inner eye
        inner_eye = plt.Circle((x0, y0), eye_size, 
                              color=cmap(0.95), alpha=0.9, zorder=8)
        ax.add_patch(inner_eye)

    def update(frame):
        # Remove old patches and images to avoid overlap
        [im.remove() for im in ax.get_images() if im.get_zorder() >= 4]
        [patch.remove() for patch in ax.patches]
        
        for idx, (lc, (segments, color_vals, name, lons, lats, cmap, wind_norm)) in enumerate(zip(typhoon_collections, typhoon_data)):
            if frame < 1 or frame >= len(lons):
                lc.set_segments([])
                continue
            segs = segments[:frame]
            colors = color_vals[:frame]
            winds = wind_norm[:frame]
            lc.set_segments([])  # Do not draw original track line
            
            # Draw artistic typhoon pattern at each track point
            for i in range(frame):
                x0, y0 = lons[i], lats[i]
                color_val = colors[i]
                wind_val = winds[i]
                size = 1.0 + 1.0 * wind_val  # Drastically increase size proportional to wind speed
                
                # Calculate movement direction angle
                if i > 0:
                    dx = lons[i] - lons[i-1]
                    dy = lats[i] - lats[i-1]
                    direction_angle = np.arctan2(dy, dx)
                else:
                    direction_angle = 0  # Default direction for first point
                
                draw_artistic_typhoon(ax, x0, y0, direction_angle, size, cmap, color_val, wind_val)
        return typhoon_collections

    # Removed colorbar and any displayed text to show only the animation

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=200, blit=True, repeat=False)
    ani.save('typhoon_tracks_animation.gif', writer='pillow', fps=5)
    plt.show()



if __name__ == '__main__':
    animate_typhoons()