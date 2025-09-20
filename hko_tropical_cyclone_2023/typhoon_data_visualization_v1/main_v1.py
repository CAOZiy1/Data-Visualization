import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

# File name for the typhoon best track data
CSV_FILE = 'HKO2023BST.csv'

# Intensity color map (customize for artistic effect)
INTENSITY_COLORS = {
    'TD': '#7fc97f',   # Tropical Depression
    'TS': '#beaed4',   # Tropical Storm
    'STS': '#fdc086',  # Severe Tropical Storm
    'TY': '#ffff99',   # Typhoon
    'STY': '#386cb0',  # Severe Typhoon
    'SuperTY': '#f0027f', # Super Typhoon (if present)
}

# Wind speed color map (knots)
WIND_SPEED_COLORS = [
    (0, '#7fc97f'),    # 0-33 knots
    (34, '#beaed4'),   # 34-47 knots
    (48, '#fdc086'),   # 48-63 knots
    (64, '#ffff99'),   # 64-84 knots
    (85, '#386cb0'),   # 85-99 knots
    (100, '#f0027f'),  # 100+ knots
]

def get_color_by_wind_speed(ws):
    for threshold, color in reversed(WIND_SPEED_COLORS):
        if ws >= threshold:
            return color
    return WIND_SPEED_COLORS[0][1]


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
    ax.set_facecolor('#f7f7f7')
    ax.grid(True, linestyle='--', color='#cccccc', alpha=0.7)
    ax.set_xlim(lon_min - 1, lon_max + 1)
    ax.set_ylim(lat_min - 1, lat_max + 1)
    ax.set_xlabel('Longitude (°E)', fontsize=14)
    ax.set_ylabel('Latitude (°N)', fontsize=14)
    plt.title('Typhoon Tracks Animation (2023)', fontsize=20, fontweight='bold', color='#222222', pad=20)

    # Prepare lines for each typhoon
    typhoon_lines = []
    typhoon_data = []
    for name, group in grouped:
        lats = (group['Latitude (0.01 degree N)'] / 100.0).to_numpy()
        lons = (group['Longitude (0.01 degree E)'] / 100.0).to_numpy()
        winds = group['Estimated maximum surface winds (knot)'].to_numpy()
        line, = ax.plot([], [], lw=2.5, alpha=0.85, solid_capstyle='round', zorder=3)
        typhoon_lines.append(line)
        typhoon_data.append((lons, lats, winds, name))
        # Add typhoon name at the start
        ax.text(lons[0], lats[0], name, fontsize=10, fontweight='bold',
                color='#333333', alpha=0.7, zorder=4, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    max_len = max(len(lons) for lons, lats, winds, name in typhoon_data)

    def update(frame):
        for idx, (line, (lons, lats, winds, _)) in enumerate(zip(typhoon_lines, typhoon_data)):
            if frame < 1 or frame >= len(lons):
                line.set_data([], [])
                line.set_linewidth(2.5)
                continue
            # Show up to current frame
            x = lons[:frame+1]
            y = lats[:frame+1]
            ws = winds[:frame+1]
            # Color and width by wind speed at head
            color = get_color_by_wind_speed(ws[-1])
            width = 2 + (ws[-1] / 50)  # scale width
            line.set_data(x, y)
            line.set_color(color)
            line.set_linewidth(width)
        return typhoon_lines

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=120, blit=True, repeat=False)
    ani.save('typhoon_tracks_animation.gif', writer='pillow', fps=10)
    plt.show()

if __name__ == '__main__':
    animate_typhoons()
