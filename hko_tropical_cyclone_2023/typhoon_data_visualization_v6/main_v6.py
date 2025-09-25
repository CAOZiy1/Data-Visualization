import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection   # 新增

# 自动定位：脚本同目录的 HKO2023BST.csv
CSV_FILE = Path(__file__).with_name("HKO2023BST.csv")

def animate_typhoons(save_gif=True, interval=200, fps=5,
                     show_track=True,      # 是否显示轨迹
                     hold_last_frames=15   # 结尾多停留帧数
                     ):
    if not CSV_FILE.exists():
        print(f"[ERROR] 未找到数据文件: {CSV_FILE}")
        return

    # 读数据（前三行是标题说明，跳过）
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
            print("[ERROR] 缺少列:", col)
            return

    grouped = df.groupby('Tropical Cyclone Name')

    all_lats = df['Latitude (0.01 degree N)'] / 100.0
    all_lons = df['Longitude (0.01 degree E)'] / 100.0
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    # 让 Axes 填满整个 Figure，去掉边距
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    ax.set_aspect('equal', adjustable='box')
    # 固定范围，避免每帧重算 autoscale (提升速度)
    pad_lat = (lat_max - lat_min) * 0.05
    pad_lon = (lon_max - lon_min) * 0.05
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)

    # 去除脊线（虽然 axis('off') 已经隐藏，但保险）
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 配色
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
        color_vals = 1.0 - wind_norm   # 反转：风越大越亮
        typhoon_tracks.append((name, lons, lats, winds, wind_norm, color_vals, cmaps[name]))

    max_len = max(len(t[1]) for t in typhoon_tracks)
    total_frames = max_len + hold_last_frames

    def update(frame):
        # 映射到数据帧（含结尾停留）
        data_frame = min(frame, max_len - 1)

        # 清理旧帧元素
        for ln in list(ax.lines):
            if getattr(ln, "_typhoon_art", False):
                ln.remove()
        for p in list(ax.patches):
            if getattr(p, "_typhoon_art", False):
                p.remove()

        artists = []

        for (name, lons, lats, winds, wind_norm, color_vals, cmap) in typhoon_tracks:
            track_len = len(lons)
            if track_len == 0:
                continue

            # 针对每个台风单独截断索引
            idx = min(data_frame, track_len - 1)

            # 方向（当停留在最后帧时保持最后一次有效移动方向）
            if idx == 0:
                direction_angle = 0.0
            else:
                dx = lons[idx] - lons[idx - 1]
                dy = lats[idx] - lats[idx - 1]
                direction_angle = 0.0 if (dx == 0 and dy == 0) else np.arctan2(dy, dx)

            # 轨迹线
            if show_track and idx > 0:
                tr_x = lons[:idx + 1]
                tr_y = lats[:idx + 1]
                track_color = cmap(color_vals[idx] * 0.6 + 0.2)
                track_line, = ax.plot(
                    tr_x, tr_y,
                    color=track_color,
                    linewidth=1.2,
                    alpha=0.55,
                    zorder=3
                )
                track_line._typhoon_art = True
                artists.append(track_line)

            # 螺旋主体
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
                frame=data_frame
            )

        # 收集
        for ln in ax.lines:
            if getattr(ln, "_typhoon_art", False) and ln not in artists:
                artists.append(ln)
        for p in ax.patches:
            if getattr(p, "_typhoon_art", False):
                artists.append(p)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=False)
    if save_gif:
        out = CSV_FILE.parent / "typhoon_tracks_animation.gif"
        print(f"[INFO] 保存 GIF: {out}")
        ani.save(out, writer="pillow", fps=fps, dpi=90)  # dpi 降一点提速

    plt.show()

def draw_artistic_typhoon(ax, x0, y0, direction_angle, size, cmap, color_val, wind_val,
                          n_spikes=70, frame=0):
    # 螺旋臂数量（限制上限 5 提速）
    n_arms = 3 + int(2 * min(1, wind_val + 0.15))
    base_rotation = frame * (0.35 + 0.45 * wind_val)

    # 点数减半
    turns = 1.2 + 1.8 * wind_val
    theta_max = turns * 2 * np.pi
    n_points = 70
    thetas_base = np.linspace(0, theta_max, n_points)

    radial_growth = size * (0.10 + 0.30 * wind_val)
    cos_dir = np.cos(direction_angle)
    sin_dir = np.sin(direction_angle)

    for arm in range(n_arms):
        phase = arm * 2 * np.pi / n_arms
        # 用确定性扰动（不再每帧随机 b，避免闪烁 + 减少随机开销）
        jitter = 0.15 * wind_val * np.sin(thetas_base * (2.2 + 1.2 * wind_val) + phase * 1.4)
        thetas = thetas_base + phase + base_rotation + jitter * 0.55

        a = size * (0.25 + 0.32 * wind_val)
        b = radial_growth * (0.85 + 0.25 * np.sin(phase + frame * 0.02))
        r = a + b * thetas

        arm_x = np.cos(thetas)
        arm_y = np.sin(thetas)
        forward_factor = np.clip(arm_x * cos_dir + arm_y * sin_dir, 0, 1)
        r *= (1 + forward_factor * 0.75 * wind_val)

        x = x0 + r * arm_x
        y = y0 + r * arm_y

        drift_scale = size * (0.015 + 0.08 * wind_val)
        x += drift_scale * np.sin(thetas * (3.2 + wind_val) + frame * 0.035 + arm)
        y += drift_scale * np.cos(thetas * (2.8 + 0.5 * wind_val) + frame * 0.045 + arm * 1.1)

        # 生成分段 (n_points-1 段)
        segs = np.stack(
            [np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])],
            axis=1
        )

        seg_frac = np.linspace(0, 1, n_points - 1)
        line_alpha = (0.50 + 0.40 * wind_val) * (1 - 0.60 * seg_frac)
        line_alpha *= (0.65 + 0.35 * forward_factor[:-1])
        col_mix = color_val * (0.45 + 0.25 * wind_val) + seg_frac * 0.40
        col_mix = np.clip(col_mix, 0, 1)

        cols = cmap(col_mix)
        cols[:, 3] = np.clip(line_alpha, 0.05, 1)

        widths = 0.7 + 1.3 * (1 - seg_frac) * (0.35 + 0.65 * wind_val)

        lc = LineCollection(
            segs,
            colors=cols,
            linewidths=widths,
            zorder=9
        )
        lc._typhoon_art = True
        ax.add_collection(lc)

    # 内核
    core_r_base = size * (0.65 + 1.4 * wind_val)
    for i in range(3):
        cr = core_r_base * (0.38 + 0.52 * i / 3)
        circ = plt.Circle(
            (x0, y0),
            cr,
            color=cmap(0.65 + 0.35 * (i / 3)),
            alpha=0.17 - 0.03 * i,
            zorder=11 + i
        )
        circ._typhoon_art = True
        ax.add_patch(circ)

    # 外淡环
    for i in range(2):
        r_out = size * (2.0 + 2.6 * wind_val) * (1 + 0.18 * i)
        ring = plt.Circle(
            (x0, y0),
            r_out,
            color=cmap(color_val * (0.40 + 0.06 * i)),
            alpha=0.055 - 0.015 * i,
            zorder=7
        )
        ring._typhoon_art = True
        ax.add_patch(ring)

    if wind_val > 0.7:
        # 降低数量
        for _ in range(5):
            ang = np.random.rand() * 2 * np.pi
            jitter_len = size * (2.0 + 2.2 * np.random.rand())
            ex = x0 + jitter_len * np.cos(ang)
            ey = y0 + jitter_len * np.sin(ang)
            ln, = ax.plot(
                [x0, ex], [y0, ey],
                color=cmap(0.20 + 0.60 * np.random.rand()),
                alpha=0.12,
                linewidth=0.8,
                zorder=6
            )
            ln._typhoon_art = True

if __name__ == "__main__":
    animate_typhoons()