


import pandas as pd
# Artistic animation using pygame and pymunk
import pygame
import pymunk
import random

# Use pasted weather data directly
weather_data = [
    ["King's Park", 23.9, 32.4],
    ["Wong Chuk Hang", 24.4, 33.6],
    ["Ta Kwu Ling", 24.9, 33.3],
    ["Lau Fau Shan", 26.1, 34.5],
    ["Tai Po", 26.3, 33.3],
    ["Sha Tin", 26.3, 33.2],
    ["Tuen Mun", 26.9, 32.5],
    ["Tseung Kwan O", 24.6, 32.8],
    ["Sai Kung", 26.5, 32.7],
    ["Cheung Chau", 23.5, 31.6],
    ["Chek Lap Kok", 26.7, 35.1],
    ["Tsing Yi", 24.7, 33.8],
    ["Tsuen Wan Ho Koon", 25.2, 32.5],
    ["Tsuen Wan Shing Mun Valley", 25.9, 33.5],
    ["Hong Kong Park", 23.4, 31.8],
    ["Shau Kei Wan", 23.2, 31.6],
    ["Kowloon City", 24.2, 33.6],
    ["Wong Tai Sin", 25.8, 33.5],
    ["Stanley", 24.4, 33.1],
    ["Kwun Tong", 24.3, 32.6],
    ["Sham Shui Po", 24.9, 34.1],
    ["Kai Tak Runway Park", 24.4, 32.5],
    ["Yuen Long Park", 25.6, 35.3],
    ["Tai Mei Tuk", 27.0, 31.1],
]
weather_df = pd.DataFrame(weather_data, columns=["Station", "MinTemp", "MaxTemp"])


# Add radiation data from user
radiation_data = [
    ["Ping Chau", 0.08],
    ["Tap Mun", 0.08],
    ["Kat O", 0.10],
    ["Yuen Ng Fan", 0.11],
    ["Tai Mei Tuk", 0.11],
    ["Sha Tau Kok", 0.10],
    ["Kwun Tong", 0.12],
    ["Sai Wan Ho", 0.08],
    ["King's Park", 0.14],
    ["Tsim Bei Tsui", 0.13],
    ["Cape D'Aguilar", 0.14],
    ["Chek Lap Kok", 0.15],
]
radiation_df = pd.DataFrame(radiation_data, columns=["Station", "Radiation"])

# Merge only stations present in both datasets for animation
if not radiation_df.empty:
    merged_df = pd.merge(weather_df, radiation_df, on="Station", how="inner")
else:
    merged_df = weather_df.copy()


# Artistic fractal-inspired animated graph using temperature and radiation
import matplotlib.animation as animation
import numpy as np

if not merged_df.empty:
    # Prepare data
    temp = merged_df['MaxTemp'].to_numpy()
    rad = merged_df['Radiation'].to_numpy()
    stations = merged_df['Station'].to_numpy()
    n = len(temp)

    # Normalize for color/size
    temp_norm = (temp - temp.min()) / (temp.max() - temp.min())
    rad_norm = (rad - rad.min()) / (rad.max() - rad.min())

    # --- Save a static matplotlib figure ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,8))
    scatter = ax.scatter(temp, rad, s=200*rad_norm+100, c=temp, cmap='plasma', alpha=0.8, edgecolors='k')
    for i, label in enumerate(stations):
        ax.text(temp[i], rad[i]+0.01, label, fontsize=8, ha='center', color='white', bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.2'))
    ax.set_xlabel('Max Temperature (°C)')
    ax.set_ylabel('Radiation (uSv/h)')
    ax.set_title('HK Weather & Radiation (Static)')
    plt.grid(True, alpha=0.3)
    plt.savefig('weather_radiation_static.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Artistic Pygame Animation ---
    pygame.init()
    W, H = 900, 900
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Artistic HK Weather & Radiation (Pygame + Pymunk)")
    clock = pygame.time.Clock()

    # Pymunk physics
    space = pymunk.Space()
    space.gravity = (0, 900)

    # Create artistic particles for each station
    particles = []
    for i in range(n):
        mass = 1 + 2 * rad_norm[i]
        radius = 25 + 45 * temp_norm[i]
        x = int(W/2 + 300 * np.cos(2 * np.pi * i / n))
        y = int(H/2 + 300 * np.sin(2 * np.pi * i / n))
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        body.position = x, y
        shape = pymunk.Circle(body, radius)
        # Color: temp = red, rad = blue, blend for purple
        color = (
            int(180 + 75 * temp_norm[i]),
            int(60 + 120 * rad_norm[i]),
            int(180 + 75 * rad_norm[i])
        )
        shape.color = color + (180,)
        shape.elasticity = 0.8
        shape.friction = 0.5
        space.add(body, shape)
        particles.append((body, shape, stations[i]))

    # Add artistic boundaries
    static_lines = [
        pymunk.Segment(space.static_body, (50, 50), (W-50, 50), 8),
        pymunk.Segment(space.static_body, (W-50, 50), (W-50, H-50), 8),
        pymunk.Segment(space.static_body, (W-50, H-50), (50, H-50), 8),
        pymunk.Segment(space.static_body, (50, H-50), (50, 50), 8),
    ]
    for line in static_lines:
        line.elasticity = 0.95
        line.friction = 0.5
    space.add(*static_lines)

    font = pygame.font.SysFont('arial', 18, bold=True)
    running = True
    t = 0
    last_frame = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((24, 24, 48))
        # Draw boundaries
        for line in static_lines:
            pygame.draw.line(screen, (80, 80, 180), line.a, line.b, 8)
        # Draw particles
        for i, (body, shape, label) in enumerate(particles):
            pos = int(body.position.x), int(body.position.y)
            pygame.draw.circle(screen, shape.color[:3], pos, int(shape.radius))
            # Artistic glow
            for glow in range(1, 4):
                alpha = max(10, 60 - 15*glow)
                s = pygame.Surface((2*shape.radius, 2*shape.radius), pygame.SRCALPHA)
                pygame.draw.circle(s, shape.color[:3]+(alpha,), (int(shape.radius), int(shape.radius)), int(shape.radius*glow*1.2), width=0)
                screen.blit(s, (pos[0]-shape.radius, pos[1]-shape.radius), special_flags=pygame.BLEND_RGBA_ADD)
            # Draw station label
            label_surf = font.render(str(label), True, (255,255,255))
            screen.blit(label_surf, (pos[0]-label_surf.get_width()//2, pos[1]-label_surf.get_height()//2))
        # Physics step
        space.step(1/60.0)
        # Add random artistic force
        for i, (body, shape, label) in enumerate(particles):
            fx = 200 * (np.sin(t/30 + i) + random.uniform(-0.2,0.2))
            fy = 200 * (np.cos(t/40 + i) + random.uniform(-0.2,0.2))
            body.apply_force_at_local_point((fx, fy))
        t += 1
        pygame.display.flip()
        last_frame = screen.copy()
        clock.tick(60)
    if last_frame is not None:
        pygame.image.save(last_frame, "weather_radiation_artistic.png")
    pygame.quit()
else:
    print("No data to plot.")
