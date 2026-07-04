"""Generate a minimalist icon for SMR_CST_TES_Simulator"""
from PIL import Image, ImageDraw

SIZE = 256
img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Dark background circle
r = SIZE // 2 - 4
draw.ellipse([4, 4, SIZE - 4, SIZE - 4], fill='#121826')

# Three vertical energy bars - minimalist style
bar_w = 36
gap = 16
total_w = bar_w * 3 + gap * 2
start_x = (SIZE - total_w) // 2
bottom_y = SIZE - 48

bars = [
    ('#2F7BED', 0.75),   # SMR blue
    ('#F59E0B', 0.95),   # CST orange
    ('#14B8A6', 0.85),   # TES teal
]

for (color, h_ratio), i in zip(bars, range(3)):
    x0 = start_x + i * (bar_w + gap)
    x1 = x0 + bar_w
    h = int((bottom_y - 56) * h_ratio)
    y0 = bottom_y - h
    # Round corners with smaller radius
    r = 8
    draw.rounded_rectangle([x0, y0, x1, bottom_y], radius=r, fill=color)

# Subtle glow dots above bars
dot_r = 4
for i in range(3):
    cx = start_x + bar_w // 2 + i * (bar_w + gap)
    cy = bottom_y - int((bottom_y - 56) * bars[i][1]) - 14
    draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=bars[i][0])

# Save as .ico
img.save('icon.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("icon.ico created")