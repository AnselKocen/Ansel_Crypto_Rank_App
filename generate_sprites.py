"""
Generate pixel art sprites for Crypto Cat Adventure platformer.
Style: Blue sky, green grass, hero with red cape, golden coins.
Base pixel size = 16x16, SCALE=4 → output 64x64 per frame/tile.
"""
from PIL import Image, ImageDraw
import os, math

SPRITES_DIR = os.path.join(os.path.dirname(__file__), "web", "static", "sprites")
CATS_DIR = os.path.join(SPRITES_DIR, "cats")
os.makedirs(CATS_DIR, exist_ok=True)

S = 4  # scale factor: each art pixel = 4x4 real pixels
BASE = 16  # base tile size in art pixels
OUT = BASE * S  # 64px output

def make_img(w_art, h_art):
    return Image.new("RGBA", (w_art * S, h_art * S), (0, 0, 0, 0))

def px(draw, x, y, color):
    if len(color) == 3:
        color = color + (255,)
    draw.rectangle([x * S, y * S, (x + 1) * S - 1, (y + 1) * S - 1], fill=color)

def fill_rect(draw, x, y, w, h, color):
    for dx in range(w):
        for dy in range(h):
            px(draw, x + dx, y + dy, color)

# ==========================================
# COLOR PALETTE
# ==========================================
SKY_TOP = (135, 206, 250)
SKY_BOT = (176, 224, 255)
CLOUD = (255, 255, 255)
CLOUD_SHADOW = (220, 230, 240)
GRASS_TOP = (76, 175, 80)
GRASS_LIGHT = (102, 187, 106)
DIRT = (139, 90, 43)
DIRT_DARK = (110, 70, 30)
DIRT_LIGHT = (160, 110, 60)
STONE = (120, 120, 140)
STONE_DARK = (90, 90, 110)
STONE_LIGHT = (150, 150, 170)
# Hero
SKIN = (255, 210, 170)
HAIR = (80, 50, 30)
CAPE_RED = (220, 50, 50)
CAPE_DARK = (170, 30, 30)
SHIRT = (60, 60, 180)
SHIRT_LIGHT = (90, 90, 210)
PANTS = (70, 70, 80)
BOOTS = (100, 60, 30)
EYE = (40, 40, 40)
# Items
GOLD = (255, 200, 50)
GOLD_DARK = (200, 150, 30)
GOLD_LIGHT = (255, 235, 130)
QB_ORANGE = (247, 147, 26)
QB_DARK = (200, 110, 20)
QB_LIGHT = (255, 200, 100)
GREEN_OK = (46, 175, 80)
GREEN_DARK = (30, 140, 60)
GREEN_LIGHT = (120, 220, 140)
RED_X = (220, 60, 60)
FLAG_RED = (220, 50, 50)
HEART_RED = (230, 50, 50)
HEART_LIGHT = (255, 120, 120)


# ==========================================
# BACKGROUND: 800x600 sky with clouds and hills
# ==========================================
def generate_background():
    w, h = 800, 600
    img = Image.new("RGBA", (w, h), SKY_TOP)
    draw = ImageDraw.Draw(img)

    # Sky gradient
    for y in range(h):
        t = y / h
        r = int(SKY_TOP[0] * (1 - t) + SKY_BOT[0] * t)
        g = int(SKY_TOP[1] * (1 - t) + SKY_BOT[1] * t)
        b = int(SKY_TOP[2] * (1 - t) + SKY_BOT[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Clouds (pixel-style, chunky)
    cloud_positions = [(80, 60), (300, 40), (550, 70), (700, 30), (150, 100), (450, 90)]
    for cx, cy in cloud_positions:
        # Main body
        for dx in range(-20, 21, S):
            for dy in range(-8, 9, S):
                if dx * dx / 400 + dy * dy / 64 < 1:
                    draw.rectangle([cx + dx, cy + dy, cx + dx + S - 1, cy + dy + S - 1], fill=CLOUD)
        # Top bumps
        for bx, by in [(-12, -10), (4, -14), (14, -8)]:
            for dx in range(-10, 11, S):
                for dy in range(-6, 7, S):
                    if dx * dx / 100 + dy * dy / 36 < 1:
                        draw.rectangle([cx + bx + dx, cy + by + dy, cx + bx + dx + S - 1, cy + by + dy + S - 1], fill=CLOUD)
        # Shadow
        for dx in range(-16, 17, S):
            for dy in range(4, 12, S):
                if dx * dx / 256 + (dy - 6) ** 2 / 36 < 1:
                    draw.rectangle([cx + dx, cy + dy, cx + dx + S - 1, cy + dy + S - 1], fill=CLOUD_SHADOW)

    # Distant hills
    hill_green = (100, 170, 100)
    hill_dark = (80, 145, 80)
    for hx, hw, hh in [(100, 200, 80), (350, 250, 100), (600, 180, 70), (750, 150, 60)]:
        base_y = 450
        for x in range(hx - hw, hx + hw):
            if 0 <= x < w:
                dist = abs(x - hx) / hw
                top_y = int(base_y - hh * (1 - dist * dist))
                c = hill_dark if (x // S) % 3 == 0 else hill_green
                draw.rectangle([x, top_y, x, base_y], fill=c)

    img.save(os.path.join(SPRITES_DIR, "background.png"))
    print("Created background.png (800x600)")


# ==========================================
# PLAYER: Hero with red cape - 16x16 art, 12 frames
# ==========================================
def generate_player():
    n_frames = 12
    img = make_img(BASE * n_frames, BASE)
    draw = ImageDraw.Draw(img)

    def draw_hero(ox, oy, walk_phase=0, jumping=False):
        # Cape (behind body)
        cape_sway = walk_phase % 3
        cape_pixels = [
            (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
            (4, 5), (4, 6), (4, 7), (4, 8),
        ]
        if cape_sway == 1:
            cape_pixels += [(3, 7), (3, 8)]
        elif cape_sway == 2:
            cape_pixels += [(4, 9)]

        for cx, cy in cape_pixels:
            c = CAPE_DARK if cy >= 7 else CAPE_RED
            px(draw, ox + cx, oy + cy, c)

        # Head
        fill_rect(draw, ox + 6, oy + 1, 4, 4, SKIN)
        # Hair
        px(draw, ox + 6, oy + 1, HAIR)
        px(draw, ox + 7, oy + 0, HAIR)
        px(draw, ox + 8, oy + 0, HAIR)
        px(draw, ox + 9, oy + 1, HAIR)
        px(draw, ox + 9, oy + 2, HAIR)
        # Eye
        px(draw, ox + 8, oy + 2, EYE)
        # Mouth
        px(draw, ox + 8, oy + 4, (200, 150, 130))

        # Body (shirt)
        fill_rect(draw, ox + 6, oy + 5, 4, 3, SHIRT)
        px(draw, ox + 7, oy + 5, SHIRT_LIGHT)
        px(draw, ox + 8, oy + 5, SHIRT_LIGHT)

        # Belt
        px(draw, ox + 6, oy + 7, GOLD_DARK)
        px(draw, ox + 7, oy + 7, GOLD)
        px(draw, ox + 8, oy + 7, GOLD)
        px(draw, ox + 9, oy + 7, GOLD_DARK)

        # Arms
        arm_y = 6 + (1 if walk_phase % 2 else 0)
        px(draw, ox + 5, oy + arm_y, SKIN)
        px(draw, ox + 10, oy + arm_y, SKIN)
        if jumping:
            px(draw, ox + 5, oy + 5, SKIN)
            px(draw, ox + 10, oy + 5, SKIN)

        # Legs
        if jumping:
            # Tucked legs
            px(draw, ox + 6, oy + 8, PANTS)
            px(draw, ox + 9, oy + 8, PANTS)
            px(draw, ox + 6, oy + 9, BOOTS)
            px(draw, ox + 9, oy + 9, BOOTS)
        else:
            leg_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 1), (1, 0)]
            lo = leg_offsets[walk_phase % 6]
            # Left leg
            for ly in range(8, 10):
                px(draw, ox + 7, oy + ly + lo[0], PANTS)
            px(draw, ox + 7, oy + 10 + lo[0], BOOTS)
            # Right leg
            for ly in range(8, 10):
                px(draw, ox + 8, oy + ly + lo[1], PANTS)
            px(draw, ox + 8, oy + 10 + lo[1], BOOTS)

        # Sword (small, at right hand)
        if not jumping:
            sword_y = arm_y
            px(draw, ox + 11, oy + sword_y, (200, 200, 210))
            px(draw, ox + 11, oy + sword_y - 1, (220, 220, 230))
            px(draw, ox + 11, oy + sword_y - 2, (240, 240, 250))

    # Idle (4 frames): subtle cape movement
    for i in range(4):
        draw_hero(i * BASE, 0, walk_phase=i, jumping=False)

    # Run (6 frames): legs alternate
    for i in range(6):
        draw_hero((4 + i) * BASE, 0, walk_phase=i, jumping=False)

    # Jump (2 frames)
    for i in range(2):
        draw_hero((10 + i) * BASE, 0, walk_phase=i, jumping=True)

    img.save(os.path.join(SPRITES_DIR, "player.png"))
    print("Created player.png (12 frames, hero with red cape)")


# ==========================================
# TILESET: 5 tiles, 16x16 art each
# 0=grass_ground, 1=stone_platform, 2=question_block, 3=gold_block, 4=used_block
# ==========================================
def generate_tileset():
    img = make_img(BASE * 5, BASE)
    draw = ImageDraw.Draw(img)

    # --- Tile 0: Grass Ground ---
    ox = 0
    # Grass top (3 rows)
    for x in range(16):
        px(draw, ox + x, 0, GRASS_LIGHT if x % 3 == 0 else GRASS_TOP)
        px(draw, ox + x, 1, GRASS_TOP)
        # Grass blades
        if x % 4 == 1:
            px(draw, ox + x, 0, GRASS_LIGHT)
    # Dirt
    for x in range(16):
        for y in range(2, 16):
            c = DIRT_DARK if (x + y) % 7 == 0 else (DIRT_LIGHT if (x * 3 + y * 5) % 11 == 0 else DIRT)
            px(draw, ox + x, y, c)

    # --- Tile 1: Stone Platform ---
    ox = BASE
    for x in range(16):
        for y in range(16):
            if y == 0:
                px(draw, ox + x, y, STONE_LIGHT)
            elif y == 1:
                px(draw, ox + x, y, STONE_LIGHT)
            elif y == 15:
                px(draw, ox + x, y, STONE_DARK)
            else:
                px(draw, ox + x, y, STONE)
    # Brick lines
    for y in [5, 10]:
        for x in range(16):
            px(draw, ox + x, y, STONE_DARK)
    # Vertical lines
    for x in [0, 5, 11]:
        for y in range(2, 15):
            if y not in [5, 10]:
                px(draw, ox + x, y, STONE_DARK)

    # --- Tile 2: Question Block ---
    ox = BASE * 2
    # Border
    for x in range(16):
        for y in range(16):
            if x == 0 or x == 15 or y == 0 or y == 15:
                px(draw, ox + x, y, QB_DARK)
            elif x == 1 or y == 1:
                px(draw, ox + x, y, QB_LIGHT)
            else:
                px(draw, ox + x, y, QB_ORANGE)
    # "?" mark
    q_pixels = [(6, 4), (7, 4), (8, 4), (9, 4), (10, 4),
                (5, 5), (10, 5),
                (9, 6), (10, 6),
                (8, 7), (9, 7),
                (7, 8), (8, 8),
                (7, 9),
                (7, 11), (8, 11)]
    for qx, qy in q_pixels:
        px(draw, ox + qx, qy, QB_LIGHT)
    # Corner rivets
    for rx, ry in [(2, 2), (13, 2), (2, 13), (13, 13)]:
        px(draw, ox + rx, ry, QB_LIGHT)

    # --- Tile 3: Gold Block (correct answer) ---
    ox = BASE * 3
    for x in range(16):
        for y in range(16):
            if x == 0 or x == 15 or y == 0 or y == 15:
                px(draw, ox + x, y, GREEN_DARK)
            elif x == 1 or y == 1:
                px(draw, ox + x, y, GREEN_LIGHT)
            else:
                px(draw, ox + x, y, GREEN_OK)
    # Checkmark
    check = [(5, 9), (6, 10), (7, 11), (8, 10), (9, 9), (10, 8), (11, 7)]
    for cx, cy in check:
        px(draw, ox + cx, cy, GREEN_LIGHT)
        px(draw, ox + cx, cy - 1, GREEN_LIGHT)

    # --- Tile 4: Used Block (wrong answer) ---
    ox = BASE * 4
    ub = (100, 90, 90)
    ub_d = (70, 60, 60)
    ub_l = (140, 130, 130)
    for x in range(16):
        for y in range(16):
            if x == 0 or x == 15 or y == 0 or y == 15:
                px(draw, ox + x, y, ub_d)
            elif x == 1 or y == 1:
                px(draw, ox + x, y, ub_l)
            else:
                px(draw, ox + x, y, ub)
    # X mark
    for i in range(8):
        px(draw, ox + 4 + i, 4 + i, ub_l)
        px(draw, ox + 11 - i, 4 + i, ub_l)

    img.save(os.path.join(SPRITES_DIR, "tiles.png"))
    print("Created tiles.png (5 tiles: grass, stone, question, gold, used)")


# ==========================================
# COIN: 6 frames spinning, 16x16
# ==========================================
def generate_coin():
    n_frames = 6
    img = make_img(BASE * n_frames, BASE)
    draw = ImageDraw.Draw(img)

    widths = [5, 4, 3, 1, 3, 4]
    for f in range(n_frames):
        ox = f * BASE
        w = widths[f]
        center = 8
        # Draw coin as ellipse
        for y in range(3, 13):
            dist = abs(y - 8)
            if dist < 5:
                hw = int(w * math.sqrt(1 - (dist / 5) ** 2))
                for x in range(center - hw, center + hw + 1):
                    if 0 <= x < 16:
                        edge = abs(x - center) >= hw - 1 or dist >= 4
                        px(draw, ox + x, y, GOLD_DARK if edge else GOLD)
        # "B" symbol on wide frames
        if w >= 4:
            for by in [6, 7, 8, 9]:
                px(draw, ox + center, by, GOLD_LIGHT)
            px(draw, ox + center + 1, 6, GOLD_LIGHT)
            px(draw, ox + center + 1, 8, GOLD_LIGHT)
            px(draw, ox + center + 1, 9, GOLD_LIGHT)
        # Sparkle
        if f == 0:
            px(draw, ox + center - 2, 4, GOLD_LIGHT)

    img.save(os.path.join(SPRITES_DIR, "coin.png"))
    print("Created coin.png (6 frames)")


# ==========================================
# FLAG: 16x16
# ==========================================
def generate_flag():
    img = make_img(BASE, BASE)
    draw = ImageDraw.Draw(img)

    # Pole
    for y in range(2, 15):
        px(draw, 4, y, (180, 180, 180))
    # Ball on top
    px(draw, 4, 1, GOLD)
    px(draw, 3, 1, GOLD)
    # Flag triangle
    for row in range(6):
        y = 2 + row
        for x in range(5, 5 + 8 - row):
            c = FLAG_RED if (x + y) % 3 != 0 else CAPE_DARK
            if x < 16:
                px(draw, x, y, c)
    # Base
    for x in range(2, 7):
        px(draw, x, 14, STONE)
        px(draw, x, 15, STONE_DARK)

    img.save(os.path.join(SPRITES_DIR, "flag.png"))
    print("Created flag.png")


# ==========================================
# HEART: 16x16
# ==========================================
def generate_heart():
    img = make_img(BASE, BASE)
    draw = ImageDraw.Draw(img)
    # Heart pattern
    pattern = [
        "  **  **  ",
        " ******** ",
        "**********",
        "**********",
        "**********",
        " ******** ",
        "  ******  ",
        "   ****   ",
        "    **    ",
    ]
    for ry, row in enumerate(pattern):
        for rx, ch in enumerate(row):
            if ch == '*':
                c = HEART_LIGHT if ry < 2 and rx < 5 else HEART_RED
                px(draw, 3 + rx, 3 + ry, c)

    img.save(os.path.join(SPRITES_DIR, "heart.png"))
    print("Created heart.png")


def generate_heart_empty():
    img = make_img(BASE, BASE)
    draw = ImageDraw.Draw(img)
    pattern = [
        "  **  **  ",
        " ******** ",
        "**********",
        "**********",
        "**********",
        " ******** ",
        "  ******  ",
        "   ****   ",
        "    **    ",
    ]
    grey = (80, 80, 100)
    grey_d = (55, 55, 75)
    for ry, row in enumerate(pattern):
        for rx, ch in enumerate(row):
            if ch == '*':
                c = grey if (rx + ry) % 2 == 0 else grey_d
                px(draw, 3 + rx, 3 + ry, c)

    img.save(os.path.join(SPRITES_DIR, "heart_empty.png"))
    print("Created heart_empty.png")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("=== Generating Crypto Cat Adventure Sprites (v2) ===\n")
    generate_background()
    generate_player()
    generate_tileset()
    generate_coin()
    generate_flag()
    generate_heart()
    generate_heart_empty()
    print(f"\n=== Done! All sprites in {SPRITES_DIR} ===")
