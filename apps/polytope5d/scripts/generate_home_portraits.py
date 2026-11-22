"""Normalize the three team cut-outs for the homepage hero: same face scale, face centred,
eyes on a common line, transparent padding, exported as WebP (alpha).
Also writes small square WebP thumbnails for the team section.
Usage: python scripts/generate_home_portraits.py public/team public/home"""
import sys
from PIL import Image

src_dir, out_dir = sys.argv[1], sys.argv[2]

# measured in the 1254x1254 originals: eye-midpoint (x, y), inter-eye distance
faces = {
    'eldar':   dict(cx=725, eye=465, iod=120),
    'malik':   dict(cx=610, eye=445, iod=120),
    'andreas': dict(cx=620, eye=450, iod=112),
}
REF_IOD = 120
CW, CH = 1200, 1120          # canvas in reference units
EYE_X, EYE_Y = 600, 340      # where the eye midpoint lands
OUT_W = 880                  # exported width (≈2x the rendered size on desktop)

for name, f in faces.items():
    im = Image.open(f'{src_dir}/{name}.png').convert('RGBA')
    s = REF_IOD / f['iod']
    if abs(s - 1) > 1e-3:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    cx, ey = f['cx'] * s, f['eye'] * s
    canvas = Image.new('RGBA', (CW, CH), (0, 0, 0, 0))
    canvas.alpha_composite(im, (round(EYE_X - cx), round(EYE_Y - ey)))
    out = canvas.resize((OUT_W, round(OUT_W * CH / CW)), Image.LANCZOS)
    out.save(f'{out_dir}/{name}-hero.webp', 'WEBP', quality=86, method=6, alpha_quality=90)
    print(name, out.size)

# team-section thumbnails (circle portraits, rendered at ≤240 px)
for name in faces:
    im = Image.open(f'{src_dir}/{name}.png').convert('RGBA').resize((520, 520), Image.LANCZOS)
    im.save(f'{out_dir}/{name}-team.webp', 'WEBP', quality=86, method=6, alpha_quality=90)
    print(name, 'team', im.size)
