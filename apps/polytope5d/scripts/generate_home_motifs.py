"""Generate the decorative SVG motifs of the mathematics.art homepage.

All geometry is computed (real polyhedra / surfaces), projected and written as
static SVG so the page needs no runtime JS for them.
Usage: python scripts/generate_home_motifs.py src/assets/home
"""
import sys, math, itertools
import numpy as np

OUT = sys.argv[1]
CYAN = '#33d7f0'
ICE = '#9ff3ff'
WHITE = '#effdff'


def f(v):
    return f'{v:.1f}'.rstrip('0').rstrip('.') if abs(v) < 1e5 else str(v)


def rot(ax, ay, az):
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(P, cx, cy, scale, dist=6.0):
    """perspective; +z points towards the viewer, screen y down"""
    k = dist / (dist - P[:, 2])
    return np.stack([cx + P[:, 0] * k * scale, cy - P[:, 1] * k * scale], 1)


def glow_filter(fid, sd):
    return (f'<filter id="{fid}" x="-30%" y="-30%" width="160%" height="160%">'
            f'<feGaussianBlur stdDeviation="{sd}" result="b"/>'
            f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')


# card motifs are cropped to their painted content (+7 units for the glow);
# values measured once with getBBox() in Chrome
CROP = {
    'card-polytope.svg': (33.0, 12.3, 174.0, 185.7),
    'card-surface.svg': (7.0, 38.4, 231.2, 165.1),
    'card-compute.svg': (21.0, 15.2, 198.0, 192.8),
}


def write(name, w, h, body):
    x, y = 0, 0
    if name in CROP:
        x, y, w, h = CROP[name]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{f(x)} {f(y)} {f(w)} {f(h)}" '
           f'width="{round(w)}" height="{round(h)}" fill="none">{body}</svg>\n')
    with open(f'{OUT}/{name}', 'w') as fh:
        fh.write(svg)
    print(name, len(svg) // 1024, 'KB')


# ---------------------------------------------------------------- geometry
def icosahedron():
    p = (1 + 5 ** .5) / 2
    V = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0],
                  [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
                  [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], float)
    V /= np.linalg.norm(V[0])
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return V, F


def geodesic(freq=2, spherify=1.0):
    V, F = icosahedron()
    verts, idx, faces = [], {}, []

    def vid(p):
        key = tuple(np.round(p, 6))
        if key not in idx:
            idx[key] = len(verts)
            verts.append(p)
        return idx[key]

    for a, b, c in F:
        A, B, C = V[a], V[b], V[c]
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                p = A + (B - A) * i / freq + (C - A) * j / freq
                n = p / np.linalg.norm(p)
                grid[i, j] = vid(p * (1 - spherify) + n * spherify)
        for i in range(freq):
            for j in range(freq - i):
                faces.append((grid[i, j], grid[i + 1, j], grid[i, j + 1]))
                if i + j < freq - 1:
                    faces.append((grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]))
    return np.array(verts), faces


def edges_of(faces):
    E = set()
    for t in faces:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            E.add((min(a, b), max(a, b)))
    return sorted(E)


# ------------------------------------------------------- card 1: polytope
def card_polytope():
    W, H = 240, 200
    V, F = geodesic(2, 1.0)
    R = rot(0.42, -0.55, 0.18)
    P = V @ R.T
    S = project(P, W / 2, H / 2 - 6, 74)
    light = np.array([-0.45, 0.75, 0.55]); light /= np.linalg.norm(light)
    body = ['<defs>', glow_filter('g', 1.6),
            '<radialGradient id="floor" cx="50%" cy="50%" r="50%">'
            '<stop offset="0" stop-color="#33d7f0" stop-opacity=".34"/>'
            '<stop offset="1" stop-color="#33d7f0" stop-opacity="0"/></radialGradient>',
            '<linearGradient id="face" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{WHITE}"/><stop offset="1" stop-color="{CYAN}"/></linearGradient>',
            '</defs>',
            f'<ellipse cx="{W/2}" cy="{H-22}" rx="80" ry="13" fill="url(#floor)"/>']
    # back edges (faint), then lit front faces, then front edges
    front = []
    for t in F:
        n = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
        c = P[list(t)].mean(0)
        if np.dot(n, c) < 0:
            n = -n
        front.append(n[2] > 0)
        t_n = n / np.linalg.norm(n)
    E = edges_of(F)
    zmid = P[:, 2]
    back_paths, front_paths = [], []
    for a, b in E:
        seg = f'M{f(S[a,0])} {f(S[a,1])}L{f(S[b,0])} {f(S[b,1])}'
        (front_paths if (zmid[a] + zmid[b]) / 2 > -0.12 else back_paths).append(seg)
    body.append(f'<path d="{"".join(back_paths)}" stroke="{CYAN}" stroke-opacity=".22" stroke-width=".7"/>')
    faces_svg = []
    order = sorted(range(len(F)), key=lambda i: P[list(F[i])][:, 2].mean())
    for i in order:
        t = F[i]
        n = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
        c = P[list(t)].mean(0)
        if np.dot(n, c) < 0:
            n = -n
        n /= np.linalg.norm(n)
        if n[2] <= 0:
            continue
        lam = max(0.0, float(np.dot(n, light)))
        op = 0.03 + 0.30 * lam ** 1.6
        pts = ' '.join(f'{f(S[k,0])},{f(S[k,1])}' for k in t)
        faces_svg.append(f'<polygon points="{pts}" fill="url(#face)" fill-opacity="{op:.3f}"/>')
    body.extend(faces_svg)
    body.append(f'<g filter="url(#g)"><path d="{"".join(front_paths)}" stroke="{ICE}" stroke-opacity=".9" '
                f'stroke-width=".95" stroke-linejoin="round"/></g>')
    # vertices: front ones as dots, a few brighter
    dots = []
    for k in range(len(V)):
        if P[k, 2] > 0.05:
            r = 1.0 + 1.1 * P[k, 2]
            dots.append(f'<circle cx="{f(S[k,0])}" cy="{f(S[k,1])}" r="{r:.2f}" fill="{WHITE}" fill-opacity="{0.55+0.4*P[k,2]:.2f}"/>')
    body.append(f'<g filter="url(#g)">{"".join(dots)}</g>')
    write('card-polytope.svg', W, H, ''.join(body))


# ------------------------------------------- card 2: saddle (computational geometry)
def card_surface():
    W, H = 240, 200
    N = 16
    u = np.linspace(-1, 1, N + 1)
    X, Y = np.meshgrid(u, u)
    Z = 0.62 * (X ** 2 - Y ** 2)
    P0 = np.stack([X.ravel(), Z.ravel(), Y.ravel()], 1)        # y is up
    R = rot(0.50, 0.62, 0.0)
    P = P0 @ R.T
    S = project(P, W / 2, H / 2 + 4, 82, dist=5.5).reshape(N + 1, N + 1, 2)
    Pz = P[:, 2].reshape(N + 1, N + 1)
    Pg = P.reshape(N + 1, N + 1, 3)
    light = np.array([-0.3, 0.8, 0.5]); light /= np.linalg.norm(light)
    zmin, zmax = Pz.min(), Pz.max()
    body = ['<defs>', glow_filter('g', 1.8), glow_filter('g2', 3.2),
            '<radialGradient id="floor" cx="50%" cy="50%" r="50%">'
            '<stop offset="0" stop-color="#33d7f0" stop-opacity=".26"/>'
            '<stop offset="1" stop-color="#33d7f0" stop-opacity="0"/></radialGradient>',
            '</defs>',
            f'<ellipse cx="{W/2}" cy="{H-20}" rx="88" ry="12" fill="url(#floor)"/>']
    quads = []
    for i in range(N):
        for j in range(N):
            c = [Pg[i, j], Pg[i, j + 1], Pg[i + 1, j + 1], Pg[i + 1, j]]
            n = np.cross(c[1] - c[0], c[3] - c[0]); n /= np.linalg.norm(n)
            if n[2] < 0:
                n = -n
            lam = max(0.0, float(np.dot(n, light)))
            depth = (np.mean([q[2] for q in c]) - zmin) / (zmax - zmin)
            op = (0.02 + 0.17 * lam ** 2) * (0.35 + 0.65 * depth)
            pts = ' '.join(f'{f(S[a,b,0])},{f(S[a,b,1])}' for a, b in ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)))
            quads.append((np.mean([q[2] for q in c]), f'<polygon points="{pts}" fill="{ICE}" fill-opacity="{op:.3f}"/>'))
    quads.sort(key=lambda q: q[0])
    body.extend(q[1] for q in quads)
    # grid lines, opacity by depth
    lines = []
    for i in range(N + 1):
        for direction in (0, 1):
            pts = S[i, :, :] if direction == 0 else S[:, i, :]
            zs = Pz[i, :] if direction == 0 else Pz[:, i]
            for k in range(N):
                d = ((zs[k] + zs[k + 1]) / 2 - zmin) / (zmax - zmin)
                op = 0.16 + 0.62 * d
                lines.append(f'<path d="M{f(pts[k,0])} {f(pts[k,1])}L{f(pts[k+1,0])} {f(pts[k+1,1])}" '
                             f'stroke-opacity="{op:.2f}"/>')
    body.append(f'<g stroke="{ICE}" stroke-width=".7" filter="url(#g)">{"".join(lines)}</g>')
    # bright front boundary curve (the "glowing" edge)
    edge = S[:, N, :] if Pz[:, N].mean() > Pz[:, 0].mean() else S[:, 0, :]
    d = 'M' + 'L'.join(f'{f(x)} {f(y)}' for x, y in edge)
    body.append(f'<path d="{d}" stroke="{WHITE}" stroke-width="1.6" filter="url(#g2)" stroke-linecap="round"/>')
    edge2 = S[N, :, :] if Pz[N, :].mean() > Pz[0, :].mean() else S[0, :, :]
    d2 = 'M' + 'L'.join(f'{f(x)} {f(y)}' for x, y in edge2)
    body.append(f'<path d="{d2}" stroke="{CYAN}" stroke-width="1.1" stroke-opacity=".85" filter="url(#g)"/>')
    write('card-surface.svg', W, H, ''.join(body))


# ------------------------------------------- card 3: cube cluster (compute)
def card_compute():
    W, H = 240, 200
    cubes = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0),
             (0, 0, 1), (1, 0, 1), (0, 1, 1), (0, 0, 2)]
    s = 0.84  # cube edge (gap = 1 - s)
    c30, s30 = math.cos(math.pi / 6), 0.5
    unit = 34

    def iso(x, y, z):
        return (W / 2 + (x - y) * c30 * unit, 116 + (x + y) * s30 * unit - z * unit)

    body = ['<defs>', glow_filter('g', 1.4),
            '<radialGradient id="floor" cx="50%" cy="50%" r="50%">'
            '<stop offset="0" stop-color="#33d7f0" stop-opacity=".3"/>'
            '<stop offset="1" stop-color="#33d7f0" stop-opacity="0"/></radialGradient>',
            f'<linearGradient id="top" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{WHITE}" stop-opacity=".46"/>'
            f'<stop offset="1" stop-color="{ICE}" stop-opacity=".18"/></linearGradient>',
            f'<linearGradient id="left" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{ICE}" stop-opacity=".26"/>'
            f'<stop offset="1" stop-color="{CYAN}" stop-opacity=".06"/></linearGradient>',
            f'<linearGradient id="right" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{CYAN}" stop-opacity=".16"/>'
            f'<stop offset="1" stop-color="{CYAN}" stop-opacity=".02"/></linearGradient>',
            '</defs>']
    # shift so the cluster is centred: x-y spans [-3,3] -> fine; z up to 3
    gx, gy = iso(1.5, 1.5, 0)
    body.append(f'<ellipse cx="{f(gx)}" cy="{f(gy+14)}" rx="92" ry="20" fill="url(#floor)"/>')
    for (x, y, z) in sorted(cubes, key=lambda c: (c[0] + c[1] + c[2], c[2])):
        x0, y0, z0 = x + (1 - s) / 2, y + (1 - s) / 2, z
        # visible faces for viewer at (+x,+y,+z): top (z+s), front-left (y+s plane), front-right (x+s plane)
        top = [iso(x0, y0, z0 + s), iso(x0 + s, y0, z0 + s), iso(x0 + s, y0 + s, z0 + s), iso(x0, y0 + s, z0 + s)]
        left = [iso(x0, y0 + s, z0 + s), iso(x0 + s, y0 + s, z0 + s), iso(x0 + s, y0 + s, z0), iso(x0, y0 + s, z0)]
        right = [iso(x0 + s, y0, z0 + s), iso(x0 + s, y0 + s, z0 + s), iso(x0 + s, y0 + s, z0), iso(x0 + s, y0, z0)]
        for pts, fill in ((left, 'url(#left)'), (right, 'url(#right)'), (top, 'url(#top)')):
            body.append(f'<polygon points="{" ".join(f"{f(a)},{f(b)}" for a, b in pts)}" fill="#141c1f" fill-opacity=".55"/>')
            body.append(f'<polygon points="{" ".join(f"{f(a)},{f(b)}" for a, b in pts)}" fill="{fill}"/>')
        # hidden back edges, faint
        hb = [iso(x0, y0, z0), iso(x0 + s, y0, z0), iso(x0, y0 + s, z0), iso(x0, y0, z0 + s)]
        body.append(f'<path d="M{f(hb[0][0])} {f(hb[0][1])}L{f(hb[1][0])} {f(hb[1][1])}M{f(hb[0][0])} {f(hb[0][1])}'
                    f'L{f(hb[2][0])} {f(hb[2][1])}M{f(hb[0][0])} {f(hb[0][1])}L{f(hb[3][0])} {f(hb[3][1])}" '
                    f'stroke="{CYAN}" stroke-opacity=".22" stroke-width=".6"/>')
        outline = [top[0], top[1], right[3], right[2], left[3], top[3], top[0]]
        inner = [top[2], right[2]]
        d = 'M' + 'L'.join(f'{f(a)} {f(b)}' for a, b in outline)
        d += f'M{f(top[1][0])} {f(top[1][1])}L{f(top[2][0])} {f(top[2][1])}L{f(top[3][0])} {f(top[3][1])}'
        d += f'M{f(inner[0][0])} {f(inner[0][1])}L{f(inner[1][0])} {f(inner[1][1])}'
        body.append(f'<path d="{d}" stroke="{ICE}" stroke-width=".95" stroke-linejoin="round" filter="url(#g)"/>')
        tx, ty = top[2]
        body.append(f'<circle cx="{f(tx)}" cy="{f(ty)}" r="1.3" fill="{WHITE}"/>')
    write('card-compute.svg', W, H, ''.join(body))


# ------------------------------------------- hero: wireframe icosahedron
def hero_icosahedron():
    W = H = 520
    V, F = icosahedron()
    R = rot(0.62, 0.38, 0.28)
    P = V @ R.T
    S = project(P, W / 2, H / 2, 205, dist=7)
    body = ['<defs>', glow_filter('g', 2.4), glow_filter('g2', 5),
            '<radialGradient id="halo" cx="50%" cy="50%" r="50%">'
            '<stop offset="0" stop-color="#33d7f0" stop-opacity=".10"/>'
            '<stop offset="1" stop-color="#33d7f0" stop-opacity="0"/></radialGradient>',
            '<radialGradient id="node"><stop offset="0" stop-color="#9ff3ff" stop-opacity=".9"/>'
            '<stop offset=".35" stop-color="#33d7f0" stop-opacity=".35"/><stop offset="1" stop-color="#33d7f0" stop-opacity="0"/></radialGradient>',
            '</defs>',
            f'<circle cx="{W/2}" cy="{H/2}" r="250" fill="url(#halo)"/>',
            f'<circle cx="{W/2}" cy="{H/2}" r="238" stroke="{CYAN}" stroke-opacity=".16" stroke-width=".8"/>',
            f'<circle cx="{W/2}" cy="{H/2}" r="238" stroke="{CYAN}" stroke-opacity=".38" stroke-width="1" '
            f'stroke-dasharray="1 7" stroke-linecap="round" transform="rotate(-20 {W/2} {H/2})"/>']
    # string-art hatching on front faces
    hatch = []
    for t in F:
        n = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
        if np.dot(n, P[list(t)].mean(0)) < 0:
            n = -n
        if n[2] <= 0:
            continue
        A, B, C = (S[k] for k in t)
        k = 9
        for i in range(1, k):
            a = A + (B - A) * i / k
            b = A + (C - A) * i / k
            hatch.append(f'M{f(a[0])} {f(a[1])}L{f(b[0])} {f(b[1])}')
    body.append(f'<path d="{"".join(hatch)}" stroke="{ICE}" stroke-opacity=".2" stroke-width=".6"/>')
    E = edges_of(F)
    back, front = [], []
    for a, b in E:
        seg = f'M{f(S[a,0])} {f(S[a,1])}L{f(S[b,0])} {f(S[b,1])}'
        (front if (P[a, 2] + P[b, 2]) / 2 > -0.2 else back).append(seg)
    body.append(f'<path d="{"".join(back)}" stroke="{CYAN}" stroke-opacity=".35" stroke-width="1" stroke-dasharray="2 5"/>')
    body.append(f'<g filter="url(#g)"><path d="{"".join(front)}" stroke="{ICE}" stroke-opacity=".95" stroke-width="1.5"/></g>')
    halos, nodes = [], []
    for k in range(len(V)):
        z = P[k, 2]
        r = 2.2 + 1.8 * max(0, z)
        halos.append(f'<circle cx="{f(S[k,0])}" cy="{f(S[k,1])}" r="{r*4.2:.1f}" fill="url(#node)" fill-opacity="{0.35+0.65*max(0,z):.2f}"/>')
        nodes.append(f'<circle cx="{f(S[k,0])}" cy="{f(S[k,1])}" r="{r:.1f}" fill="{WHITE}" fill-opacity="{0.55+0.45*max(0,z):.2f}"/>')
    body.append(''.join(halos))
    body.append(f'<g filter="url(#g)">{"".join(nodes)}</g>')
    write('hero-icosahedron.svg', W, H, ''.join(body))


# ------------------------------------------- hero: mesh wave (bottom-left)
def hero_mesh():
    W, H = 1400, 360
    nu, nv = 44, 16
    u = np.linspace(-1, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    U, Vv = np.meshgrid(u, v)
    Xw = U * 3.4
    Zw = Vv * 2.6 - 1.3                   # depth
    Yw = 0.34 * np.sin(1.25 * Xw + 0.9 * Zw) + 0.18 * np.cos(2.1 * Zw - 0.6 * Xw)
    P = np.stack([Xw.ravel(), Yw.ravel(), -Zw.ravel()], 1)
    R = rot(0.38, 0.0, -0.05)
    P = P @ R.T
    S = project(P, W * 0.46, H * 0.56, 230, dist=6.4).reshape(nv + 1, nu + 1, 2)
    body = ['<defs>', glow_filter('g', 1.6),
            f'<linearGradient id="fadeX" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{W}" y2="0">'
            f'<stop offset="0" stop-color="{ICE}" stop-opacity=".95"/>'
            f'<stop offset=".45" stop-color="{CYAN}" stop-opacity=".65"/>'
            f'<stop offset=".85" stop-color="{CYAN}" stop-opacity="0"/></linearGradient>',
            '</defs>']
    lines = []
    for j in range(nv + 1):        # lines along x, fading with depth
        depth = j / nv
        op = 0.55 * (1 - depth) ** 1.3 + 0.05
        d = 'M' + 'L'.join(f'{f(x)} {f(y)}' for x, y in S[j, :, :])
        lines.append(f'<path d="{d}" stroke-opacity="{op:.2f}" stroke-width="{0.6 + 0.7*(1-depth):.2f}"/>')
    for i in range(0, nu + 1):     # lines in depth direction
        d = 'M' + 'L'.join(f'{f(x)} {f(y)}' for x, y in S[:, i, :])
        lines.append(f'<path d="{d}" stroke-opacity=".22" stroke-width=".55"/>')
    body.append(f'<g stroke="url(#fadeX)" filter="url(#g)">{"".join(lines)}</g>')
    write('hero-mesh.svg', W, H, ''.join(body))


card_polytope()
card_surface()
card_compute()
hero_icosahedron()
hero_mesh()
