"""Video engine: line-art + colored picture  ->  silent 'a hand draws it step by step' video.

render(line_png, color_png, plan, hand_png, out_mp4)
plan = {"draw_seconds": 24, "bpm": 118,
        "parts":  [{"name": "head", "boxes": [[x0,y0,x1,y1], ...]}, ...]   # in drawing order
        "guides": [{"type": "ellipse", "c": [x,y], "r": [rx,ry]} |
                   {"type": "line"|"poly"|"quad", "pts": [[x,y], ...]}]}
Coordinates are in line-art pixels. parts/guides may be missing -> automatic fallback.
"""
import math, os, subprocess, tempfile, numpy as np, cv2
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 30
BLUE_INK = np.array([235, 120, 25], np.float32)   # BGR: colour of the lines being drawn now
GUIDE = np.array([240, 190, 120], np.float32)     # BGR: light blue pencil guides
PAPER_BGR = (246, 252, 255)


# ---------------------------------------------------------------- helpers
def align_color(line, color):
    """Warp the colored picture onto the line art (AI edits can shift a few pixels),
    then darken it with the line art so outlines are always crisp."""
    color = cv2.resize(color, (line.shape[1], line.shape[0]))
    try:
        a = (cv2.cvtColor(line, cv2.COLOR_BGR2GRAY) < 140).astype(np.float32)
        b = (cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) < 90).astype(np.float32)
        a, b = cv2.GaussianBlur(a, (0, 0), 3), cv2.GaussianBlur(b, (0, 0), 3)
        warp = np.eye(2, 3, dtype=np.float32)
        cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        color = cv2.warpAffine(color, warp, (line.shape[1], line.shape[0]),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderValue=(255, 255, 255))
    except cv2.error:
        pass
    color = np.minimum(color, line)                     # keep the exact outlines of the line art
    color[color.min(2) > 243] = PAPER_BGR
    return color


def skeleton_segments(ink):
    sk = skeletonize(ink)
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))
    NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    nbrs = lambda p: [(p[0]+d[0], p[1]+d[1]) for d in NB if (p[0]+d[0], p[1]+d[1]) in pix]
    junction = {p for p in pix if len(nbrs(p)) > 2}
    free = pix - junction; seen = set(); segs = []

    def walk(start):
        path = [start]; seen.add(start); cur = start
        while True:
            nx = [q for q in nbrs(cur) if q in free and q not in seen]
            if not nx: break
            nx.sort(key=lambda q: abs(q[0]-cur[0]) + abs(q[1]-cur[1]))
            cur = nx[0]; seen.add(cur); path.append(cur)
        return path
    for p in [p for p in free if sum(1 for q in nbrs(p) if q in free) <= 1]:
        if p not in seen: segs.append(walk(p))
    for p in free:
        if p not in seen: segs.append(walk(p))
    if not segs: return []
    endpts = np.array([s[0] for s in segs] + [s[-1] for s in segs])
    extra = {}
    for jp in junction:   # junction pixels ride along with the nearest segment end (no gaps)
        i = int(((endpts - np.array(jp))**2).sum(1).argmin()) % len(segs)
        extra.setdefault(i, []).append(jp)
    return [(s, extra.get(i, [])) for i, s in enumerate(segs)]


def order_group(group, cur):
    out = []; group = group[:]
    while group:
        best, bd, rev = 0, 1e18, False
        for i, (pth, _) in enumerate(group):
            for r, q in ((False, pth[0]), (True, pth[-1])):
                d = (q[0]-cur[0])**2 + (q[1]-cur[1])**2
                if d < bd: bd, best, rev = d, i, r
        pth, jx = group.pop(best); pth = pth[::-1] if rev else pth
        out.append(pth + jx); cur = pth[-1]
    return out, cur


def split_into_steps(segs, parts, n_auto=6):
    """Group skeleton segments by the plan's parts (drawing order). Smaller boxes win overlaps;
    leftovers go to the part whose box is closest. Without a plan: split by length, top to bottom."""
    parts = [p for p in (parts or []) if p.get('boxes')]
    if parts:
        area = lambda p: sum((b[2]-b[0])*(b[3]-b[1]) for b in p['boxes'])
        prio = sorted(range(len(parts)), key=lambda i: parts[i].get('priority', area(parts[i])))
        groups = [[] for _ in parts]
        for s in segs:
            y, x = s[0][len(s[0])//2]
            for gi in prio:
                if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in parts[gi]['boxes']):
                    groups[gi].append(s); break
            else:
                def dist(p):
                    return min(max(b[0]-x, 0, x-b[2])**2 + max(b[1]-y, 0, y-b[3])**2 for b in p['boxes'])
                groups[min(range(len(parts)), key=lambda i: dist(parts[i]))].append(s)
        steps = []; cur = None
        for g in groups:
            if not g: continue
            top = min((p for s, _ in g for p in s), key=lambda p: p[0])
            o, cur = order_group(g, top); steps.append(o)
        return steps
    # fallback: one continuous drawing from the top, cut into equal steps
    allp = [p for s, _ in segs for p in s]
    o, _ = order_group(segs, min(allp, key=lambda p: p[0]))
    tot = sum(len(s) for s in o); steps, cur, acc = [[]], 0, 0
    for s in o:
        steps[-1].append(s); acc += len(s)
        if acc >= tot*len(steps)/n_auto and len(steps) < n_auto: steps.append([])
    return [s for s in steps if s]


def guide_points(gd):
    t = gd.get('type')
    if t == 'ellipse':
        (cx, cy), (rx, ry) = gd['c'], gd['r']
        return [(cy + ry*math.sin(a), cx + rx*math.cos(a)) for a in np.linspace(-math.pi*0.9, math.pi*1.1, 160)]
    pts = gd.get('pts') or []
    if t == 'quad' and len(pts) == 3:
        (x0, y0), (x1, y1), (x2, y2) = pts
        return [((1-u)**2*y0+2*(1-u)*u*y1+u*u*y2, (1-u)**2*x0+2*(1-u)*u*x1+u*u*x2) for u in np.linspace(0, 1, 120)]
    if len(pts) < 2: return []
    pts = pts + ([pts[0]] if t == 'poly' else [])
    out = []
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        n = int(math.hypot(xb-xa, yb-ya)/4) + 2
        out += [(ya+(yb-ya)*u, xa+(xb-xa)*u) for u in np.linspace(0, 1, n)]
    return out


def load_hand(path, scale=0.75):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = im[..., :3].astype(np.float32); a = im[..., 3].astype(np.float32)/255
    ys_, xs_ = np.nonzero(a > 0.6); k = np.argmin(xs_ + ys_); tx, ty = xs_[k], ys_[k]   # marker tip
    # extend the sleeve past the photo edge (along the arm) so the arm always leaves the screen
    hh0, ww0 = a.shape; EXT = 1400; d = np.array([0.55, 0.83])
    X, Y = np.meshgrid(np.arange(ww0+EXT, dtype=np.float32), np.arange(hh0+EXT, dtype=np.float32))
    t = np.maximum(np.maximum((X-(ww0-2))/d[0], (Y-(hh0-2))/d[1]), 0)
    mx, my = (X - t*d[0]).astype(np.float32), (Y - t*d[1]).astype(np.float32)
    rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    shadow = cv2.GaussianBlur(a, (41, 41), 0)*0.28
    return rgb, a[..., None], shadow[..., None], int(tx*scale), int(ty*scale)


# ---------------------------------------------------------------- main
def render(line_png, color_png, plan, hand_png, out_mp4, seed=3):
    rng = np.random.default_rng(seed)
    plan = plan or {}
    line = cv2.imread(line_png); ih, iw = line.shape[:2]
    color = align_color(line, cv2.imread(color_png))
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink = cv2.morphologyEx((gray < 140).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    ink_alpha = np.clip((245 - gray)/200, 0, 1)*(cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)

    steps = split_into_steps(skeleton_segments(ink), plan.get('parts'))
    guides = [g for g in (guide_points(gd) for gd in plan.get('guides', [])) if g]

    # layout + background
    PW, PH = 1360, 960; PX, PY = (W-PW)//2 - 40, 58
    sc = min((PW-80)/iw, (PH-70)/ih); dw, dh = int(iw*sc), int(ih*sc)
    OX, OY = PX + (PW-dw)//2, PY + (PH-dh)//2 + 10
    bg = np.zeros((H, W, 3), np.float32)
    gx = np.linspace(0, 1, W)[None, :, None]; gy = np.linspace(0, 1, H)[:, None, None]
    c1, c2 = np.array([0.72, 0.86, 1.0]), np.array([0.80, 0.76, 0.99])
    bg[:] = (c1*(1-(gx+gy)/2) + c2*(gx+gy)/2)*255
    for _ in range(90):
        cv2.circle(bg, (int(rng.random()*W), int(rng.random()*H)), int(4+rng.random()*10), (255, 255, 255), -1, cv2.LINE_AA)
    sh = bg.copy(); cv2.rectangle(sh, (PX+14, PY+18), (PX+PW+14, PY+PH+18), (120, 100, 110), -1)
    bg = cv2.addWeighted(bg, 0.8, sh, 0.2, 0)
    cv2.rectangle(bg, (PX, PY), (PX+PW, PY+PH), PAPER_BGR, -1)
    cv2.rectangle(bg, (PX+PW//2-75, PY-18), (PX+PW//2+75, PY+26), (115, 90, 245), -1)
    TW = 250; th_ = int(TW*ih/iw); TX, TY = W-TW-22, 80          # "what we are drawing" in the corner
    cv2.rectangle(bg, (TX-10, TY-10), (TX+TW+10, TY+th_+10), (255, 255, 255), -1)
    bg[TY:TY+th_, TX:TX+TW] = cv2.resize(color, (TW, th_), interpolation=cv2.INTER_AREA)
    cv2.rectangle(bg, (TX+TW-40, TY-28), (TX+TW+14, TY+2), (60, 200, 250), -1)
    cv2.drawMarker(bg, (TX+TW-13, TY-13), (255, 255, 255), cv2.MARKER_STAR, 22, 3)
    bg = bg.astype(np.uint8)

    h_rgb, h_a, h_sh, hx0, hy0 = load_hand(hand_png)

    def paste_hand(frame, tip):
        x, y = int(tip[0]) - hx0, int(tip[1]) - hy0
        hh, ww = h_a.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+ww), min(H, y+hh)
        if x0 >= x1 or y0 >= y1: return
        sub = frame[y0:y1, x0:x1].astype(np.float32)
        oy, ox = 26, 18
        sa = np.zeros((y1-y0, x1-x0, 1), np.float32)
        ys0, xs0 = y0-y-oy, x0-x-ox
        src = h_sh[max(ys0, 0):y1-y-oy, max(xs0, 0):x1-x-ox]
        sa[max(-ys0, 0):max(-ys0, 0)+src.shape[0], max(-xs0, 0):max(-xs0, 0)+src.shape[1]] = src
        a = h_a[y0-y:y1-y, x0-x:x1-x]
        frame[y0:y1, x0:x1] = (sub*(1-sa)*(1-a) + h_rgb[y0-y:y1-y, x0-x:x1-x]*a).astype(np.uint8)

    def badge(frame, n):
        cv2.circle(frame, (150, 178), 66, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.circle(frame, (144, 170), 66, (242, 84, 92), -1, cv2.LINE_AA)
        cv2.circle(frame, (144, 170), 55, (255, 255, 255), 6, cv2.LINE_AA)
        s = str(n); fs = 2.4; (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, fs, 6)
        cv2.putText(frame, s, (144-tw//2, 170+th//2), cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), 6, cv2.LINE_AA)

    paper = np.full((ih, iw, 3), PAPER_BGR, np.float32)
    reveal = np.zeros((ih, iw), np.uint8); stepmap = np.zeros((ih, iw), np.uint8)
    guide_layer = np.zeros((ih, iw), np.uint8)
    DARK = np.array([30, 22, 25], np.float32)

    def composite(cur_step, guide_amt=1.0, colmask=None):
        img = paper.copy()
        if guide_amt > 0 and guides:
            ga = (cv2.GaussianBlur(guide_layer, (3, 3), 0).astype(np.float32)/255*0.55*guide_amt)[..., None]
            img = img*(1-ga) + GUIDE*ga
        m = ((reveal > 0).astype(np.float32)*ink_alpha)[..., None]
        col = np.where((stepmap == cur_step)[..., None], BLUE_INK, DARK)
        img = img*(1-m) + col*m
        if colmask is not None:
            cm = colmask.astype(np.float32)[..., None]/255
            img = img*(1-cm) + color.astype(np.float32)*cm
        return img.astype(np.uint8)

    def frame_of(pic, step=None, tip=None):
        f = bg.copy(); f[OY:OY+dh, OX:OX+dw] = cv2.resize(pic, (dw, dh), interpolation=cv2.INTER_AREA)
        if step: badge(f, step)
        if tip is not None: paste_hand(f, (OX + tip[1]*sc, OY + tip[0]*sc))
        return f
    REST = (ih*0.98, iw*1.04)

    tmp = tempfile.mkdtemp(); vid = os.path.join(tmp, 'v.mp4'); wav = os.path.join(tmp, 'm.wav')
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', vid],
                          stdin=subprocess.PIPE)
    def emit(fr, n=1):
        b = fr.tobytes()
        for _ in range(n): ff.stdin.write(b)

    # 1) intro: finished picture pops in
    for i in range(int(2.2*FPS)):
        t = i/FPS; z = min(1.0, 0.55 + 0.45*(1-math.exp(-t*7))*(1+0.12*math.exp(-t*3)*math.sin(t*14)))
        fr = bg.copy(); zw, zh = int(dw*z), int(dh*z); x, y = OX+(dw-zw)//2, OY+(dh-zh)//2
        fr[y:y+zh, x:x+zw] = cv2.resize(color, (zw, zh), interpolation=cv2.INTER_AREA); emit(fr)
    emit(frame_of(composite(0)), int(0.3*FPS))

    step_no = 0
    # 2) light pencil guide shapes
    if guides:
        step_no = 1
        per = sum(len(g) for g in guides)/(3.0*FPS); b = 0
        for g in guides:
            prev = None
            for p in g:
                q = (int(p[1]), int(p[0]))
                if prev: cv2.line(guide_layer, prev, q, 255, 5, cv2.LINE_AA)
                prev = q; b += 1
                if b >= per: b -= per; emit(frame_of(composite(step_no), step_no, p))
        emit(frame_of(composite(step_no), step_no, REST), int(0.8*FPS))

    # 3) ink, part by part (new lines blue, then black)
    tot = max(1, sum(len(s) for st in steps for s in st))
    ppf = tot/(plan.get('draw_seconds', 24)*FPS)
    for st in steps:
        step_no += 1; b = 0.0
        for s in st:
            for p in s:
                cv2.circle(reveal, (p[1], p[0]), 7, 255, -1)
                np.putmask(stepmap, (reveal > 0) & (stepmap == 0), step_no)
                b += 1
                if b >= ppf: b -= ppf; emit(frame_of(composite(step_no), step_no, p))
        emit(frame_of(composite(step_no), step_no, REST), int(0.5*FPS))
        emit(frame_of(composite(-1), step_no, REST), int(0.5*FPS))
    reveal[:] = 255; stepmap[stepmap == 0] = 99

    # 4) colouring (guides fade away)
    step_no += 1
    ysI, xsI = np.nonzero(color.max(2) < 240)
    y0, y1, x0, x1 = (ysI.min(), ysI.max(), xsI.min(), xsI.max()) if len(ysI) else (0, ih-1, 0, iw-1)
    colmask = np.zeros((ih, iw), np.uint8); rows = 7; CF = int(3.4*FPS); rowh = max(1, (y1-y0)/rows); prev = None
    for f in range(CF):
        t = f/CF*rows; r = int(t); u = t-r
        xx = x0+(x1-x0)*(u if r % 2 == 0 else 1-u); yy = y0+rowh*(r+0.5)+math.sin(f*2.1)*rowh*0.3
        cv2.ellipse(colmask, (int(xx), int(yy)), (int(rowh*0.9), int(rowh*0.75)), 0, 0, 360, 255, -1)
        if prev: cv2.line(colmask, prev, (int(xx), int(yy)), 255, int(rowh*1.4))
        prev = (int(xx), int(yy))
        emit(frame_of(composite(-1, 1-f/CF, cv2.GaussianBlur(colmask, (21, 21), 0)), step_no, (yy, xx)))
    # 5) finale with sparkles
    done = frame_of(color)
    for f in range(int(2.8*FPS)):
        fr = done.copy(); t = f/FPS
        for i, (sx, sy) in enumerate([(230, 330), (1560, 1000), (200, 880), (1720, 900), (700, 140), (1250, 1030), (980, 1020)]):
            s = max(0, math.sin(t*3+i*0.8))*34+5; colr = [(40, 200, 255), (242, 84, 92), (160, 110, 255), (150, 200, 60)][i % 4]
            pts = np.array([[sx+(s if a % 2 == 0 else s*0.35)*math.sin(a*math.pi/4+t),
                             sy-(s if a % 2 == 0 else s*0.35)*math.cos(a*math.pi/4+t)] for a in range(8)], np.int32)
            cv2.fillPoly(fr, [pts], colr, cv2.LINE_AA)
        emit(fr)
    ff.stdin.close(); ff.wait()

    dur = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', vid]))
    subprocess.run(['python3', os.path.join(HERE, 'music.py'), str(dur), wav, str(plan.get('bpm', 118))], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', vid, '-i', wav, '-c:v', 'copy', '-c:a', 'aac',
                    '-b:a', '192k', '-shortest', '-movflags', '+faststart', out_mp4], check=True)
    return dur, step_no


if __name__ == '__main__':   # python3 render.py line.png color.png plan.json|- out.mp4
    import sys, json
    plan = json.load(open(sys.argv[3])) if sys.argv[3] != '-' else None
    print(render(sys.argv[1], sys.argv[2], plan, os.path.join(HERE, 'hand.png'), sys.argv[4]))
