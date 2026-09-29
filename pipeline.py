"""Kids Channel video maker - runs inside GitHub Actions.

python pipeline.py make <record_id>              -> out/video.mp4 (+ line.png, color.png)
python pipeline.py done <record_id> <video_url>  -> Airtable: Video attachment + Status "Video Ready"
python pipeline.py fail <record_id> <message>    -> Airtable: Status "Error" + Notes
"""
import os, sys, json, base64, time, requests, cv2, numpy as np
from render import render

AIRTABLE_TOKEN = os.environ.get('AIRTABLE_TOKEN', '')
BASE = os.environ.get('AIRTABLE_BASE_ID', '')
TABLE = os.environ.get('AIRTABLE_TABLE', '')
OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
IMAGE_MODELS = os.environ.get('IMAGE_MODELS', 'gpt-image-1.5,gpt-image-1,gpt-image-1-mini').split(',')
VISION_MODELS = os.environ.get('VISION_MODELS', 'gpt-4.1,gpt-4o,gpt-4o-mini').split(',')
OUT = 'out'
HERE = os.path.dirname(os.path.abspath(__file__))

STYLE = (" Black and white line art ONLY: bold clean black outlines of equal thickness, "
         "no shading, no gray, no color, no fills, no text, no watermark, pure white background. "
         "One original cute cartoon character, full body, centered, with wide empty margins around it.")
COLOR_PROMPT = ("Color this exact drawing with bright, cheerful flat colors like a kids cartoon. "
                "Keep every black outline exactly the same, same pose, same size, same position. "
                "No shading, no gradients, pure white background, no text.")


# ---------------------------------------------------------------- Airtable
def at_url(rid): return f'https://api.airtable.com/v0/{BASE}/{TABLE}/{rid}'
def at_get(rid):
    r = requests.get(at_url(rid), headers={'Authorization': f'Bearer {AIRTABLE_TOKEN}'}, timeout=30)
    r.raise_for_status(); return r.json()['fields']
def at_update(rid, fields):
    r = requests.patch(at_url(rid), headers={'Authorization': f'Bearer {AIRTABLE_TOKEN}'},
                       json={'fields': fields, 'typecast': True}, timeout=30)
    if r.status_code >= 400: print('Airtable error:', r.text)
    r.raise_for_status()


# ---------------------------------------------------------------- OpenAI
def oa_post(path, **kw):
    for attempt in range(3):
        r = requests.post('https://api.openai.com/v1/' + path, headers={'Authorization': f'Bearer {OPENAI_KEY}'},
                          timeout=300, **kw)
        if r.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(10*(attempt+1)); continue
        return r

def gen_line_art(prompt):
    for model in IMAGE_MODELS:
        for extra in ({'background': 'opaque'}, {}):
            r = oa_post('images/generations', json={'model': model, 'prompt': prompt + STYLE,
                                                     'size': '1536x1024', 'quality': 'medium', 'n': 1, **extra})
            if r.ok:
                print('line art model:', model); return model, base64.b64decode(r.json()['data'][0]['b64_json'])
            print(model, extra, '->', r.status_code, r.text[:300])
    raise RuntimeError('image generation failed')

def gen_color(model, line_png):
    for extra in ({'input_fidelity': 'high', 'background': 'opaque'}, {'background': 'opaque'}, {}):
        r = oa_post('images/edits', files={'image': ('line.png', line_png, 'image/png')},
                    data={'model': model, 'prompt': COLOR_PROMPT, 'size': '1536x1024', 'quality': 'medium', **extra})
        if r.ok: return base64.b64decode(r.json()['data'][0]['b64_json'])
        print('edit', extra, '->', r.status_code, r.text[:300])
    raise RuntimeError('coloring failed')

def clean_line_art(png_bytes):
    """Force pure black lines on pure white (AI sometimes adds light gray)."""
    im = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:   # transparent background -> white
        a = im[..., 3:4].astype(np.float32)/255
        im = (im[..., :3].astype(np.float32)*a + 255*(1-a)).astype(np.uint8)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = np.where(g > 200, 255, g).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def make_plan(line_img):
    """Ask a vision model where each body part is (drawing order) and which guide shapes to sketch first."""
    h, w = line_img.shape[:2]
    view = line_img.copy()
    for x in range(0, w, 100):
        cv2.line(view, (x, 0), (x, h), (255, 170, 0), 1); cv2.putText(view, str(x), (x+2, 16), 0, 0.5, (255, 0, 0), 1)
    for y in range(0, h, 100):
        cv2.line(view, (0, y), (w, y), (0, 150, 255), 1); cv2.putText(view, str(y), (2, y-3), 0, 0.5, (0, 0, 255), 1)
    ok, buf = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_url = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    ask = f"""This is black-and-white line art ({w}x{h} px) for a kids "how to draw" video. A pixel grid with labels is drawn on top to help you read coordinates (x to the right, y down).
Plan the drawing lesson:
1. "parts": split the character into 4-7 parts in the natural order a drawing teacher would draw them
   (usually head, then face details, then body, then arms/legs, then wings/tail/extras).
   For each part give "name" and "boxes": one or more rectangles [x0,y0,x1,y1] that tightly cover that part's lines.
2. "guides": 4-8 simple light construction shapes a teacher sketches first:
   {{"type":"ellipse","c":[cx,cy],"r":[rx,ry]}} for head/body, {{"type":"line","pts":[[x,y],[x,y]]}} for limbs,
   {{"type":"poly","pts":[[x,y],[x,y],[x,y]]}} for wings/ears, {{"type":"quad","pts":[[x,y],[x,y],[x,y]]}} for a tail curve.
Use real pixel coordinates of this image. Return ONLY JSON: {{"parts":[...],"guides":[...]}}"""
    for model in VISION_MODELS:
        r = oa_post('chat/completions', json={'model': model, 'response_format': {'type': 'json_object'},
                    'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': ask},
                                                              {'type': 'image_url', 'image_url': {'url': img_url}}]}]})
        if not r.ok: print(model, '->', r.status_code, r.text[:300]); continue
        try:
            plan = json.loads(r.json()['choices'][0]['message']['content'])
            return validate_plan(plan, w, h), model
        except Exception as e:
            print('plan parse error', e)
    return {}, None

def validate_plan(plan, w, h):
    cl = lambda v, m: max(0, min(m, float(v)))
    parts = []
    for p in plan.get('parts', [])[:8]:
        boxes = []
        for b in p.get('boxes', []):
            if len(b) == 4:
                x0, y0, x1, y1 = cl(b[0], w), cl(b[1], h), cl(b[2], w), cl(b[3], h)
                if x1 - x0 > 5 and y1 - y0 > 5: boxes.append([x0, y0, x1, y1])
        if boxes: parts.append({'name': str(p.get('name', '')), 'boxes': boxes})
    guides = []
    for g in plan.get('guides', [])[:10]:
        try:
            if g['type'] == 'ellipse': guides.append({'type': 'ellipse', 'c': [cl(g['c'][0], w), cl(g['c'][1], h)],
                                                      'r': [abs(float(g['r'][0])), abs(float(g['r'][1]))]})
            elif g['type'] in ('line', 'poly', 'quad'):
                guides.append({'type': g['type'], 'pts': [[cl(x, w), cl(y, h)] for x, y in g['pts']]})
        except Exception: pass
    return {'parts': parts if len(parts) >= 2 else [], 'guides': guides}


# ---------------------------------------------------------------- commands
def make(rid):
    os.makedirs(OUT, exist_ok=True)
    f = at_get(rid)
    prompt = f.get('Image Prompt') or f.get('Topic') or 'a cute baby dragon'
    at_update(rid, {'Status': 'Rendering'})
    model, raw = gen_line_art(prompt)
    line = clean_line_art(raw); cv2.imwrite(f'{OUT}/line.png', line)
    ok, enc = cv2.imencode('.png', line)
    open(f'{OUT}/color.png', 'wb').write(gen_color(model, enc.tobytes()))
    plan, vm = make_plan(line)
    plan.update({'draw_seconds': 24, 'bpm': 118})
    plan['guides'] = []   # no light-blue helper lines: only the lines of the final drawing are drawn
    json.dump(plan, open(f'{OUT}/plan.json', 'w'), indent=1)
    print('plan by', vm, ':', len(plan.get('parts', [])), 'parts,', len(plan.get('guides', [])), 'guides')
    dur, steps = render(f'{OUT}/line.png', f'{OUT}/color.png', plan, os.path.join(HERE, 'hand.png'), f'{OUT}/video.mp4')
    print(f'video ready: {dur:.1f}s, {steps} steps')

def done(rid, url):
    at_update(rid, {'Video': [{'url': url, 'filename': 'video.mp4'}], 'Status': 'Video Ready'})

def fail(rid, msg):
    at_update(rid, {'Status': 'Error', 'Notes': msg})


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'make': make, 'done': done, 'fail': fail}[cmd](*sys.argv[2:])
