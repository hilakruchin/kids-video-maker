"""Gentle, happy acoustic-guitar track (original). usage: python3 guitar_music.py seconds out.wav"""
import sys, numpy as np, wave
dur = float(sys.argv[1]); out = sys.argv[2]
sr = 44100; N = int(sr*dur); mix = np.zeros(N)
rng = np.random.default_rng(5)
bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 96; beat = 60/bpm; FAST = bpm >= 110; e8 = beat/2
hz = lambda m: 440*2**((m-69)/12)

cache = {}
def pluck(m, length=1.8, bright=0.5):
    key = (m, bright)
    if key in cache: return cache[key]
    f = hz(m); P = int(sr/f); L = int(length*sr)
    y = np.zeros(L)
    buf = rng.uniform(-1, 1, P)
    for _ in range(2):  # soften the initial noise (warmer tone)
        buf = 0.5*(buf + np.roll(buf, 1))
    y[:P] = buf
    decay = 0.996 if m < 50 else 0.994
    for n in range(P, L):
        y[n] = decay*0.5*(y[n-P] + y[n-P-1])
    # body resonance: tiny lowpass + attack click removal
    y = np.convolve(y, [0.25, 0.5, 0.25], 'same')
    y *= (1-np.exp(-np.arange(L)/40))
    cache[key] = y
    return y

def add(sig, t, amp=1.0):
    i = int(t*sr)
    if i >= N: return
    j = min(N, i+len(sig)); mix[i:j] += amp*sig[:j-i]

chords = [
    [43,47,50,55,59,67],   # G
    [50,57,62,66],         # D
    [40,47,52,55,59,64],   # Em
    [48,52,55,60,64],      # C
]
# strum pattern in 8ths: (position, direction, strength)
pattern = [(0,'D',1.0),(2,'D',0.8),(3,'U',0.55),(5,'U',0.55),(6,'D',0.8),(7,'U',0.5)]
bars = int(dur/(4*beat)) + 1
for bar in range(bars):
    t0 = bar*4*beat; ch = chords[bar % 4]
    for pos, d, s in pattern:
        strings = ch if d == 'D' else list(reversed(ch))[:4]
        gap = 0.012 if d == 'D' else 0.008
        for k, m in enumerate(strings):
            add(pluck(m), t0 + pos*e8 + k*gap + rng.uniform(0, 0.003), 0.13*s*(0.85+0.3*rng.random()))
    root = ch[0]-12 if ch[0] > 45 else ch[0]
    if FAST:   # driving plucked bass on 8ths
        for e in range(8):
            add(pluck(root + (12 if e in (3, 7) else 0), 0.6), t0+e*e8, 0.24 if e % 2 == 0 else 0.16)
    else:
        add(pluck(root, 2.2), t0, 0.28); add(pluck(root, 2.2), t0+2*beat, 0.18)

# light shaker + soft kick + hand clap on 2 & 4
for i in range(int(dur/e8)+1):
    t = i*e8; L = int(0.06*sr); tt = np.arange(L)/sr
    sh = np.diff(rng.normal(0, 1, L+1))*np.exp(-tt*(60 if i % 2 else 35))
    add(sh, t, 0.035 if i % 2 else 0.05)
for b in range(int(dur/beat)+1):
    t = b*beat; L = int(0.25*sr); tt = np.arange(L)/sr
    if b % 2 == 0 or FAST:
        f = 50+60*np.exp(-tt*35); add(np.sin(2*np.pi*np.cumsum(f)/sr)*np.exp(-tt*12), t, 0.45 if FAST else 0.35)
    if b % 2 == 1:
        n = rng.normal(0, 1, int(0.12*sr)); n = np.convolve(n, [1, -0.7], 'same')
        add(n*np.exp(-np.arange(len(n))/sr*25), t, 0.16 if FAST else 0.07)

# glockenspiel whistle melody from bar 3
mel = [71,None,74,None, 79,None,78,76, 74,None,71,None, 74,None,None,None,
       71,None,74,None, 76,None,74,72, 71,None,67,None, 69,None,None,None]
for bar in range(2, bars):
    for e in range(8):
        m = mel[(bar % 4)*8 + e] if (bar % 4)*8+e < len(mel) else None
        if m:
            L = int(0.9*sr); tt = np.arange(L)/sr; f = hz(m+12)
            g = (np.sin(2*np.pi*f*tt) + 0.25*np.sin(2*np.pi*f*2.76*tt))*np.exp(-tt*4.5)
            add(g, bar*4*beat + e*e8, 0.05)

mix = np.tanh(mix*1.1); mix /= np.max(np.abs(mix)); mix *= 0.75
fi = int(0.05*sr); mix[:fi] *= np.linspace(0, 1, fi)
fo = int(2.0*sr); mix[-fo:] *= np.linspace(1, 0, fo)
with wave.open(out, 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((mix*32767).astype(np.int16).tobytes())
