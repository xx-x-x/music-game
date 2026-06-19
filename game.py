#!/usr/bin/env python3
"""
MUSIC GAME
==========
Load a track · trigger sounds with hand gestures or on-screen buttons.
Toggle sound effects and stunning visual effects (5 modes).

✌️  Peace right → slot 0    ✌️  Peace left  → slot 1
👌  OK    right → slot 2    👌  OK    left  → slot 3
✌️✌️ Both peace → slot 4

[L] load  [SPACE] play/pause  [C] config  [H] hints  [F] fullscreen  [ESC] quit
"""

import sys, os, math, time, random, threading, collections, glob, json
import pygame
import numpy as np

os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')

_BASE      = os.path.dirname(__file__)
_SAM       = os.path.join(_BASE, "sounds_and_music")
LOGO_PATH  = os.path.join(_BASE, "logo.png")
SRC_DIR    = os.path.join(_SAM,  "music")    if os.path.isdir(os.path.join(_SAM,"music"))    else os.path.join(_BASE,"src")
SOUNDS_DIR = os.path.join(_SAM,  "sound_effects") if os.path.isdir(os.path.join(_SAM,"sound_effects")) else os.path.join(_BASE,"sounds")
CFG_PATH   = os.path.join(os.path.dirname(__file__), "config.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

try:
    import miniaudio; AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
try:
    import sounddevice as sd; SD_OK = True
except ImportError:
    SD_OK = False
try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_py
    from mediapipe.tasks.python import vision as _mp_vis
    CV_OK = True
except ImportError:
    CV_OK = False

# ── screen ────────────────────────────────────────────────────────────────────
W, H = 1280, 720
FPS  = 120
SR   = 44100
BLOCK = 512

# ── palette  #011936 · #465362 · #82a3a1 · #9fc490 · #c0dfa1 ─────────────────
P0 = (  1,  25,  54)   # deep navy     (background)
P1 = ( 70,  83,  98)   # steel blue
P2 = (130, 163, 161)   # muted teal
P3 = (159, 196, 144)   # sage green
P4 = (192, 223, 161)   # light mint
PAL = [P0, P1, P2, P3, P4]

# brighter accent versions for glows
PA2 = (180, 230, 225)
PA3 = (210, 255, 185)
PA4 = (230, 255, 200)

# ── ui colours ────────────────────────────────────────────────────────────────
BG     = P0
BORDER = P1
TXT    = P4
TXTSUB = P2
CL     = P2       # left hand
CR     = P3       # right hand
CEFF   = P4       # effect accent
YEL    = PA4
WHT    = (240, 250, 245)
RED    = (220,  60,  60)
GRN    = P3

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17),
]

# ── gesture slots ─────────────────────────────────────────────────────────────
GESTURE_SLOTS = [
    'peace_right', 'peace_left', 'ok_right', 'ok_left', 'both_peace',
    'key_1','key_2','key_3','key_4','key_5','key_6','key_7','key_8','key_9',
]
GESTURE_LABELS = {
    'peace_right': 'V-sign  RIGHT',
    'peace_left':  'V-sign  LEFT',
    'ok_right':    'OK      RIGHT',
    'ok_left':     'OK      LEFT',
    'both_peace':  'V+V  Both',
    **{f'key_{i}': f'Hotkey  [{i}]' for i in range(1,10)},
}
GESTURE_EMOJI = {
    'peace_right': 'V-R', 'peace_left': 'V-L',
    'ok_right':    'OK-R', 'ok_left':   'OK-L',
    'both_peace':  'V+V',
    **{f'key_{i}': str(i) for i in range(1,10)},
}
# pygame key constants are plain ints, safe to use before init
KEY_SLOTS = {49+i: f'key_{i+1}' for i in range(9)}   # 49='1', 50='2', ... 57='9'

VFX_NAMES = ['Nebula', 'Snowflakes', 'Shader']


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def lerp(a, b, t):
    return a + (b - a) * clamp(t)

def lerp_col(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))

def pal_col(t):
    """Sample a smooth gradient across the 5 palette colors. t ∈ [0,1]."""
    cols = [P1, P2, P3, P4, PA4, P3, P2, P1]
    t = t % 1.0
    n = len(cols) - 1
    i = int(t * n)
    f = t * n - i
    return lerp_col(cols[min(i, n-1)], cols[min(i+1, n)], f)

class Smooth:
    def __init__(self, n=8, init=0.0):
        self.buf = collections.deque([init]*n, maxlen=n)
    def __call__(self, v):
        self.buf.append(v); return sum(self.buf)/len(self.buf)

_glow_cache: dict = {}   # radius → pre-built white glow surface

def _glow(surf, cx, cy, r, color, max_alpha=180):
    """Additive soft glow — reuses cached alpha mask, tinted per call."""
    if r < 2: return
    if r not in _glow_cache:
        gs = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
        layers = min(r, 10)
        for i in range(layers):
            ri = max(1, r - i*(r//layers))
            a  = int(220 * (i/layers)**1.4)
            pygame.draw.circle(gs, (255, 255, 255, a), (r, r), ri)
        _glow_cache[r] = gs
    # tint the cached mask to the requested color at the requested alpha
    gs = _glow_cache[r]
    tinted = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
    tinted.fill((*color, 0))
    tinted.blit(gs, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    # scale alpha by max_alpha/220
    if max_alpha < 200:
        tinted.set_alpha(int(max_alpha * 255 // 220))
    surf.blit(tinted, (cx-r, cy-r), special_flags=pygame.BLEND_RGBA_ADD)


# ─────────────────────────────────────────────────────────────────────────────
# Sound effects
# ─────────────────────────────────────────────────────────────────────────────

def _load_wav(path):
    src = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32,
                                nchannels=1, sample_rate=SR)
    arr = np.frombuffer(src.samples, dtype=np.float32).copy()
    peak = np.abs(arr).max()
    if peak > 0: arr /= peak
    return arr * 0.85

def _scan_sounds():
    exts = ('.wav','.mp3','.ogg','.flac')
    return sorted(f for f in glob.glob(os.path.join(SOUNDS_DIR,'*'))
                  if os.path.splitext(f)[1].lower() in exts)

EFFECT_FILES = []; EFFECT_NAMES = []; _effect_cache = {}

def _init_effects():
    global EFFECT_FILES, EFFECT_NAMES
    EFFECT_FILES = _scan_sounds()
    EFFECT_NAMES = [os.path.splitext(os.path.basename(p))[0][:24] for p in EFFECT_FILES]
    if not EFFECT_FILES:
        print(f'WARNING: no sound files in {SOUNDS_DIR}')

def get_effect(idx):
    if not EFFECT_FILES: return np.zeros(1, dtype=np.float32)
    idx = idx % len(EFFECT_FILES)
    if idx not in _effect_cache:
        try:   _effect_cache[idx] = _load_wav(EFFECT_FILES[idx])
        except Exception as e:
            print(f'effect load error: {e}')
            _effect_cache[idx] = np.zeros(1, dtype=np.float32)
    return _effect_cache[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    'peace_right':0,'peace_left':1,'ok_right':2,'ok_left':3,'both_peace':4,
    **{f'key_{i}': i for i in range(1,10)},
}

def load_config():
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH) as f: data = json.load(f)
            old_map = {'right_punch':'peace_right','left_punch':'peace_left',
                       'right_pinch':'ok_right','left_pinch':'ok_left','both_up':'both_peace'}
            migrated = {new: data[old] for old,new in old_map.items() if old in data}
            merged = {**DEFAULT_CFG, **migrated}
            for k in GESTURE_SLOTS:
                if k in data: merged[k] = data[k]
            return merged
        except Exception: pass
    return dict(DEFAULT_CFG)

def save_config(cfg):
    with open(CFG_PATH,'w') as f: json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Engine
# ─────────────────────────────────────────────────────────────────────────────

class AudioEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = None; self._pos = 0
        self._playing = False; self._title = ''; self._duration = 0
        self._sfx = []; self._stream = None

    def load(self, path):
        if not AUDIO_OK: return False
        try:
            src = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32,
                                        nchannels=2, sample_rate=SR)
            raw = np.frombuffer(src.samples, dtype=np.float32).reshape(-1, 2)
            with self._lock:
                self._data = raw; self._pos = 0; self._duration = len(raw)
            self._title = os.path.splitext(os.path.basename(path))[0]
            return True
        except Exception as e: print(f'load error: {e}'); return False

    def toggle_play(self):
        with self._lock: self._playing = not self._playing

    def seek(self, frac):
        with self._lock:
            if self._data is not None: self._pos = int(clamp(frac)*self._duration)

    def play_sfx(self, idx):
        samp = get_effect(idx)
        with self._lock: self._sfx.append([samp, 0])

    @property
    def loaded(self): return self._data is not None
    @property
    def playing(self): return self._playing
    @property
    def title(self): return self._title

    def progress(self):
        with self._lock:
            return (self._pos/self._duration) if self._duration else 0.0

    def duration_s(self): return self._duration/SR
    def position_s(self):
        with self._lock: return self._pos/SR

    def start(self):
        if not SD_OK: return
        self._stream = sd.OutputStream(samplerate=SR, channels=2, dtype='float32',
                                       blocksize=BLOCK, callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream: self._stream.stop(); self._stream.close()

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            playing=self._playing; data=self._data; pos=self._pos
            dur=self._duration
            sfx_list=list(self._sfx)   # snapshot — avoid mutation during iteration
        block = np.zeros((frames,2), dtype=np.float32)
        if playing and data is not None:
            end=min(pos+frames,dur); n=end-pos
            if n>0: block[:n]=data[pos:end]
            new_pos=pos+frames
            if new_pos>=dur: new_pos=0
            with self._lock: self._pos=new_pos
        dead=[]
        for item in sfx_list:
            samp,off=item[0],item[1]; n=min(frames,len(samp)-off)
            if n>0:
                slc=samp[off:off+n]
                block[:n,0]+=slc; block[:n,1]+=slc; item[1]+=n
            if item[1]>=len(samp): dead.append(item)
        if dead:
            with self._lock:
                for d in dead:
                    try: self._sfx.remove(d)
                    except ValueError: pass
        np.clip(block,-1.0,1.0,out=block); outdata[:]=block


# ─────────────────────────────────────────────────────────────────────────────
# Hand Tracker
# ─────────────────────────────────────────────────────────────────────────────

class HandTracker:
    def __init__(self):
        self._results=None; self._frame=None
        self._lock=threading.Lock(); self._running=False

    def start(self):
        self._running=True; threading.Thread(target=self._run,daemon=True).start()
    def stop(self): self._running=False

    def get(self):
        with self._lock: return self._results, self._frame

    def _run(self):
        if not CV_OK: return
        base_opts=_mp_py.BaseOptions(model_asset_path=MODEL_PATH)
        opts=_mp_vis.HandLandmarkerOptions(
            base_options=base_opts, running_mode=_mp_vis.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        detector=_mp_vis.HandLandmarker.create_from_options(opts)
        ts=0; fail_streak=0
        while self._running:
            cap=cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release(); time.sleep(1.0); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,H)
            fail_streak=0
            while self._running:
                ok,frame=cap.read()
                if not ok:
                    fail_streak+=1
                    if fail_streak>10:   # camera disconnected — break to outer loop to reopen
                        break
                    time.sleep(0.05); continue
                fail_streak=0
                frame=cv2.flip(frame,1)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                ts+=33
                try:
                    result=detector.detect_for_video(mp_img,ts)
                except (RuntimeError, Exception):
                    break   # executor shut down — exit inner loop cleanly
                hands={}
                if result.hand_landmarks:
                    for i,lm_list in enumerate(result.hand_landmarks):
                        label=result.handedness[i][0].category_name
                        pts=[(lm.x,lm.y,lm.z) for lm in lm_list]
                        hands[label]=pts
                with self._lock: self._results=hands; self._frame=frame
            cap.release()
            if self._running: time.sleep(0.5)   # brief pause before retry


# ─────────────────────────────────────────────────────────────────────────────
# Gesture Detector  —  ✌️ Peace  and  👌 OK
# ─────────────────────────────────────────────────────────────────────────────

def _up(lm, tip, pip): return lm[tip][1] < lm[pip][1]

def _is_peace(lm):
    return (_up(lm,8,6) and _up(lm,12,10) and
            not _up(lm,16,14) and not _up(lm,20,18))

def _is_ok(lm):
    tx,ty=lm[4][0],lm[4][1]; ix,iy=lm[8][0],lm[8][1]
    return (math.hypot(tx-ix,ty-iy)<0.08 and
            _up(lm,12,10) and _up(lm,16,14) and _up(lm,20,18))

class GestureDetector:
    COOLDOWN = 0.35
    def __init__(self):
        self._peace={'Left':False,'Right':False}
        self._ok={'Left':False,'Right':False}
        self._last_t=collections.defaultdict(float)
        self.fired=[]; self.active_gesture={'Left':None,'Right':None}

    def _fire(self, slot, now):
        if now-self._last_t[slot]>self.COOLDOWN:
            self.fired.append(slot); self._last_t[slot]=now

    def update(self, hands):
        now=time.time(); self.fired=[]
        lm_l=hands.get('Left'); lm_r=hands.get('Right')
        pl=_is_peace(lm_l) if lm_l else False
        pr=_is_peace(lm_r) if lm_r else False
        ol=_is_ok(lm_l)    if lm_l else False
        or_=_is_ok(lm_r)   if lm_r else False

        if pr and not self._peace['Right']: self._fire('peace_right',now)
        if pl and not self._peace['Left']:  self._fire('peace_left',now)
        if or_ and not self._ok['Right']:   self._fire('ok_right',now)
        if ol  and not self._ok['Left']:    self._fire('ok_left',now)
        if pl and pr and not (self._peace['Left'] and self._peace['Right']):
            self._fire('both_peace',now)

        self._peace['Left']=pl; self._peace['Right']=pr
        self._ok['Left']=ol;    self._ok['Right']=or_

        for side,lm in [('Left',lm_l),('Right',lm_r)]:
            if lm is None: self.active_gesture[side]=None
            elif _is_peace(lm): self.active_gesture[side]='V'
            elif _is_ok(lm):    self.active_gesture[side]='OK'
            else:               self.active_gesture[side]=None


# ─────────────────────────────────────────────────────────────────────────────
# VFX Engine  —  5 visual modes
# ─────────────────────────────────────────────────────────────────────────────

class VFXEngine:
    MAX_PARTICLES = 600
    TRAIL_LEN     = 70

    def __init__(self):
        self.mode = 0
        self._t   = 0.0
        self._particles = []   # each: [x,y,vx,vy,life,max_life,col,size]
        # pre-allocated reusable full-screen SRCALPHA layer (cleared each frame)
        self._lay = pygame.Surface((W, H), pygame.SRCALPHA)

        # persistent star field for nebula mode
        rng = random.Random(42)
        self._stars = [(rng.randint(0,W), rng.randint(0,H),
                        rng.uniform(0.3,1.0)) for _ in range(160)]

        # snowflakes: each is [x, y, size, speed, angle, angle_vel, col_t]
        # spawned once and loop forever (wrap at bottom)
        self._flakes = []
        self._init_flakes()

    # ── public interface ──────────────────────────────────────────────────────

    def next_mode(self): self.mode=(self.mode+1)%len(VFX_NAMES); self._clear()
    def prev_mode(self): self.mode=(self.mode-1)%len(VFX_NAMES); self._clear()

    def _clear(self):
        self._particles.clear()
        self._init_flakes()

    def on_gesture(self, slot, lm):
        """Called when any gesture fires. lm = hand landmark list or None."""
        if lm is None: return
        cx, cy = int(lm[9][0]*W), int(lm[9][1]*H)  # palm center (landmark 9)
        if 'peace' in slot or 'both' in slot:
            self._color_bomb(cx, cy)

    def update(self, dt, hands):
        self._t += dt
        m = VFX_NAMES[self.mode]
        if m == 'Nebula':      self._upd_nebula(dt, hands)
        elif m == 'Snowflakes': self._upd_flakes(dt, hands)
        self._particles=[p for p in self._particles if p[4]>0]
        for p in self._particles:
            p[0]+=p[2]*dt*60; p[1]+=p[3]*dt*60
            p[3]+=0.04*dt*60  # gravity
            p[2]*=0.98; p[4]-=dt

    def draw(self, surf):
        m = VFX_NAMES[self.mode]
        if m == 'Nebula':       self._drw_nebula(surf)
        elif m == 'Snowflakes': self._drw_flakes(surf)

    # ── color bomb (peace sign trigger) ─────────────────────────────────────

    def _color_bomb(self, cx, cy):
        m = VFX_NAMES[self.mode]
        if m == 'Nebula':
            for i in range(120):
                angle = random.uniform(0, math.pi*2)
                speed = random.uniform(2.0, 7.0)
                col   = pal_col(random.random())
                self._emit(cx, cy, math.cos(angle)*speed, math.sin(angle)*speed,
                           random.uniform(0.8,1.8), col, random.randint(3,8))
        elif m == 'Snowflakes':
            # burst: scatter all flakes outward from touch point then let them drift back
            for f in self._flakes:
                dx = f[0] - cx; dy = f[1] - cy
                dist = math.hypot(dx, dy) or 1
                push = random.uniform(4.0, 12.0) / dist * 120
                f[6] = dx/dist * push   # vx impulse stored in slot 6
                f[7] = dy/dist * push   # vy impulse stored in slot 7
            # also change all flake colours
            t0 = random.random()
            for i, f in enumerate(self._flakes):
                f[8] = (t0 + i/len(self._flakes)) % 1.0

    # ── nebula ────────────────────────────────────────────────────────────────

    def _upd_nebula(self, dt, hands):
        for side, lm in [('Left',hands.get('Left')),('Right',hands.get('Right'))]:
            if lm is None: continue
            cx=int(lm[9][0]*W); cy=int(lm[9][1]*H)
            col = pal_col(self._t*0.15+(0.5 if side=='Right' else 0.0))
            if len(self._particles)<self.MAX_PARTICLES:
                for _ in range(3):
                    ang  = random.uniform(0,math.pi*2)
                    sp   = random.uniform(0.3,1.5)
                    off  = random.gauss(0,20)
                    self._emit(cx+off, cy+off,
                               math.cos(ang)*sp, math.sin(ang)*sp-0.5,
                               random.uniform(0.8,2.2),
                               pal_col(random.random()),
                               random.randint(2,6))

    def _drw_nebula(self, surf):
        lay = self._lay; lay.fill((0,0,0,0))
        # star field
        for sx,sy,br in self._stars:
            a=int(br*100); pygame.draw.circle(lay,(*P2,a),(sx,sy),1)
        self._drw_particles_on(lay)
        surf.blit(lay,(0,0), special_flags=pygame.BLEND_RGBA_ADD)

    # ── snowflakes ────────────────────────────────────────────────────────────

    _N_FLAKES = 180

    def _init_flakes(self):
        # flake: [x, y, size, fall_speed, angle, angle_vel, vx, vy, col_t]
        rng = random.Random()
        self._flakes = [
            [rng.uniform(0, W),
             rng.uniform(-H, 0),          # start above screen so they trickle in
             rng.uniform(2.0, 8.0),        # size (radius)
             rng.uniform(0.4, 1.6),        # base fall speed (px/frame at 60fps)
             rng.uniform(0, math.pi*2),    # wobble angle
             rng.uniform(-0.02, 0.02),     # wobble angular velocity
             0.0,                          # vx impulse (from colour bomb)
             0.0,                          # vy impulse (from colour bomb)
             rng.random()]                 # col_t (palette position)
            for _ in range(self._N_FLAKES)
        ]

    def _upd_flakes(self, dt, hands):
        speed60 = dt * 60
        for f in self._flakes:
            # wobble angle advances
            f[4] += f[5] * speed60
            # horizontal wobble + impulse decay
            f[6] *= 0.92
            f[7] *= 0.92
            f[0] += math.sin(f[4]) * 0.6 * speed60 + f[6] * speed60
            f[1] += f[3] * speed60 + f[7] * speed60
            # wrap
            if f[1] > H + f[2]:
                f[1] = -f[2]
                f[0] = random.uniform(0, W)
            if f[0] < -f[2]:  f[0] = W + f[2]
            if f[0] > W+f[2]: f[0] = -f[2]
            # slow colour drift
            f[8] = (f[8] + dt * 0.04) % 1.0

        # hand proximity: flakes near a hand speed up and glow
        for side in ('Left', 'Right'):
            lm = hands.get(side)
            if lm is None: continue
            hx = lm[9][0]*W; hy = lm[9][1]*H
            for f in self._flakes:
                d = math.hypot(f[0]-hx, f[1]-hy)
                if d < 120:
                    pull = (1 - d/120) * 3.0
                    f[6] += (hx-f[0]) / max(d,1) * pull * dt * 60
                    f[7] += (hy-f[1]) / max(d,1) * pull * dt * 60

    def _draw_snowflake(self, surf, x, y, size, col, alpha):
        """6-armed snowflake using lines."""
        ix, iy = int(x), int(y)
        c = (*col, alpha)
        arm = int(size)
        for i in range(6):
            a = math.pi * i / 3
            ex = ix + int(math.cos(a) * arm)
            ey = iy + int(math.sin(a) * arm)
            pygame.draw.line(surf, c, (ix, iy), (ex, ey), max(1, int(size*0.22)))
            # two small branches on each arm at 60% length
            for side_sign in (-1, 1):
                bx = ix + int(math.cos(a) * arm * 0.55)
                by = iy + int(math.sin(a) * arm * 0.55)
                ba = a + side_sign * math.pi / 4
                br = arm * 0.35
                pygame.draw.line(surf, c, (bx, by),
                                 (bx + int(math.cos(ba)*br), by + int(math.sin(ba)*br)),
                                 max(1, int(size*0.15)))

    def _drw_flakes(self, surf):
        lay = self._lay; lay.fill((0,0,0,0))
        for f in self._flakes:
            x, y, size, _, _, _, _, _, col_t = f
            col = pal_col(col_t)
            # larger flakes are more opaque
            alpha = int(80 + (size / 8.0) * 140)
            self._draw_snowflake(lay, x, y, size, col, alpha)
            if size > 5:
                _glow(lay, int(x), int(y), int(size*1.8), col, int(alpha*0.4))
        surf.blit(lay, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

    # ── particle helpers ──────────────────────────────────────────────────────

    def _emit(self, x, y, vx, vy, life, col, size):
        if len(self._particles)<self.MAX_PARTICLES:
            self._particles.append([x,y,vx,vy,life,life,col,size])

    def _drw_particles(self, surf):
        lay=self._lay; lay.fill((0,0,0,0))
        self._drw_particles_on(lay)
        surf.blit(lay,(0,0),special_flags=pygame.BLEND_RGBA_ADD)

    def _drw_particles_on(self, lay):
        for x,y,vx,vy,life,max_life,col,size in self._particles:
            if max_life <= 0: continue
            t=max(0.0, min(1.0, life/max_life))
            a=int(t**0.8*220); r=max(1,int(size*t**0.3))
            safe_col=tuple(max(0,min(255,int(c))) for c in col)
            pygame.draw.circle(lay,(*safe_col,a),(int(x),int(y)),r)
            if r>3: _glow(lay,int(x),int(y),r+6,safe_col,int(a*0.4))


# shared mutable state so draw_geo_hand can access hands
_last_hands: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Toggle / mode control buttons
# ─────────────────────────────────────────────────────────────────────────────

class ControlBar:
    H_BAR  = 40
    BTN_W  = 130
    BTN_H  = 32
    GAP    = 8

    def __init__(self):
        self.sound_on = False   # start muted
        self.vfx_on   = True

    def _layout(self):
        """Two buttons pinned to the top-right corner."""
        y = 8
        return [
            ('sound', pygame.Rect(W - self.BTN_W*2 - self.GAP - 8, y, self.BTN_W, self.BTN_H),
             'SOUND', self.sound_on),
            ('vfx',   pygame.Rect(W - self.BTN_W    - 8,            y, self.BTN_W, self.BTN_H),
             'VFX',   self.vfx_on),
        ]

    def hit(self, pos):
        for id_, rect, *_ in self._layout():
            if rect.collidepoint(pos): return id_
        return None

    def draw(self, surf, font):
        for id_, rect, label, active in self._layout():
            col_bg = lerp_col(P1, P2, 0.5) if active else P1
            border  = P3 if active else BORDER
            pygame.draw.rect(surf, col_bg, rect, border_radius=6)
            pygame.draw.rect(surf, border, rect, 2, border_radius=6)
            text = label + (': ON' if active else ': OFF')
            lbl  = font.render(text, True, P4 if active else TXTSUB)
            surf.blit(lbl, (rect.centerx - lbl.get_width()//2,
                            rect.centery - lbl.get_height()//2))


class ModeBar:
    """Second row: < Nebula Shader > — centered, y=48."""
    BTN_H  = 30
    NAM_W  = 72   # width per mode name button
    ARR_W  = 28   # width of < > arrows
    GAP    = 6

    def __init__(self, vfx: VFXEngine):
        self._vfx = vfx

    def _layout(self):
        y    = 48
        # total width of all buttons
        total = self.ARR_W + self.GAP + len(VFX_NAMES)*(self.NAM_W+self.GAP) + self.ARR_W
        x0   = W//2 - total//2
        items = [('prev', pygame.Rect(x0, y, self.ARR_W, self.BTN_H), '<')]
        x0  += self.ARR_W + self.GAP
        for i, name in enumerate(VFX_NAMES):
            items.append((f'mode_{i}', pygame.Rect(x0, y, self.NAM_W, self.BTN_H), name))
            x0 += self.NAM_W + self.GAP
        items.append(('next', pygame.Rect(x0, y, self.ARR_W, self.BTN_H), '>'))
        return items

    def hit(self, pos):
        for id_, rect, _ in self._layout():
            if rect.collidepoint(pos): return id_
        return None

    def draw(self, surf, font):
        for id_, rect, label in self._layout():
            active = (id_ == f'mode_{self._vfx.mode}')
            col_bg = P2 if active else P1
            border  = P4 if active else BORDER
            pygame.draw.rect(surf, col_bg, rect, border_radius=6)
            pygame.draw.rect(surf, border, rect, 2, border_radius=6)
            lbl = font.render(label, True, P4 if active else TXTSUB)
            surf.blit(lbl, (rect.centerx - lbl.get_width()//2,
                            rect.centery - lbl.get_height()//2))


# ─────────────────────────────────────────────────────────────────────────────
# Effect pad buttons
# ─────────────────────────────────────────────────────────────────────────────

class EffectButtons:
    PAD_W=160; PAD_H=58; GAP=8

    def __init__(self, cfg):
        self.cfg=cfg; self._glow={s:0.0 for s in GESTURE_SLOTS}

    def flash(self,slot): self._glow[slot]=1.0

    def update(self,dt):
        for s in GESTURE_SLOTS: self._glow[s]=max(0.0,self._glow[s]-dt*5.0)

    def _rects(self):
        total=(self.PAD_W+self.GAP)*len(GESTURE_SLOTS)-self.GAP
        x0=W//2-total//2; y0=H-76-self.PAD_H
        return [(slot, pygame.Rect(x0+i*(self.PAD_W+self.GAP),y0,self.PAD_W,self.PAD_H))
                for i,slot in enumerate(GESTURE_SLOTS)]

    def hit_test(self,pos):
        for slot,rect in self._rects():
            if rect.collidepoint(pos): return slot
        return None

    def draw(self,surf,font):
        for slot,rect in self._rects():
            g=self._glow[slot]
            idx=self.cfg.get(slot,0)%max(1,len(EFFECT_NAMES))
            name=EFFECT_NAMES[idx] if EFFECT_NAMES else '?'
            bg=lerp_col(P1,(70,120,110),g*0.8)
            border=lerp_col(BORDER,PA3,g)
            pygame.draw.rect(surf,bg,rect,border_radius=8)
            pygame.draw.rect(surf,border,rect,2,border_radius=8)
            emoji=list(GESTURE_EMOJI.values())[GESTURE_SLOTS.index(slot)]
            lbl=font.render(emoji,True,WHT)
            surf.blit(lbl,(rect.centerx-lbl.get_width()//2,rect.y+6))
            elbl=font.render(name[:15],True,PA4 if g>0.1 else TXTSUB)
            surf.blit(elbl,(rect.centerx-elbl.get_width()//2,rect.y+30))


# ─────────────────────────────────────────────────────────────────────────────
# File Picker
# ─────────────────────────────────────────────────────────────────────────────

class FilePicker:
    def __init__(self):
        self.visible=False; self._files=[]; self._scroll=0; self._hover=-1
        self._ROWS=10; self._ROW_H=42; self._W,self._H=640,480
        self._x=(W-self._W)//2; self._y=(H-self._H)//2

    def _scan(self):
        exts=('.mp3','.wav','.flac','.ogg','.m4a')
        self._files=sorted(f for f in glob.glob(os.path.join(SRC_DIR,'*'))
                           if os.path.splitext(f)[1].lower() in exts)

    def open(self): self._scan(); self._scroll=0; self.visible=True
    def close(self): self.visible=False

    def handle_event(self, ev, engine):
        if not self.visible: return False
        if ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE: self.close(); return True
        if ev.type==pygame.MOUSEMOTION: self._hover=self._row_at(ev.pos)
        if ev.type==pygame.MOUSEBUTTONDOWN:
            if ev.button==4: self._scroll=max(0,self._scroll-1)
            elif ev.button==5: self._scroll=min(max(0,len(self._files)-self._ROWS),self._scroll+1)
            else:
                row=self._row_at(ev.pos)
                if row>=0:
                    path=self._files[self._scroll+row]
                    threading.Thread(target=lambda:engine.load(path),daemon=True).start()
                    self.close()
                else: self.close()
        return True

    def _row_at(self,pos):
        mx,my=pos
        if not(self._x<mx<self._x+self._W): return -1
        ry=my-self._y-56; row=ry//self._ROW_H
        visible=self._files[self._scroll:self._scroll+self._ROWS]
        return row if 0<=row<len(visible) else -1

    def draw(self,surf,font_m,font_s):
        if not self.visible: return
        bx,by=self._x,self._y; bw,bh=self._W,self._H
        s=pygame.Surface((bw,bh),pygame.SRCALPHA); s.fill((*P0,240))
        pygame.draw.rect(s,P2,(0,0,bw,bh),2,border_radius=10); surf.blit(s,(bx,by))
        surf.blit(font_m.render('SELECT TRACK',True,P3),(bx+20,by+16))
        for i,path in enumerate(self._files[self._scroll:self._scroll+self._ROWS]):
            ry=by+56+i*self._ROW_H
            col=WHT if i==self._hover else TXT
            if i==self._hover: pygame.draw.rect(surf,P1,(bx+4,ry,bw-8,self._ROW_H-4),border_radius=4)
            surf.blit(font_s.render(os.path.splitext(os.path.basename(path))[0][:52],True,col),(bx+20,ry+10))
        if not self._files:
            surf.blit(font_s.render(f'No tracks in {SRC_DIR}',True,RED),(bx+20,by+56))


# ─────────────────────────────────────────────────────────────────────────────
# Config Panel
# ─────────────────────────────────────────────────────────────────────────────

class ConfigPanel:
    ROW_H  = 40
    ROWS   = 10   # visible rows at a time

    def __init__(self, cfg):
        self.visible = False
        self.cfg     = cfg
        self._sel    = 0
        self._scroll = 0          # first visible gesture row
        self._W = 860; self._H = 520
        self._x = (W - self._W)//2; self._y = (H - self._H)//2

    def _n(self): return max(1, len(EFFECT_FILES))

    def open(self):  self.visible = True
    def close(self): save_config(self.cfg); self.visible = False

    def _randomize(self):
        idxs = list(range(len(EFFECT_FILES)))
        random.shuffle(idxs)
        for i, slot in enumerate(GESTURE_SLOTS):
            self.cfg[slot] = idxs[i % len(idxs)]

    # returns rect for a button label/position
    def _btn_rects(self, bx, by, bw, bh):
        btn_y = by + bh - 48
        rnd   = pygame.Rect(bx + bw//2 - 180, btn_y, 160, 34)
        save  = pygame.Rect(bx + bw//2 + 20,  btn_y, 160, 34)
        return rnd, save

    def handle_event(self, ev):
        if not self.visible: return False
        bx,by = self._x,self._y; bw,bh = self._W,self._H
        rnd_r, save_r = self._btn_rects(bx,by,bw,bh)

        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_ESCAPE, pygame.K_c): self.close()
            elif ev.key == pygame.K_UP:
                self._sel = (self._sel-1) % len(GESTURE_SLOTS)
                self._scroll = min(self._scroll, self._sel)
                if self._sel < self._scroll: self._scroll = self._sel
            elif ev.key == pygame.K_DOWN:
                self._sel = (self._sel+1) % len(GESTURE_SLOTS)
                if self._sel >= self._scroll + self.ROWS:
                    self._scroll = self._sel - self.ROWS + 1
            elif ev.key == pygame.K_LEFT:
                s = GESTURE_SLOTS[self._sel]
                self.cfg[s] = (self.cfg[s]-1) % self._n()
            elif ev.key == pygame.K_RIGHT:
                s = GESTURE_SLOTS[self._sel]
                self.cfg[s] = (self.cfg[s]+1) % self._n()
            elif ev.key == pygame.K_r: self._randomize()
            elif ev.key == pygame.K_s: save_config(self.cfg)

        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx,my = ev.pos
            if ev.button in (4,5):   # scroll wheel
                self._scroll = max(0, min(len(GESTURE_SLOTS)-self.ROWS,
                                          self._scroll + (-1 if ev.button==4 else 1)))
                return True
            if ev.button != 1: return True
            if rnd_r.collidepoint(mx,my):  self._randomize(); return True
            if save_r.collidepoint(mx,my): save_config(self.cfg); return True
            if not(bx<mx<bx+bw and by<my<by+bh): self.close(); return True
            for vi in range(self.ROWS):
                gi = self._scroll + vi
                if gi >= len(GESTURE_SLOTS): break
                slot = GESTURE_SLOTS[gi]
                ry   = by + 76 + vi*self.ROW_H
                if not(ry < my < ry+self.ROW_H): continue
                self._sel = gi
                # < > arrows
                arr_l = pygame.Rect(bx+bw-150, ry+4, 28, 28)
                arr_r = pygame.Rect(bx+bw-46,  ry+4, 28, 28)
                if arr_l.collidepoint(mx,my):
                    self.cfg[slot] = (self.cfg[slot]-1) % self._n()
                elif arr_r.collidepoint(mx,my):
                    self.cfg[slot] = (self.cfg[slot]+1) % self._n()
        return True

    def draw(self, surf, font_m, font_s):
        if not self.visible: return
        bx,by = self._x,self._y; bw,bh = self._W,self._H
        s = pygame.Surface((bw,bh), pygame.SRCALPHA); s.fill((*P0,248))
        pygame.draw.rect(s, P3, (0,0,bw,bh), 2, border_radius=10)
        surf.blit(s, (bx,by))

        # title
        surf.blit(font_m.render('GESTURE  SOUND  CONFIG', True, P3), (bx+16, by+14))
        surf.blit(font_s.render('[UP/DN] select   [LT/RT] change sound   [R] randomize   [S] save   [ESC] close',
                                True, TXTSUB), (bx+16, by+42))

        # column headers
        surf.blit(font_s.render('GESTURE',   True, P2), (bx+20,  by+62))
        surf.blit(font_s.render('SOUND',     True, P2), (bx+280, by+62))
        surf.blit(font_s.render('#',         True, P2), (bx+bw-220, by+62))

        total = len(GESTURE_SLOTS)
        for vi in range(self.ROWS):
            gi = self._scroll + vi
            if gi >= total: break
            slot   = GESTURE_SLOTS[gi]
            ry     = by + 76 + vi*self.ROW_H
            active = gi == self._sel

            if active:
                pygame.draw.rect(surf, (*P2,50), (bx+4, ry, bw-8, self.ROW_H-2), border_radius=5)
            col = WHT if active else TXT

            # gesture label
            surf.blit(font_s.render(GESTURE_LABELS[slot], True, col), (bx+20, ry+10))

            # current sound name + arrows
            idx  = self.cfg.get(slot, 0) % self._n()
            name = (EFFECT_NAMES[idx] if EFFECT_NAMES else '?')[:34]
            # left arrow
            pygame.draw.polygon(surf, P2 if active else P1,
                                [(bx+bw-152, ry+18),(bx+bw-134, ry+8),(bx+bw-134, ry+28)])
            surf.blit(font_s.render(name, True, PA4 if active else TXT), (bx+280, ry+10))
            # index counter
            surf.blit(font_s.render(f'{idx+1}/{self._n()}', True, TXTSUB), (bx+bw-128, ry+10))
            # right arrow
            pygame.draw.polygon(surf, P2 if active else P1,
                                [(bx+bw-44, ry+18),(bx+bw-62, ry+8),(bx+bw-62, ry+28)])

        # scrollbar
        if total > self.ROWS:
            sb_h = bh - 120
            th   = max(20, sb_h * self.ROWS // total)
            ty   = int((self._scroll / (total-self.ROWS)) * (sb_h-th))
            pygame.draw.rect(surf, P1, (bx+bw-10, by+76, 6, sb_h), border_radius=3)
            pygame.draw.rect(surf, P2, (bx+bw-10, by+76+ty, 6, th), border_radius=3)

        # Randomize / Save buttons
        rnd_r, save_r = self._btn_rects(bx,by,bw,bh)
        for rect, label, active in [(rnd_r,'RANDOMIZE',False),(save_r,'SAVE',True)]:
            pygame.draw.rect(surf, lerp_col(P1,P2,0.4) if active else P1, rect, border_radius=6)
            pygame.draw.rect(surf, P3 if active else BORDER, rect, 2, border_radius=6)
            lbl = font_s.render(label, True, PA4 if active else TXT)
            surf.blit(lbl, (rect.centerx-lbl.get_width()//2, rect.centery-lbl.get_height()//2))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _font(size, bold=True):
    for name in ['Courier','CourierNew','DejaVuSansMono','monospace']:
        try: return pygame.font.SysFont(name,size,bold=bold)
        except: pass
    return pygame.font.Font(None,size)

def draw_hand(surf, lm, color):
    def px(x,y): return int(x*W),int(y*H)
    for a,b in HAND_CONNECTIONS:
        pygame.draw.line(surf,(*color,140),px(*lm[a][:2]),px(*lm[b][:2]),2)
    for i,(x,y,*_) in enumerate(lm):
        pxp=px(x,y); r=6 if i in(4,8,12,16,20) else 4
        pygame.draw.circle(surf,color,pxp,r)
        pygame.draw.circle(surf,WHT,pxp,r,1)

def draw_progress(surf,font,x,y,w,h,prog,title,pos_s,dur_s):
    pygame.draw.rect(surf,P0,(x,y,w,h),border_radius=4)
    if prog>0: pygame.draw.rect(surf,P2,(x,y,int(w*prog),h),border_radius=4)
    pygame.draw.rect(surf,BORDER,(x,y,w,h),1,border_radius=4)
    surf.blit(font.render(title[:60] if title else 'No track loaded',True,TXT),(x,y-22))
    t_lbl=font.render(f'{int(pos_s//60)}:{int(pos_s%60):02d}  /  {int(dur_s//60)}:{int(dur_s%60):02d}',True,TXTSUB)
    surf.blit(t_lbl,(x+w-t_lbl.get_width(),y-22))

def draw_effect_flash(surf,font_l,name,alpha):
    if alpha<=0: return
    lbl=font_l.render(name,True,(*PA4,int(clamp(alpha)*220)))
    surf.blit(lbl,(W//2-lbl.get_width()//2,H//2-lbl.get_height()//2-70))


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

class MusicGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('MUSIC GAME')
        self._screen=pygame.display.set_mode((W,H),pygame.RESIZABLE)
        self._clock=pygame.time.Clock(); self._fs=False

        self._font_s  = _font(15)
        self._font_m  = _font(18)
        self._font_l  = _font(44)
        self._font_xl = _font(62)

        # set window / dock icon
        if os.path.exists(LOGO_PATH):
            icon = pygame.image.load(LOGO_PATH).convert_alpha()
            icon = pygame.transform.smoothscale(icon, (64, 64))
            pygame.display.set_icon(icon)

        self._engine       = AudioEngine()
        self._tracker      = HandTracker()
        self._gesture      = GestureDetector()
        self._picker       = FilePicker()
        self._cfg          = load_config()
        self._config_panel = ConfigPanel(self._cfg)
        self._eff_buttons  = EffectButtons(self._cfg)
        self._vfx          = VFXEngine()
        self._ctrl         = ControlBar()
        self._mode_bar     = ModeBar(self._vfx)

        self._flash_name  = ''; self._flash_alpha=0.0
        self._hand_flash  = {'Left':0.0,'Right':0.0}
        self._show_hints  = True; self._running=True
        self._zen         = False   # [Z] blank canvas mode
        # pre-allocated reusable surfaces to avoid per-frame allocation
        self._dark_surf  = pygame.Surface((W, H), pygame.SRCALPHA)
        self._hand_surf  = pygame.Surface((W, H), pygame.SRCALPHA)

        # smoothed shader channel offsets (float pixels, lerped each frame)
        self._sdx_g = self._sdy_g = 0.0
        self._sdx_b = self._sdy_b = 0.0

    def _trigger(self, slot, lm=None):
        eff_idx=self._cfg.get(slot,0)
        if self._ctrl.sound_on:
            self._engine.play_sfx(eff_idx)
        if EFFECT_NAMES:
            self._flash_name=EFFECT_NAMES[eff_idx%len(EFFECT_NAMES)]
            self._flash_alpha=1.0
        self._eff_buttons.flash(slot)
        hand='Right' if 'right' in slot else 'Left'
        self._hand_flash[hand]=1.0
        if self._ctrl.vfx_on:
            self._vfx.on_gesture(slot, lm)

    def run(self):
        self._engine.start(); self._tracker.start()
        while self._running:
            dt=self._clock.tick(FPS)/1000.0
            self._dt=dt
            self._events(); self._update(dt); self._render()
        self._tracker.stop(); self._engine.stop(); pygame.quit()

    def _events(self):
        global _last_hands
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: self._running=False; return
            if self._config_panel.handle_event(ev): continue
            if self._picker.handle_event(ev,self._engine): continue
            if ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE: self._running=False
                elif ev.key==pygame.K_SPACE: self._engine.toggle_play()
                elif ev.key==pygame.K_l: self._picker.open()
                elif ev.key==pygame.K_c: self._config_panel.open()
                elif ev.key==pygame.K_f: self._toggle_fs()
                elif ev.key==pygame.K_h: self._show_hints=not self._show_hints
                elif ev.key==pygame.K_z: self._zen=not self._zen
                elif ev.key in KEY_SLOTS: self._trigger(KEY_SLOTS[ev.key])
            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                pos=ev.pos
                # control bar
                id_=self._ctrl.hit(pos)
                if id_=='sound': self._ctrl.sound_on=not self._ctrl.sound_on; continue
                if id_=='vfx':   self._ctrl.vfx_on=not self._ctrl.vfx_on; continue
                # mode bar (only visible when vfx on)
                if self._ctrl.vfx_on:
                    mid=self._mode_bar.hit(pos)
                    if mid=='prev': self._vfx.prev_mode(); continue
                    if mid=='next': self._vfx.next_mode(); continue
                    if mid and mid.startswith('mode_'):
                        self._vfx.mode=int(mid.split('_')[1]); self._vfx._clear(); continue
                # effect pads
                slot=self._eff_buttons.hit_test(pos)
                if slot: self._trigger(slot); continue
                # progress seek
                px,py=pos; bar_y=H-60; bx,bw=20,W-40
                if bar_y-10<py<bar_y+20 and bx<px<bx+bw:
                    self._engine.seek((px-bx)/bw)

    def _toggle_fs(self):
        self._fs=not self._fs
        self._screen=pygame.display.set_mode((W,H),pygame.FULLSCREEN if self._fs else pygame.RESIZABLE)

    def _update(self, dt):
        global _last_hands
        hands,_=self._tracker.get()
        if hands is None: hands={}
        _last_hands={'Left':hands.get('Left'),'Right':hands.get('Right')}

        self._gesture.update(hands)
        for slot in self._gesture.fired:
            hand='Right' if 'right' in slot else 'Left'
            self._trigger(slot, hands.get(hand))

        self._flash_alpha=max(0.0,self._flash_alpha-dt*3.0)
        for h in('Left','Right'): self._hand_flash[h]=max(0.0,self._hand_flash[h]-dt*4.0)
        self._eff_buttons.update(dt)
        if self._ctrl.vfx_on: self._vfx.update(dt,hands)

    def _apply_shader(self, frame, hands, dt):
        """Chromatic aberration: left hand shifts G channel, right hand shifts B channel.
        Offsets ease toward target (or zero when hand absent) so motion is smooth."""
        h, w = frame.shape[:2]
        b_ch = frame[:,:,0]
        g_ch = frame[:,:,1]
        r_ch = frame[:,:,2]

        lm_l = hands.get('Left')
        lm_r = hands.get('Right')

        # target offsets (zero when hand not visible)
        tgt_dx_g = ((lm_l[9][0] - 0.5) * 160) if lm_l else 0.0
        tgt_dy_g = ((lm_l[9][1] - 0.5) * 100) if lm_l else 0.0
        tgt_dx_b = ((lm_r[9][0] - 0.5) * 160) if lm_r else 0.0
        tgt_dy_b = ((lm_r[9][1] - 0.5) * 100) if lm_r else 0.0

        # exponential ease: speed=4 → ~98% of the way in ~1 s
        speed = 4.0
        k = 1.0 - math.exp(-speed * dt)
        self._sdx_g += (tgt_dx_g - self._sdx_g) * k
        self._sdy_g += (tgt_dy_g - self._sdy_g) * k
        self._sdx_b += (tgt_dx_b - self._sdx_b) * k
        self._sdy_b += (tgt_dy_b - self._sdy_b) * k

        def shift(ch, dx, dy):
            idx, idy = int(dx), int(dy)
            if idx == 0 and idy == 0:
                return ch
            M = np.float32([[1, 0, idx], [0, 1, idy]])
            return cv2.warpAffine(ch, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        g_shifted = shift(g_ch, self._sdx_g, self._sdy_g)
        b_shifted = shift(b_ch, self._sdx_b, self._sdy_b)
        return np.stack([b_shifted, g_shifted, r_ch], axis=2)

    def _render(self):
        global _last_hands
        screen  = self._screen
        vfx_on  = self._ctrl.vfx_on
        mode_name = VFX_NAMES[self._vfx.mode] if vfx_on else ''
        shader_mode = vfx_on and mode_name == 'Shader'

        hands, frame = self._tracker.get()
        if hands is None: hands = {}

        # ── background ──
        if shader_mode and frame is not None and CV_OK:
            # full-brightness camera with chromatic aberration from hand positions
            processed = self._apply_shader(frame, hands, self._dt)
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            s   = pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))
            screen.blit(pygame.transform.scale(s, (W, H)), (0,0))
        elif frame is not None and CV_OK:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            s   = pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))
            s   = pygame.transform.scale(s, (W, H))
            dim_alpha = 220 if vfx_on else 80
            self._dark_surf.fill((0, 0, 0, dim_alpha))
            s.blit(self._dark_surf, (0,0))
            screen.blit(s, (0,0))
        else:
            screen.fill(P0)

        # ── VFX layer (non-shader modes) ──
        if vfx_on and not shader_mode:
            self._vfx.draw(screen)

        # ── hand overlay ──
        self._hand_surf.fill((0,0,0,0))
        for label, lm in hands.items():
            base  = CR if label=='Right' else CL
            flash = self._hand_flash.get(label, 0.0)
            col   = tuple(int(lerp(base[i], 255, flash)) for i in range(3))
            cx, cy = int(lm[9][0]*W), int(lm[9][1]*H)
            if vfx_on:
                # in Shader mode: crosshair shows hand position + current offsets
                if shader_mode:
                    size = 14
                    pygame.draw.line(self._hand_surf, (*col,180), (cx-size,cy), (cx+size,cy), 2)
                    pygame.draw.line(self._hand_surf, (*col,180), (cx,cy-size), (cx,cy+size), 2)
                    pygame.draw.circle(self._hand_surf, (*col,100), (cx,cy), size+4, 1)
                else:
                    pygame.draw.circle(self._hand_surf, (*col,220), (cx,cy), 8)
                    pygame.draw.circle(self._hand_surf, (*col, 60), (cx,cy), 20)
            else:
                draw_hand(self._hand_surf, lm, col)
        screen.blit(self._hand_surf, (0,0))

        # ── shader mode: parameter readout ──
        if shader_mode and not self._zen:
            lm_l = hands.get('Left'); lm_r = hands.get('Right')
            lines = []
            if lm_l:
                dx = int((lm_l[9][0]-0.5)*160); dy = int((lm_l[9][1]-0.5)*100)
                lines.append((f'G  dx={dx:+d}  dy={dy:+d}', CL))
            else:
                lines.append(('G  (no left hand)', TXTSUB))
            if lm_r:
                dx = int((lm_r[9][0]-0.5)*160); dy = int((lm_r[9][1]-0.5)*100)
                lines.append((f'B  dx={dx:+d}  dy={dy:+d}', CR))
            else:
                lines.append(('B  (no right hand)', TXTSUB))
            for i,(txt,col) in enumerate(lines):
                screen.blit(self._font_s.render(txt, True, col), (12, H-90+i*20))

        if not self._zen:
            # ── effect pads and music UI (only when sound is on) ──
            if self._ctrl.sound_on:
                self._eff_buttons.draw(screen, self._font_s)
                draw_progress(screen, self._font_s, 20, H-60, W-40, 14,
                              self._engine.progress(), self._engine.title,
                              self._engine.position_s(), self._engine.duration_s())
                if self._flash_alpha > 0.01:
                    draw_effect_flash(screen, self._font_xl, self._flash_name, self._flash_alpha)
                state = 'PLAYING' if self._engine.playing else 'PAUSED'
                sl = self._font_m.render(state, True, GRN if self._engine.playing else PA4)
                screen.blit(sl, (W//2 - sl.get_width()//2, H-92))

            # ── hint overlay ──
            if self._show_hints and self._ctrl.sound_on:
                for i, slot in enumerate(GESTURE_SLOTS):
                    g   = GESTURE_EMOJI[slot]
                    e   = EFFECT_NAMES[self._cfg.get(slot,0)%max(1,len(EFFECT_NAMES))] if EFFECT_NAMES else '?'
                    col = CR if 'right' in slot else (CL if 'left' in slot else CEFF)
                    y   = 10 + i*22
                    screen.blit(self._font_s.render(g,           True, col), (10, y))
                    screen.blit(self._font_s.render(f'-> {e}',   True, TXT), (80, y))

            # ── control bar (top right) ──
            self._ctrl.draw(screen, self._font_s)
            if vfx_on:
                self._mode_bar.draw(screen, self._font_s)

            # ── warnings ──
            if not AUDIO_OK or not SD_OK:
                screen.blit(self._font_s.render('! pip install miniaudio sounddevice', True, RED), (10, H-112))
            if not CV_OK:
                screen.blit(self._font_s.render('! pip install opencv-python mediapipe', True, RED), (10, H-132))

            self._picker.draw(screen,self._font_m,self._font_s)
            self._config_panel.draw(screen,self._font_m,self._font_s)

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    _init_effects()
    game=MusicGame()
    game.run()

if __name__=='__main__':
    main()
