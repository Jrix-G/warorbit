"""Auto-extracted from a Kaggle notebook."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kaggle_environments import make
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import os, random, warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'✅ KRONOS OMEGA | Device: {DEVICE.upper()}')

# ── اكتشاف البيئة بالتجربة المباشرة ──────────────────────
CANDIDATES = [
    'orbit_wars', 'orbitwars', 'planet_wars',
    'lux_ai_s2',  'kore_2022', 'halite',
    'hungry_geese', 'connectx', 'tictactoe'
]

ENV_NAME = None
for candidate in CANDIDATES:
    try:
        _t = make(candidate, debug=False)
        ENV_NAME = candidate
        print(f'🎯 ENV_NAME = "{ENV_NAME}"')
        break
    except:
        print(f'   ✗ {candidate}')

if ENV_NAME is None:
    raise ValueError('❌ لم يتم العثور على أي بيئة — أضف اسم البيئة يدوياً في السطر التالي')
    # ENV_NAME = 'اكتب_الاسم_هنا'

# ── استكشاف شكل الـ observation ──────────────────────────
_test   = make(ENV_NAME, debug=False)
_states = _test.reset(num_agents=4)
print(f'📊 Observation keys : {list(_states[0].observation.keys())}')
print(f'📊 Status           : {_states[0].status}')
print(f'✅ CELL 1 DONE')

N_PLANETS = 20
OBS_DIM   = 160  # 120 planet + 20 fleet + 20 global

def build_quantum_obs(obs_raw, player_id):
    planets   = obs_raw.get('planets', [])
    fleets    = obs_raw.get('fleets',  [])
    step      = obs_raw.get('step',    0)
    n_planets = len(planets)
    obs_vec   = []

    # ── PART 1: كل كوكب (6 features × 20 = 120) ──────────
    for i in range(N_PLANETS):
        if i < n_planets:
            p = planets[i]
            owner, ships, growth = p[1], p[5], p[6]
            is_mine    = 1.0 if owner == player_id else 0.0
            is_enemy   = 1.0 if (owner != player_id and owner != -1) else 0.0
            is_neutral = 1.0 if owner == -1 else 0.0
            ships_norm  = min(1.0, ships  / 300.0)
            growth_norm = min(1.0, growth / 10.0)
            # تهديد: أقرب عدو
            enemy_dists = []
            for j in range(n_planets):
                if j != i and planets[j][1] != player_id and planets[j][1] != -1:
                    dx = planets[i][3] - planets[j][3]
                    dy = planets[i][4] - planets[j][4]
                    enemy_dists.append((dx**2 + dy**2)**0.5)
            threat = min(enemy_dists) / 50.0 if enemy_dists else 1.0
            obs_vec.extend([is_mine, is_enemy, is_neutral,
                            ships_norm, growth_norm, min(1.0, threat)])
        else:
            obs_vec.extend([0.0] * 6)

    # ── PART 2: أساطيلي الأربع الأكبر (5 × 4 = 20) ───────
    my_fleets = [f for f in fleets if f[0] == player_id][:4]
    for fi in range(4):
        if fi < len(my_fleets):
            f = my_fleets[fi]
            obs_vec.extend([
                f[1] / N_PLANETS,
                f[2] / N_PLANETS,
                min(1, f[3] / 200.0),
                min(1, f[4] / 50.0),
                1.0
            ])
        else:
            obs_vec.extend([0.0] * 5)

    # ── PART 3: إحصاءات عالمية (20) ──────────────────────
    my_ships  = sum(p[5] for p in planets if p[1] == player_id)
    my_cnt    = sum(1    for p in planets if p[1] == player_id)
    my_growth = sum(p[6] for p in planets if p[1] == player_id)
    en_ships  = sum(p[5] for p in planets if p[1] != player_id and p[1] != -1)
    total     = sum(p[5] for p in planets) + 1
    obs_vec.extend([
        min(1, my_ships  / 500.0),
        min(1, my_cnt    / N_PLANETS),
        min(1, my_growth / 30.0),
        min(1, en_ships  / 500.0),
        min(1, step      / 400.0),
        min(1, my_ships  / total),           # نسبة الهيمنة
        float(my_cnt > len(planets) / 2),    # هل أسيطر على الأغلبية؟
        min(1, len(my_fleets) / 10.0),       # نشاط الأساطيل
    ])

    # ── Pad & Clip ─────────────────────────────────────────
    while len(obs_vec) < OBS_DIM:
        obs_vec.append(0.0)

    return np.clip(np.array(obs_vec[:OBS_DIM], dtype=np.float32), -1, 1)

print('✅ CELL 2: Quantum Observation (160 dims) — READY')

def greedy_oracle(obs_raw, player_id):
    """
    أفضل هجوم بناءً على نسبة: (نمو²) / (مسافة × دفاع)
    يُستخدم كـ safety net لأي حركة غير صالحة
    """
    planets   = obs_raw.get('planets', [])
    my_planets = [(i, p) for i, p in enumerate(planets)
                  if p[1] == player_id and p[5] > 10]
    if not my_planets:
        return None

    best_score  = -1
    best_action = None

    for src_idx, src_p in my_planets:
        for tgt_idx, tgt_p in enumerate(planets):
            if tgt_idx == src_idx:
                continue
            if tgt_p[1] == player_id:   # لا تهاجم نفسك إلا للتعزيز
                continue
            dx   = src_p[3] - tgt_p[3]
            dy   = src_p[4] - tgt_p[4]
            dist = max(1.0, (dx**2 + dy**2)**0.5)
            score = (tgt_p[6] + 1)**2 / (dist * (tgt_p[5] + 1))

            if score > best_score:
                best_score  = score
                frac        = min(9, max(1, int(src_p[5] * 0.6 / 20)))
                best_action = [src_idx, tgt_idx, frac]

    return best_action

print('✅ CELL 3: Greedy Oracle — ARMED')

# ═══════════════════════════════════════════════════════════
# CELL 4 FIXED — KronosOmegaEnv يستخدم ENV_NAME المكتشف
# ═══════════════════════════════════════════════════════════

N_PLANETS = 20
OBS_DIM   = 160

class KronosOmegaEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, player_id=0, opponent_pool=None):
        super().__init__()

        # ✅ FIX: استخدم الاسم المكتشف في Cell 1
        self.env           = make(ENV_NAME, debug=False)
        self.player_id     = player_id
        self.opponent_pool = opponent_pool or []
        self.obs_raw       = {}

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([N_PLANETS, N_PLANETS, 10])
        self._prev_my_ships  = 0
        self._prev_my_growth = 0

    # ─── Safe Observation Parser ───────────────────────────
    def _parse(self, obs_raw):
        """يتعامل مع أي شكل للـ observation"""
        planets = (obs_raw.get('planets') or
                   obs_raw.get('cells')   or
                   obs_raw.get('boards')  or [])
        fleets  = (obs_raw.get('fleets')  or
                   obs_raw.get('ships')   or
                   obs_raw.get('units')   or [])
        step    =  obs_raw.get('step', 0)
        return planets, fleets, int(step)

    # ─── Action Masking ────────────────────────────────────
    def action_masks(self):
        planets, _, _ = self._parse(self.obs_raw)
        src_mask = np.zeros(N_PLANETS, dtype=bool)
        for i, p in enumerate(planets[:N_PLANETS]):
            try:
                if p[1] == self.player_id and p[5] > 5:
                    src_mask[i] = True
            except (IndexError, TypeError):
                pass
        if not src_mask.any():
            src_mask[:] = True
        return np.concatenate([
            src_mask,
            np.ones(N_PLANETS, dtype=bool),
            np.ones(10,        dtype=bool)
        ])

    # ─── Reward Shaping ────────────────────────────────────
    def _shaped_reward(self, base_reward):
        planets, _, _ = self._parse(self.obs_raw)
        my_ships   = sum(p[5] for p in planets if p[1] == self.player_id)
        my_cnt     = sum(1    for p in planets if p[1] == self.player_id)
        my_growth  = sum(p[6] for p in planets if p[1] == self.player_id)

        r  = base_reward * 10.0
        r += (my_cnt    / max(1, len(planets))) * 0.5
        r += (my_growth - self._prev_my_growth) * 0.3
        r += (my_ships  - self._prev_my_ships)  * 0.001
        r -= 0.001

        self._prev_my_ships  = my_ships
        self._prev_my_growth = my_growth
        return float(r)

    # ─── Gym Interface ─────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        states           = self.env.reset(num_agents=4)
        self.obs_raw     = states[0].observation
        self._prev_my_ships  = 0
        self._prev_my_growth = 0
        return build_quantum_obs(self.obs_raw, self.player_id), {}

    def step(self, action):
        planets, _, _ = self._parse(self.obs_raw)
        src, tgt, frac = int(action[0]), int(action[1]), int(action[2])

        valid = (
            src < len(planets) and tgt < len(planets)
            and src != tgt
            and len(planets[src]) > 1
            and planets[src][1] == self.player_id
        )
        my_action    = [src, tgt, frac] if valid \
                       else greedy_oracle(self.obs_raw, self.player_id)
        full_actions = [None] * 4
        full_actions[self.player_id] = my_action

        states       = self.env.step(full_actions)
        self.obs_raw = states[0].observation
        base_reward  = states[self.player_id].reward or 0.0
        reward       = self._shaped_reward(base_reward)
        done         = states[0].status == 'DONE'
        obs          = build_quantum_obs(self.obs_raw, self.player_id)
        return obs, reward, done, False, {}


# ── Sanity check ───────────────────────────────────────────
print(f'🔧 Testing with ENV_NAME = "{ENV_NAME}"...')
test_env = KronosOmegaEnv()
obs, _   = test_env.reset()
mask     = test_env.action_masks()
print(f'✅ CELL 4 FIXED')
print(f'   obs shape    : {obs.shape}')
print(f'   action space : {test_env.action_space}')
print(f'   valid src    : {mask[:N_PLANETS].sum()} / {N_PLANETS}')

class KronosFeatureExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=512):
        super().__init__()
        self._features_dim = features_dim
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 512), nn.ReLU(),
            nn.Linear(512, 512),     nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU(),
        )
    @property
    def features_dim(self):
        return self._features_dim
    def forward(self, obs):
        return self.net(obs)

print('✅ CELL 5 OK')

class SelfPlayLeagueCallback(BaseCallback):
    def __init__(self, snapshot_freq=50_000, max_pool_size=5, verbose=1):
        super().__init__(verbose)
        self.snapshot_freq = snapshot_freq
        self.max_pool_size = max_pool_size
        self.pool          = []
        self.snapshot_count = 0

    def _on_step(self):
        if self.n_calls % self.snapshot_freq == 0 and self.n_calls > 0:
            path = f'/kaggle/working/snap_{self.snapshot_count}.zip'
            self.model.save(path)
            try:
                snap = MaskablePPO.load(path, device='cpu')
                self.pool.append(snap)
                if len(self.pool) > self.max_pool_size:
                    self.pool.pop(0)
                self.snapshot_count += 1
                print(f'\n🧬 Snapshot {self.snapshot_count} | Pool: {len(self.pool)}')
            except Exception as e:
                print(f'⚠️ Snapshot warning: {e}')
        return True


class TrainingProgressCallback(BaseCallback):
    def __init__(self, log_freq=10_000):
        super().__init__()
        self.log_freq   = log_freq
        self.ep_rewards = []

    def _on_step(self):
        info = self.locals.get('infos', [{}])[0]
        if 'episode' in info:
            self.ep_rewards.append(info['episode']['r'])
        if self.n_calls % self.log_freq == 0 and self.ep_rewards:
            avg = np.mean(self.ep_rewards[-20:])
            print(f'  Step {self.n_calls:>8,} | Avg Reward (20ep): {avg:+.3f}')
        return True

print('✅ CELL 6: Self-Play League Callbacks — READY')

# ═══════════════════════════════════════════════════════════
# CELL 6.5 — Speed Test قبل التدريب الكامل
# ═══════════════════════════════════════════════════════════
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

os.makedirs('/kaggle/working', exist_ok=True)

def make_env():
    env = KronosOmegaEnv(player_id=0)
    env = ActionMasker(env, lambda e: e.action_masks())
    env = Monitor(env)
    return env

train_env = DummyVecEnv([make_env])

_test_model = MaskablePPO(
    MaskableActorCriticPolicy,
    train_env,
    verbose=0,
    n_steps=512,
    batch_size=64,
    policy_kwargs=dict(
        features_extractor_class=KronosFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, 128],
    ),
    device='cpu',
)

print('⏱️  Speed test جاري — انتظر 30 ثانية...')
_start = time.time()
_test_model.learn(total_timesteps=5_000, progress_bar=False)
_elapsed = time.time() - _start

_speed   = 5_000 / _elapsed
_eta_1m  = 1_000_000 / _speed / 3600
_eta_500k = 500_000  / _speed / 3600

print(f'\n📊 النتيجة:')
print(f'   Speed    : {_speed:.0f} steps/sec')
print(f'   ETA 500K : {_eta_500k:.1f} ساعة')
print(f'   ETA 1M   : {_eta_1m:.1f}  ساعة')

if _speed > 300:
    print('\n✅ GPU سريع — شغّل 1,000,000 خطوة بأمان')
    TOTAL_STEPS = 1_000_000
elif _speed > 150:
    print('\n⚠️  متوسط — شغّل 500,000 خطوة')
    TOTAL_STEPS = 500_000
else:
    print('\n🐢 بطيء — شغّل 300,000 خطوة')
    TOTAL_STEPS = 300_000

print(f'\n🎯 TOTAL_STEPS = {TOTAL_STEPS:,}')

# ═══════════════════════════════════════════════════════════
# CELL 7 FINAL — التدريب مع Resume التلقائي
# ═══════════════════════════════════════════════════════════

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

os.makedirs('/kaggle/working', exist_ok=True)

selfplay_cb = SelfPlayLeagueCallback(
    snapshot_freq=20_000,
    max_pool_size=5
)
progress_cb   = TrainingProgressCallback(log_freq=5_000)
checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path='/kaggle/working/',
    name_prefix='kronos_omega'
)

def make_env():
    env = KronosOmegaEnv(player_id=0, opponent_pool=selfplay_cb.pool)
    env = ActionMasker(env, lambda e: e.action_masks())
    env = Monitor(env)
    return env

train_env = DummyVecEnv([make_env])

RESUME_PATH = '/kaggle/working/kronos_omega_FINAL'

# ✅ Resume تلقائي — في الـ run الثاني يكمل من حيث وقف
if os.path.exists(RESUME_PATH + '.zip'):
    print('📂 Resume من checkpoint موجود...')
    model = MaskablePPO.load(
        RESUME_PATH,
        env=train_env,
        device='cpu',
    )
    model.learning_rate = 1e-4  # أبطأ في المرحلة المتقدمة
    model.ent_coef      = 0.01  # استكشاف أقل
    print(f'   learning_rate : 1e-4')
    print(f'   ent_coef      : 0.01')

else:
    print('🆕 Training من الصفر...')
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.05,
        policy_kwargs=dict(
            features_extractor_class=KronosFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[256, 128],
        ),
        device='cpu',
    )

print(f'\n🚀 Training | Steps: {TOTAL_STEPS:,}')
print('═' * 50)

import time
_t = time.time()

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[selfplay_cb, progress_cb, checkpoint_cb],
    progress_bar=False,
    reset_num_timesteps=False,  # ✅ يكمل العداد من حيث وقف
)

_done = (time.time() - _t) / 60
model.save(RESUME_PATH)
print(f'\n🏆 DONE في {_done:.1f} دقيقة')
print(f'💾 Saved: {RESUME_PATH}')

SUBMISSION = open('/kaggle/working/kronos_omega_FINAL.zip','rb') if False else None

# كتابة ملف الإرسال المستقل
code = """
import numpy as np
_MODEL = None

def _obs(o, pid):
    # نسخة مضغوطة من build_quantum_obs
    p = o.get('planets',[])
    f = o.get('fleets',[])
    v = []
    for i in range(20):
        if i<len(p):
            pp=p[i]
            v.extend([float(pp[1]==pid),float(pp[1]!=pid and pp[1]!=-1),
                      float(pp[1]==-1),min(1,pp[5]/300.),min(1,pp[6]/10.),0.5])
        else: v.extend([0.]*6)
    mf=[ff for ff in f if ff[0]==pid][:4]
    for fi in range(4):
        if fi<len(mf): ff=mf[fi];v.extend([ff[1]/20,ff[2]/20,min(1,ff[3]/200.),min(1,ff[4]/50.),1.])
        else: v.extend([0.]*5)
    ms=sum(pp[5] for pp in p if pp[1]==pid)
    mc=sum(1 for pp in p if pp[1]==pid)
    v.extend([min(1,ms/500.),min(1,mc/20.),min(1,o.get('step',0)/400.),0.5,0.5,0.5,0.5,0.5])
    import numpy as np
    a=np.array(v[:160],dtype=np.float32)
    return np.pad(a,(0,max(0,160-len(a))))

def _greedy(o,pid):
    p=o.get('planets',[])
    mp=[(i,pp) for i,pp in enumerate(p) if pp[1]==pid and pp[5]>10]
    if not mp: return None
    b,bs=None,-1
    for si,sp in mp:
        for ti,tp in enumerate(p):
            if ti==si: continue
            if tp[1]==pid: continue
            dx=sp[3]-tp[3];dy=sp[4]-tp[4];d=max(1,(dx**2+dy**2)**.5)
            s=(tp[6]+1)**2/(d*(tp[5]+1))
            if s>bs: bs=s;b=[si,ti,min(9,max(1,int(sp[5]*.6/20)))]
    return b

def agent(obs,conf):
    global _MODEL
    if _MODEL is None:
        from sb3_contrib import MaskablePPO
        try: _MODEL=MaskablePPO.load('/kaggle/working/kronos_omega_FINAL',device='cpu')
        except: _MODEL='G'
    pid=obs.get('player',0);p=obs.get('planets',[])
    if _MODEL=='G': return _greedy(obs,pid)
    try:
        import numpy as np
        a,_=_MODEL.predict(_obs(obs,pid),deterministic=True)
        s,t,f=int(a[0]),int(a[1]),int(a[2])
        if s<len(p) and t<len(p) and s!=t and p[s][1]==pid and p[s][5]>5: return[s,t,f]
    except: pass
    return _greedy(obs,pid)
"""

with open('/kaggle/working/submission.py','w') as f:
    f.write(code)

print('✅ submission.py ready')
print('📦 Files:')
for fn in sorted(os.listdir('/kaggle/working')):
    sz = os.path.getsize(f'/kaggle/working/{fn}')
    print(f'   {fn:<45} {sz/1024:.1f} KB')
