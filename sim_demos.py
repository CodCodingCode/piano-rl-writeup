"""Purpose-built MuJoCo demos for the write-up.

Same piano + Shadow Hand model the policy trains in, driven by scripted control
that isolates one mechanism each. Press targets are taken from frames of the real
v15 rollout where exactly one key sounded and one fingertip was on it, converted
to actuator targets, so a demo press is a press the trained policy actually makes.

    .venv/bin/python sim_demos.py verify [name...]
    .venv/bin/python sim_demos.py probe  [name...]
    .venv/bin/python sim_demos.py render [name...]
"""
import sys, pathlib
import numpy as np
import mujoco

ROOT = pathlib.Path("/private/tmp/claude-501/-Users-owner-Downloads-coding-projects-piano-player/"
                    "52edffb4-0fe9-4804-ac12-6d23ee632aff")
OUT = ROOT / "scratchpad" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

from dexsim.tasks.piano_mj.piano_mj_env_cfg import PianoMjEnvCfg
from dexsim.tasks.piano_mj.vec_env import PianoMjVecEnv
from dexsim.tasks.piano_mj.piano_mj_env import KEY_SOUND_ANGLE

W, H = 900, 560
L, R = 0, 1
TH, FF, MF, RF, LF = 0, 1, 2, 3, 4
TAGS = ["TH", "FF", "MF", "RF", "LF"]
LIT = np.array([0.78, 0.40, 0.14, 1.0])

# (hand, finger, rollout frame) of a verified single-key press
PRESS = {
    (R, FF): 710,
    (R, LF): 628,
    (R, TH): 635,
    (R, MF): 549,
    (L, MF): 572,
}


def build():
    cfg = PianoMjEnvCfg()
    cfg.midi_path = "results/nettspend - we not like you (1).mid"
    cfg.fingering_method = "seq"
    cfg.random_song_start = False
    cfg.episode_length_s = 65
    cfg.__post_init__()
    env = PianoMjVecEnv(cfg, num_envs=1, threads=1).envs[0]
    env.reset()
    return env


class Rig:
    def __init__(self, env):
        self.env, self.m, self.d, self.cfg = env, env.model, env.data, env.cfg
        self.base_rgba = self.m.geom_rgba.copy()
        self.base_damping = self.m.dof_damping.copy()
        self.key_geom = np.array([
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"key_geom_{i}")
            for i in range(88)])
        self.Q = np.load(ROOT / "scratchpad" / "rollout_v15.npz")["qpos"]
        self.aname = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                      for i in range(self.m.nu)}
        self.reset()

    # --- state -----------------------------------------------------------
    def reset(self, rail=None):
        self.d.qpos[:] = self.env.ready_qpos
        if rail is not None:
            for h in range(2):
                self.d.qpos[self.env.rail_qadr[h]] = rail[h]
        self.d.qvel[:] = 0
        self.sounding = np.zeros(88, bool)
        self.ctrl0 = self.ctrl_at(self.d.qpos.copy())
        self.d.ctrl[:] = self.ctrl0
        self.m.geom_rgba[:] = self.base_rgba
        self.m.dof_damping[:] = self.base_damping
        mujoco.mj_forward(self.m, self.d)

    def ctrl_at(self, qpos):
        """Actuator targets that hold this pose (same trick the env uses)."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = qpos
        mujoco.mj_forward(self.m, d)
        return np.clip(d.actuator_length.copy(),
                       self.m.actuator_ctrlrange[:, 0],
                       self.m.actuator_ctrlrange[:, 1])

    # --- actuator groups -------------------------------------------------
    def finger_acts(self, hand, finger):
        pre = ("L_robot0_A_", "R_robot0_A_")[hand]
        return np.array([i for n, i in self.aname.items()
                         if n.startswith(pre + TAGS[finger])])

    def hand_finger_acts(self, hand):
        return np.concatenate([self.finger_acts(hand, f) for f in range(5)])

    def coupled_act(self, hand, finger):
        pre = ("L_robot0_A_", "R_robot0_A_")[hand]
        return self.aname[pre + TAGS[finger] + "J0"]

    def rail_act(self, hand):
        return self.env.rail_act[hand]

    # --- readings --------------------------------------------------------
    def frac(self):
        return np.clip(self.d.qpos[self.env.key_qadr] / KEY_SOUND_ANGLE, 0.0, 3.0)

    def light(self):
        """Paint the keys the env would call sounding: the same hysteretic latch,
        struck past key_struck_frac, released below key_release_frac."""
        f = self.frac()
        self.sounding = ((self.sounding | (f >= self.cfg.key_struck_frac))
                         & ~(f < self.cfg.key_release_frac))
        self.m.geom_rgba[self.key_geom] = self.base_rgba[self.key_geom]
        if self.sounding.any():
            self.m.geom_rgba[self.key_geom[self.sounding]] = LIT

    def tip(self, hand, finger):
        return self.d.site_xpos[self.env.tip_sites[hand][finger]].copy()

    def keypos(self, k):
        return self.d.site_xpos[self.env.key_site[k]].copy()

    # --- press setup -----------------------------------------------------
    def press_setup(self, hand, finger):
        """The real pressing pose: (press ctrl, key it sounds, that frame's qpos)."""
        q = self.Q[PRESS[(hand, finger)]].copy()
        pressed = np.clip(q[self.env.key_qadr] / KEY_SOUND_ANGLE, 0, 3)
        return self.ctrl_at(q), int(np.argmax(pressed)), q

    def start_from(self, qpos, ctrl, settle=120):
        """Park the model in a pose and let the actuators settle before filming."""
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0
        self.sounding = np.zeros(88, bool)
        self.d.ctrl[:] = ctrl
        self.m.geom_rgba[:] = self.base_rgba
        self.m.dof_damping[:] = self.base_damping
        mujoco.mj_forward(self.m, self.d)
        for _ in range(settle):
            mujoco.mj_step(self.m, self.d)

    def lift_of(self, ctrl, hand, fingers, amount):
        """Curl the given fingers' flexors away from the keys (the leap action)."""
        c = ctrl.copy()
        for f in fingers:
            cols = self.env._finger_flex_acts[hand][f]
            c[cols] = np.clip(ctrl[cols] + amount,
                              self.m.actuator_ctrlrange[cols, 0],
                              self.m.actuator_ctrlrange[cols, 1])
        return c


def cam(at, dist, az, el):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = at
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * float(np.clip(t, 0, 1)))


def ramp(t, up=0.30, hold=0.62):
    """0 -> 1 -> 0 over a normalised demo clock."""
    if t < up:
        return ease(t / up)
    if t < hold:
        return 1.0
    return 1 - ease((t - hold) / (1 - hold - 0.05))


# ============================== demos ================================
# Each is a generator of (camera) per frame; the driver steps physics.

def d_press(rig, hand=R, finger=FF, slow=1, damping=None, n=150, lift=0.75):
    """One finger comes down onto one key. Everything else holds its pose."""
    pc, key, q = rig.press_setup(hand, finger)
    up = rig.lift_of(pc, hand, [finger], lift)
    rig.start_from(q, up)
    acts = rig.finger_acts(hand, finger)
    if damping is not None:
        jid = int(np.nonzero(rig.m.jnt_qposadr == rig.env.key_qadr[key])[0][0])
        rig.m.dof_damping[int(rig.m.jnt_dofadr[jid])] = damping
    for i in range(n):
        t = i / n
        a = ramp(t, up=0.22, hold=0.46) if damping is not None else ramp(t)
        rig.d.ctrl[acts] = up[acts] + a * (pc[acts] - up[acts])
        for _ in range(max(1, 8 // slow)):
            mujoco.mj_step(rig.m, rig.d)
        rig.light()
        at = 0.35 * rig.tip(hand, finger) + 0.65 * rig.keypos(key)
        yield cam(at + np.array([-0.03, 0, 0.004]), 0.24, 130, -38)


def d_keyflick(rig, damping, key=44, n=240, impulse=-1.6):
    """Flick one key and let it go. The hand is out of the way: this is the
    key's own spring and damper, nothing else."""
    rig.reset()
    jid = int(np.nonzero(rig.m.jnt_qposadr == rig.env.key_qadr[key])[0][0])
    dof = int(rig.m.jnt_dofadr[jid])
    rig.m.dof_damping[dof] = damping
    for i in range(n):
        if i == 22:
            rig.d.qvel[dof] = impulse          # the strike
        mujoco.mj_step(rig.m, rig.d)            # one physics step per frame
        rig.light()
        at = rig.keypos(key)
        yield cam(at + np.array([-0.03, 0, 0.002]), 0.19, 112, -34)


def d_tendon(rig, hand=R, finger=FF, n=150):
    """Drive only the coupled actuator: the middle and tip joints bend as a pair."""
    pc, key, q = rig.press_setup(hand, finger)
    a0 = rig.coupled_act(hand, finger)
    up = pc.copy()
    up[a0] = np.clip(pc[a0] + 0.9, *rig.m.actuator_ctrlrange[a0])
    rig.start_from(q, up)
    for i in range(n):
        t = i / n
        a = ramp(t, up=0.42, hold=0.55)
        rig.d.ctrl[a0] = up[a0] + a * (pc[a0] - up[a0])
        for _ in range(8):
            mujoco.mj_step(rig.m, rig.d)
        rig.light()
        at = 0.65 * rig.tip(hand, finger) + 0.35 * rig.d.body(rig.env.palm_body[hand]).xpos
        yield cam(at + np.array([0, 0, 0.008]), 0.20, 150, -28)


def d_traverse(rig, lift, hand=R, n=180, amount=0.9):
    """Slide the hand along the keyboard with the fingers down, or lifted."""
    pc, key, q = rig.press_setup(hand, FF)
    # lifted = the ready pose's hover for this hand's fingers, which is what the
    # leap curriculum drives the policy toward while the rail is moving
    up = pc.copy()
    hf = rig.hand_finger_acts(hand)
    up[hf] = rig.ctrl_at(rig.env.ready_qpos)[hf]
    rig.start_from(q, pc if not lift else up)
    ia = rig.rail_act(hand)
    start = float(rig.d.qpos[rig.env.rail_qadr[hand]])
    lo, hi = rig.m.actuator_ctrlrange[ia]
    end = float(np.clip(start + 0.26, lo, hi))
    for i in range(n):
        t = i / n
        moving = 0.14 < t < 0.86
        rig.d.ctrl[:] = up if (lift and moving) else pc
        rig.d.ctrl[ia] = start + (end - start) * ease((t - 0.14) / 0.72)
        for _ in range(8):
            mujoco.mj_step(rig.m, rig.d)
        rig.light()
        at = rig.d.body(rig.env.palm_body[hand]).xpos.copy()
        yield cam(at + np.array([-0.09, 0, -0.03]), 0.40, 98, -22)


def d_reach(rig, finger, hand=R, n=150):
    """Same hand pose, same downward command, a different finger each time."""
    pc, key, q = rig.press_setup(hand, FF)
    up = rig.lift_of(pc, hand, [finger], 0.75)
    rig.start_from(q, up)
    acts = rig.finger_acts(hand, finger)
    # full travel in whichever direction takes this finger down onto the keys
    direction = np.sign(pc[acts] - up[acts])
    direction[direction == 0] = -1.0
    target = np.where(direction < 0,
                      rig.m.actuator_ctrlrange[acts, 0],
                      rig.m.actuator_ctrlrange[acts, 1])
    for i in range(n):
        t = i / n
        a = ramp(t, up=0.34, hold=0.72)
        rig.d.ctrl[acts] = up[acts] + a * (target - up[acts])
        for _ in range(8):
            mujoco.mj_step(rig.m, rig.d)
        rig.light()
        at = rig.tip(hand, finger) + np.array([-0.03, 0, -0.015])
        yield cam(at, 0.25, 130, -36)


DEMOS = {
    "press_one":   lambda r: d_press(r),
    "damper_on":   lambda r: d_keyflick(r, damping=0.1, n=150, impulse=-2.2),
    "damper_off":  lambda r: d_keyflick(r, damping=0.0, n=150, impulse=-2.2),
    "tendon":      lambda r: d_tendon(r),
    "drag_down":   lambda r: d_traverse(r, lift=False),
    "drag_lift":   lambda r: d_traverse(r, lift=True),
    "thumb_short": lambda r: d_reach(r, TH),
}


def main():
    mode = sys.argv[1]
    names = sys.argv[2:] or list(DEMOS)
    env = build()
    rig = Rig(env)
    rend = mujoco.Renderer(env.model, height=H, width=W) if mode != "verify" else None
    import imageio.v2 as imageio

    for name in names:
        frames, peak, nlit = [], 0.0, 0
        for c in DEMOS[name](rig):
            f = rig.frac()
            peak = max(peak, float(f.max()))
            nlit = max(nlit, int((f >= rig.cfg.key_struck_frac).sum()))
            if rend is not None:
                rend.update_scene(rig.d, camera=c)
                frames.append(rend.render())
        print(f"{name:12s} peak_depth {peak:4.2f}  keys_sounding_max {nlit}")
        if mode == "probe":
            for tag, idx in (("a", len(frames) // 3), ("b", int(len(frames) * .55))):
                imageio.imwrite(OUT / f"probe_{name}_{tag}.png", frames[idx])
        elif mode == "render":
            imageio.mimwrite(OUT / f"{name}.mp4", frames, fps=30, quality=8)


main()
