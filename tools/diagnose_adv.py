"""Diagnose WHY the learned adversary is flat: undertrained, or no gradient signal?

Distinguishes two hypotheses that need opposite fixes:
  H1 "undertrained"   -> the actor is still moving, just slowly. More episodes help.
  H2 "no signal"      -> the reward barely depends on the adversary's action, so the
                         critic is action-independent, the actor gets ~zero gradient,
                         and more episodes change nothing.
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/attack_framework")
sys.path.insert(0, "src/maddpg_clean")
sys.path.insert(0, "tools")

from learned_adversary import (AdversaryConfig, LearnedObservationAdversary,
                               AdversaryActor)
import train_adversary as ta
from standalone_experiment_runner import StandaloneExperimentRunner

CFG = "reward_fix_full_config.json"
VARIANT = "CC-Simple"
CKPT = "data/results/learned_adv/CC-Simple/adversary.pt"
OUT = "data/results/learned_adv/CC-Simple"
T = 256

runner = StandaloneExperimentRunner(config_path=CFG, results_dir=OUT)
maddpg, env, trainable, obs_dim = ta.build_victim_and_env(
    runner, VARIANT, "data/results/reward_fix/models")
hosts = env.engine.get_all_hosts()
n_total = getattr(env.engine, "n_total_hosts", len(hosts))
group_hosts = [hosts[i] for i in trainable]

meta = torch.load(CKPT, map_location="cpu")
cfg = AdversaryConfig(epsilon=0.30, coordinate=bool(meta.get("coordinate")))
adv = LearnedObservationAdversary(obs_dim, cfg, group_hosts=group_hosts).load(CKPT)
dev = adv.device

# reference: an UNTRAINED actor of the same shape
fresh = AdversaryActor(adv.actor_obs_dim, cfg.hidden).to(dev).eval()


def rollout(actor, label, seed=20240601):
    """Roll the victim under `actor`'s perturbation; collect raw tanh outputs."""
    import random as pyr
    pyr.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env.engine.reset_with_load(offered_load_factor=2.0)
    states = [env.engine.get_state(h) for h in hosts]
    deltas, losses = [], []
    for _ in range(T):
        joint = np.concatenate([states[i] for i in trainable]).astype(np.float32)
        with torch.no_grad():
            d = actor(torch.as_tensor(joint, device=dev).unsqueeze(0)).squeeze(0).cpu().numpy()
        deltas.append(d)
        applied = adv._project(joint, joint + d * cfg.epsilon)
        adv_states = list(states)
        for pos, i in enumerate(trainable):
            adv_states[i] = applied[pos * obs_dim:(pos + 1) * obs_dim]
        with torch.no_grad():
            t_act = maddpg.choose_action([adv_states[i] for i in trainable])
        actions = runner._build_full_actions(t_act, n_total, trainable, maddpg.n_actions)
        states, _r, info = env.step(actions)
        losses.append(float(info.get("packet_loss_rate", 0.0)) / 100.0)
    return np.asarray(deltas), np.asarray(losses)


def describe(D, label):
    sat = float((np.abs(D) > 0.99).mean())
    # does the output actually RESPOND to the observation, or is it a fixed vector?
    per_dim_sd = float(D.std(axis=0).mean())
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    idx = np.random.default_rng(0).integers(0, len(D), size=(400, 2))
    cos = float(np.mean([Dn[i] @ Dn[j] for i, j in idx if i != j]))
    print(f"  [{label}]")
    print(f"    mean|delta|              {np.abs(D).mean():.4f}   (1.0 = tanh saturated)")
    print(f"    fraction saturated       {sat:.3%}")
    print(f"    per-dim SD across time   {per_dim_sd:.5f}   (~0 => ignores the observation)")
    print(f"    mean pairwise cosine     {cos:+.4f}   (~1 => emits one FIXED direction)")
    return dict(mean_abs=float(np.abs(D).mean()), saturated=sat,
                per_dim_sd=per_dim_sd, cosine=cos)


print("=" * 72)
print("ACTOR BEHAVIOUR: trained vs untrained")
print("=" * 72)
Dt, Lt = rollout(adv.actor, "trained")
Df, Lf = rollout(fresh, "fresh")
st = describe(Dt, "TRAINED actor")
sf = describe(Df, "UNTRAINED actor (reference)")

# how far did training actually move the weights/outputs?
drift = float(np.linalg.norm(Dt - Df) / (np.linalg.norm(Df) + 1e-12))
print(f"\n  relative output difference trained vs untrained: {drift:.3f}")

print()
print("=" * 72)
print("REWARD SENSITIVITY: can the action move the reward at all?")
print("=" * 72)
print(f"  per-step loss under TRAINED   mean {Lt.mean():.5f}  SD {Lt.std():.5f}")
print(f"  per-step loss under UNTRAINED mean {Lf.mean():.5f}  SD {Lf.std():.5f}")
eff = abs(Lt.mean() - Lf.mean())
print(f"  |difference| between the two policies: {eff:.6f}")
print(f"  step-to-step NOISE SD:                 {Lt.std():.6f}")
print(f"  --> signal-to-noise ratio: {eff / (Lt.std() + 1e-12):.4f}")
print(f"      (SNR << 1 means the critic cannot tell good deltas from bad ones,")
print(f"       so the actor gradient is dominated by noise no matter how long you train)")

json.dump({"trained": st, "untrained": sf, "output_drift": drift,
           "loss_trained": float(Lt.mean()), "loss_untrained": float(Lf.mean()),
           "loss_sd": float(Lt.std()), "snr": float(eff / (Lt.std() + 1e-12))},
          open(os.path.join(OUT, "adv_diagnosis.json"), "w"), indent=2)
print("\nwrote", os.path.join(OUT, "adv_diagnosis.json"))
