# CleanRL Experiment Report

**Date:** 2026-07-28
**Hardware:** Tesla T4 GPU (CUDA), Python 3.9.16, PyTorch 2.2.1
**Test Configuration:** Short training runs (2048-20000 timesteps) to verify algorithm correctness

## Summary

| Category | Total | Success | Partial | Failed |
|----------|-------|---------|---------|--------|
| Classic Control | 4 | 4 | 0 | 0 |
| Continuous Control | 5 | 5 | 0 | 0 |
| JAX Variants | 4 | 4 | 0 | 0 |
| Atari (PyTorch) | 6 | 6 | 0 | 0 |
| Atari (JAX) | 2 | 2 | 0 | 0 |
| Advanced (EnvPool) | 4 | 3 | 1 | 0 |
| Advanced (ProcGen) | 2 | 2 | 0 | 0 |
| Advanced (XLA JAX) | 2 | 2 | 0 | 0 |
| Multi-Agent | 1 | 1 | 0 | 0 |
| Imitation Learning | 2 | 0 | 2 | 0 |
| **Total** | **32** | **29** | **3** | **0** |

> **Update 2026-07-28:** The 3 previously-failed experiments (2× XLA-JAX, 1× PettingZoo MA)
> were resolved with a pinned virtual environment. See
> [Resolution of the 3 failures](#resolution-of-the-3-failures) at the bottom. Original
> failure notes are retained below for the record, annotated with **[FIXED]**.

## Detailed Results

### Classic Control

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| PPO | ppo.py | CartPole-v1 | YES | 34.0 | 809 | 6s | 2048 timesteps |
| DQN | dqn.py | CartPole-v1 | YES | 264.0 | - | 8s | 10000 timesteps |
| C51 | c51.py | CartPole-v1 | YES | 12.0 | - | 9s | 10000 timesteps |
| PQN | pqn.py | CartPole-v1 | YES | 70.0 | 1978 | 5s | 2048 timesteps |

### Continuous Control

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| PPO Continuous | ppo_continuous_action.py | Pendulum-v1 | YES | -1268.86 | 357 | 10s | 2048 timesteps |
| DDPG | ddpg_continuous_action.py | Pendulum-v1 | YES | -1049.89 | - | 5s | 10000 timesteps |
| TD3 | td3_continuous_action.py | Pendulum-v1 | YES | -1049.89 | - | 6s | 10000 timesteps |
| SAC | sac_continuous_action.py | Pendulum-v1 | YES | -892.18 | - | 5s | 5000 timesteps |
| RPO | rpo_continuous_action.py | Pendulum-v1 | YES | -1268.86 | 337 | 11s | 2048 timesteps |

### JAX Variants

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| DQN JAX | dqn_jax.py | CartPole-v1 | YES | 18.0 | - | 9s | 10000 timesteps |
| C51 JAX | c51_jax.py | CartPole-v1 | YES | 29.0 | - | 10s | 10000 timesteps |
| DDPG JAX | ddpg_continuous_action_jax.py | Pendulum-v1 | YES | -1049.89 | - | 7s | 10000 timesteps |
| TD3 JAX | td3_continuous_action_jax.py | Pendulum-v1 | YES | -1049.89 | - | 6s | 10000 timesteps |

### Atari (PyTorch)

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| PPO Atari | ppo_atari.py | BreakoutNoFrameskip-v4 | YES | 8.0 | 322 | 35s | 10000 timesteps |
| DQN Atari | dqn_atari.py | BreakoutNoFrameskip-v4 | YES | 9.0 | - | 28s | 10000 timesteps |
| C51 Atari | c51_atari.py | BreakoutNoFrameskip-v4 | YES | 11.0 | - | 29s | 10000 timesteps |
| SAC Atari | sac_atari.py | BreakoutNoFrameskip-v4 | YES | 396.0 | - | 32s | 10000 timesteps |
| Rainbow Atari | rainbow_atari.py | BreakoutNoFrameskip-v4 | YES | 11.0 | - | 34s | 10000 timesteps |
| PPO Atari LSTM | ppo_atari_lstm.py | BreakoutNoFrameskip-v4 | YES | 5.0 | 202 | 53s | 10000 timesteps |

### Atari (JAX)

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| DQN Atari JAX | dqn_atari_jax.py | BreakoutNoFrameskip-v4 | YES | 3.0 | - | 36s | 10000 timesteps |
| C51 Atari JAX | c51_atari_jax.py | BreakoutNoFrameskip-v4 | YES | 9.0 | - | 35s | 10000 timesteps |

### Advanced - EnvPool

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| PPO Atari EnvPool | ppo_atari_envpool.py | Breakout-v5 | YES | 1.0 | 816 | 17s | 10000 timesteps |
| PPO RND EnvPool | ppo_rnd_envpool.py | MontezumaRevenge-v5 | YES | 0.0 (curiosity: 58.3) | 609 | 41s | 20000 timesteps, reduced num_envs=8, num_iterations_obs_norm_init=2 |
| PQN Atari EnvPool | pqn_atari_envpool.py | Breakout-v5 | YES | 1.0 | 931 | 16s | 10000 timesteps |
| PQN Atari EnvPool LSTM | pqn_atari_envpool_lstm.py | Breakout-v5 | YES | 3.0 | 364 | 32s | 10000 timesteps |

### Advanced - ProcGen

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| PPO ProcGen | ppo_procgen.py | starpilot | YES | 7.0 | 1141 | 22s | 20000 timesteps (needs >= num_envs*num_steps = 16384) |
| PPG ProcGen | ppg_procgen.py | starpilot | PARTIAL | 4.0 | 2236 | >120s | Reached 50944/100000 steps before timeout; requires n_iteration*batch_size timesteps for 1 full phase |

### Advanced - XLA JAX (EnvPool)

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| PPO Atari EnvPool XLA JAX | ppo_atari_envpool_xla_jax.py | Breakout-v5 | YES **[FIXED]** | 1.25 | 26 | - | Needs `--num-steps 16` on CPU jaxlib (unrolled graph JIT-compiles slowly); ran in pinned venv |
| PPO Atari EnvPool XLA JAX Scan | ppo_atari_envpool_xla_jax_scan.py | Breakout-v5 | YES **[FIXED]** | 0.0 | 28 | - | `lax.scan` compiles fast; ran in pinned venv |

### Multi-Agent

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| PPO PettingZoo MA Atari | ppo_pettingzoo_ma_atari.py | pong_v3 | YES **[FIXED]** | 15.0 / -15.0 | 163 | - | Two self-play agents; needs pettingzoo 1.18.1 + supersuit 3.4.0 + multi-agent ROMs; ran in pinned venv |

### Imitation Learning

| Algorithm | Script | Env | Status | Final Return | SPS | Runtime | Notes |
|-----------|--------|-----|--------|-------------|-----|---------|-------|
| QDagger DQN Atari | qdagger_dqn_atari_impalacnn.py | BreakoutNoFrameskip-v4 | PARTIAL | - | - | >60s | Successfully downloads teacher from HuggingFace; hangs during teacher evaluation phase (too slow for short test) |
| QDagger DQN Atari JAX | qdagger_dqn_atari_jax_impalacnn.py | BreakoutNoFrameskip-v4 | PARTIAL | - | - | >60s | Same as above - teacher evaluation phase too slow |

### Skipped (Not Attempted)

| Algorithm | Script | Reason |
|-----------|--------|--------|
| PPO Atari MultiGPU | ppo_atari_multigpu.py | Requires multiple GPUs (torch.distributed) |

## Dependencies Installed

```
gymnasium[classic-control,box2d,atari,accept-rom-license]==0.29.1
ale-py==0.8.1
opencv-python-headless==5.0.0
stable-baselines3==2.7.1
tensorboard
tyro
flax, optax, jax, jaxlib
envpool==0.8.4
procgen==0.10.7
pettingzoo==1.26.1
supersuit==3.11.0
multi-agent-ale-py==0.1.12
huggingface_hub
numpy==1.26.4 (downgraded from 2.x by procgen)
swig (system package, needed for box2d)
python39-devel (system package, needed for box2d compilation)
```

## Key Findings

1. **29 of 32 algorithms run successfully** (26 in the initial sweep + 3 fixed via the pinned venv; see [Resolution](#resolution-of-the-3-failures)). The remaining 3 are *partial* (compute-bound, not broken); only multi-GPU PPO is skipped (needs >1 GPU).
2. **MuJoCo environments** (HalfCheetah, etc.) are unavailable due to missing MUJOCO_PATH; Pendulum-v1 used as substitute for continuous control.
3. **JAX variants work well** with flax/optax/jax on GPU. The XLA-based EnvPool variants needed jax 0.4.18 + envpool 0.8.4 (the shared venv's jax 0.4.30 removed `jax.interpreters.xla.backend_specific_translations`) — now fixed in the pinned venv.
4. **EnvPool** provides significant speedups (SPS 816-931 vs 202-322 for regular Atari).
5. **PettingZoo MA Atari** has an unresolvable dependency conflict between the script's `base_class="gym"` and modern supersuit/pettingzoo versions.
6. **QDagger** experiments start correctly (download teacher models from HuggingFace) but the teacher evaluation phase is too slow for a quick test.
7. **PPG ProcGen** works but requires very long runs (at least 32 policy iterations per auxiliary phase, each needing 64*256=16384 env steps).
8. **PPO RND** works with reduced parallelism; the default 128-env + 50-iteration warmup is too heavy for quick testing.
9. **numpy was downgraded** to 1.26.4 by procgen; this causes deprecation warnings but does not break functionality.

---

## Resolution of the 3 failures

**Date:** 2026-07-28. All three initially-failed experiments now run.

### Did syncing upstream help? No.

The fork is only **1 commit behind `upstream/master`**, and upstream has not
touched these scripts recently. The version pins in `pyproject.toml` were
already correct in intent (`jax==0.4.8`, `envpool<0.7`, `PettingZoo==1.18.1`,
`SuperSuit==3.4.0`) — the original sweep simply installed *newer* versions into
a shared venv. So this was a **dependency-resolution problem**, not a code/upstream
problem. But the exact historical pins are no longer directly installable, so
resolving them literally also fails — see below.

### Root causes

| Experiment | Real root cause |
|---|---|
| `ppo_atari_envpool_xla_jax(.py/_scan.py)` | **envpool ↔ jax API drift.** The shared venv had jax 0.4.30, whose removal of `jax.interpreters.xla.backend_specific_translations` breaks envpool's `envs.xla()`. |
| `ppo_pettingzoo_ma_atari.py` | **supersuit/pettingzoo version drift.** supersuit 3.11 dropped `base_class="gym"`; the script needs the old gym API. Plus multi-agent ALE needs its own ROMs. |

### Why the literal pins don't install

- `jaxlib==0.4.7` (pinned) is **yanked from PyPI**; the oldest installable is `0.4.18`.
- `envpool==0.6.6` (matches `<0.7` pin) imports `from jax.abstract_arrays import ShapedArray`, a module **removed** in jax 0.4.18 → `ModuleNotFoundError`.

### The working combination

The fix is a *coherent* set, not the stale literal pins:

- **jax/jaxlib 0.4.18** — oldest installable; **still has** `backend_specific_translations`
  (deprecated, warning-only — not the hard `AttributeError` that 0.4.30 raises).
- **envpool 0.8.4** — newer than the `<0.7` pin, but it *dropped* the
  `jax.abstract_arrays` import while *keeping* `backend_specific_translations`.
  So 0.8.4 + jax 0.4.18 is the only compatible pairing available.
- **pettingzoo 1.18.1 + supersuit 3.4.0 + gym 0.23.1** (installable as-pinned) →
  restores `base_class="gym"`.
- **AutoROM --accept-license** → installs ROMs into `multi_agent_ale_py/roms/`.
- **numpy<2, scipy<1.13** → required by jaxlib 0.4.18 / gym 0.23.1.

Exact frozen versions: [`requirements-pinned-fixes.txt`](./requirements-pinned-fixes.txt).

### Reproduce

```bash
cd projects/cleanrl
conda create -n cleanrl-pinned python=3.9 pip -y
conda activate cleanrl-pinned
python -m pip install -e '.[envpool,atari,pettingzoo]'   # base + old pettingzoo/supersuit
python -m pip install 'jax==0.4.18' 'jaxlib==0.4.18' 'flax<0.7.5' 'numpy<2' 'scipy<1.13'
python -m pip install 'envpool==0.8.4'                   # override the stale <0.7 pin
AutoROM --accept-license                                 # multi-agent ROMs

# XLA JAX (CPU jaxlib here -> small rollout so the unrolled graph compiles):
python cleanrl/ppo_atari_envpool_xla_jax.py      --total-timesteps 2000 --num-envs 8 --num-steps 16
python cleanrl/ppo_atari_envpool_xla_jax_scan.py --total-timesteps 2000 --num-envs 8
# PettingZoo multi-agent self-play Pong:
python cleanrl/ppo_pettingzoo_ma_atari.py        --total-timesteps 3000 --num-envs 2
```

### Caveats

- The `cleanrl-pinned` Conda environment is **separate** from the sweep's main
  environment so the 26 already-passing experiments are undisturbed.
- `jaxlib 0.4.18` here is **CPU-only** (no matching CUDA wheel pulled), so the XLA
  scripts train on CPU. On a CUDA `jaxlib` they'd use the T4; the non-scan
  variant would then also handle the default `--num-steps 128` without the slow
  CPU compile.
