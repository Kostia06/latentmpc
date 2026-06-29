# LatentMPC — a self-supervised world-model agent

An agent that learns **how a world works from raw pixels — with no rewards and no labels** —
then **reaches goals it has never seen** by *imagining* outcomes and planning in a learned
representation space. It's a JEPA-style world model (à la Meta's V-JEPA 2-AC / LeCun's
objective-driven AI) wrapped in model-predictive control.

|  Agent reaching goals  |  Imagined vs. actual future  |
|:----------------------:|:----------------------------:|
| ![demo](assets/demo.gif) | ![imagine](assets/imagine.gif) |
| white = agent, green = a **new** goal each episode | left = what *actually* happened · right = what the model *imagined* |

## Result

| Policy | Goals reached |
|---|---|
| Random force | **12%** |
| **LatentMPC** (plans in latent space) | **100%** |

Trained in **~75 s on a single RTX 3080**. The agent never trains on goals — it only learns
the dynamics from random play, then plans each goal on the fly.

## How it works

1. **Encoder** `E : (2 stacked frames) → z` — a compact latent. Two frames so it can infer the
   dot's *hidden velocity* (the env has momentum).
2. **World model** `P : (z, force) → next z` — predicts the **future latent, not future pixels**
   (the JEPA idea). Trained multi-step from random play, with a VICReg term to prevent
   representation collapse and a small `z → position` readout used as the planning target.
3. **Planner (CEM / MPC)** — to reach a goal it samples hundreds of force sequences, **rolls
   them forward in latent space** with `P`, scores them by predicted distance to the goal,
   executes the first force, then **replans every step**.

Because random forces cancel out under momentum, only *coordinated* planning makes progress —
which is why random reaches 12% and planning reaches 100%.

## Run

```bash
pip install torch numpy imageio        # CUDA build of torch recommended
python latentmpc.py                    # trains, evaluates, writes demo.gif + imagine.gif
```

Single file, ~180 lines, no simulator/GL dependencies.

## Talking points

- A **world-model agent**: perception → a learned **latent dynamics model** → planning.
- **Self-supervised**: the world model is trained only to predict future *embeddings* from
  random play — no rewards, no goal labels.
- **Hidden-state inference**: the encoder recovers velocity from a pair of frames.
- **Zero-shot goals**: solved by *planning*, not a memorized policy.

## Scaling to DeepMind Control (in progress)

I extended this to a real robot environment — DeepMind Control `reacher` (a 2-link arm,
torque control, learning from rendered MuJoCo pixels via headless EGL). The whole pipeline
works end to end: GPU rendering, data collection, the latent world model trains, and the
`z → to_target` readout is accurate.

**It runs into the problem that *defines* model-based RL: model exploitation.** On the arm's
nonlinear dynamics the world model isn't accurate enough over a multi-step rollout, so the CEM
planner finds action sequences that *fool the model* into predicting success while the real
arm never reaches the target — performing **below random** (confidently wrong). This is
exactly the failure mode that **Dreamer** and **TD-MPC2** are designed to fix, with machinery
this minimal planner deliberately lacks:

- **Model ensembles + uncertainty penalties** — don't trust the model where it's unsure.
- **A learned value function** — guide planning beyond a short, error-compounding horizon.

So the toy domain is solved cleanly; scaling to real control needs value-guided planning.
That's the next build — and knowing *why* the simple version breaks is the whole point.

## Roadmap

- **Value-guided latent planning (TD-MPC2-style)** to beat model exploitation on DeepMind Control.
- **From-scratch JEPA encoder** (masking + EMA target) in place of the readout.
- **LLM goal layer**: type *"reach the top-left"* → it plans there.
- **Weights & Biases** logging + ablations (horizon, latent dim, multi-step depth).
