#!/usr/bin/env python3
"""
LatentMPC -- a self-supervised world-model agent (JEPA-style), momentum edition.

  ENV: 2D point-mass with MOMENTUM. Action = FORCE; the dot has velocity + friction.
       The model sees a STACK of the last 2 frames, so the encoder must INFER VELOCITY.
  WORLD MODEL (from random play, no rewards):
       E: 2-frame stack -> latent z          (captures position AND velocity)
       P: (z, force) -> next z               (the world model, trained multi-step)
       head: z -> position                   (small readout, so we can plan on position
                                              and visualize -- the rep stays self-supervised)
       VICReg term prevents latent collapse.
  PLANNING (live): CEM imagines force sequences with P, reads predicted POSITION via head,
       and picks the sequence that drives position to the goal; execute first force, replan.
  Random force averages to ~zero net thrust -> random fails; only coordinated planning wins.
  A decoder (viz only) renders the model's IMAGINED future next to what ACTUALLY happens.

Run:  ~/ml/envs/torch/bin/python latentmpc.py
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, imageio

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
HIMG, ZDIM = 48, 64

def _disk(p, r=2.0):
    yy, xx = np.mgrid[0:HIMG, 0:HIMG]
    return ((xx - p[0]*(HIMG-1))**2 + (yy - p[1]*(HIMG-1))**2) <= r*r
def _up(img, k=3): return np.repeat(np.repeat(img, k, 0), k, 1)

class PointMass:
    def __init__(self, dt=0.06, accel=2.0, friction=0.90):
        self.dt, self.accel, self.fric = dt, accel, friction
    def reset(self):
        self.pos  = np.random.uniform(0.15, 0.85, 2).astype(np.float32)
        self.vel  = np.zeros(2, np.float32)
        self.goal = np.random.uniform(0.15, 0.85, 2).astype(np.float32)
        return self.frame(self.pos)
    def step(self, f):
        f = np.clip(f, -1, 1).astype(np.float32)
        self.vel = self.fric*self.vel + self.dt*self.accel*f
        self.pos = self.pos + self.dt*self.vel
        for i in (0, 1):
            if self.pos[i] < 0.05: self.pos[i] = 0.05; self.vel[i] *= -0.5
            if self.pos[i] > 0.95: self.pos[i] = 0.95; self.vel[i] *= -0.5
        return self.frame(self.pos)
    def frame(self, p):
        c = np.zeros((HIMG, HIMG), np.float32); c[_disk(p)] = 1.0; return c[None]
    def render_rgb(self):
        c = np.zeros((HIMG, HIMG, 3), np.float32)
        c[_disk(self.goal)] = (0.1, 0.9, 0.2); c[_disk(self.pos)] = (1.0, 1.0, 1.0)
        return _up((c*255).astype(np.uint8))
    def dist(self): return float(np.linalg.norm(self.pos - self.goal))

def stack2(prev, cur): return np.concatenate([prev, cur], 0)

class Encoder(nn.Module):
    def __init__(self, zd=ZDIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64*6*6, 256), nn.ReLU(), nn.Linear(256, zd))
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, zd=ZDIM, ad=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zd+ad, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, zd))
    def forward(self, z, a): return z + self.net(torch.cat([z, a], -1))

class Decoder(nn.Module):
    def __init__(self, zd=ZDIM):
        super().__init__()
        self.fc = nn.Linear(zd, 64*6*6)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())
    def forward(self, z): return self.net(self.fc(z).view(-1, 64, 6, 6))

def vicreg(z):
    z = z - z.mean(0); std = torch.sqrt(z.var(0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std))
    cov = (z.T @ z) / (z.shape[0]-1); cov = cov - torch.diag(torch.diag(cov))
    return var, (cov**2).sum() / z.shape[1]

# ----------------------------- data + training -----------------------------
def collect(env, episodes=700, ep_len=50):
    FR, AC, PO = [], [], []
    for _ in range(episodes):
        f = env.reset(); fr = [f]; ac = []; po = [env.pos.copy()]
        for _ in range(ep_len):
            a = np.random.uniform(-1, 1, 2).astype(np.float32)
            f = env.step(a); fr.append(f); ac.append(a); po.append(env.pos.copy())
        FR.append(np.array(fr)); AC.append(np.array(ac)); PO.append(np.array(po))
    return (torch.tensor(np.array(FR)), torch.tensor(np.array(AC)), torch.tensor(np.array(PO)))

def stacks_at(FR, e, t):
    prev = FR[e, torch.clamp(t-1, min=0)]; cur = FR[e, t]
    return torch.cat([prev, cur], 1)

def train(FR, A, PO, steps=12000, bs=128, K=6, lr=1e-3):
    enc, pred, dec, head = Encoder().to(DEV), Predictor().to(DEV), Decoder().to(DEV), nn.Linear(ZDIM, 2).to(DEV)
    opt  = torch.optim.Adam(list(enc.parameters())+list(pred.parameters())+list(head.parameters()), lr=lr)
    optd = torch.optim.Adam(dec.parameters(), lr=lr)
    FR, A, PO = FR.to(DEV), A.to(DEV), PO.to(DEV); E, T = FR.shape[0], A.shape[1]
    for s in range(steps):
        e  = torch.randint(0, E, (bs,), device=DEV)
        t0 = torch.randint(1, T-K, (bs,), device=DEV)
        z = enc(stacks_at(FR, e, t0)); var, cov = vicreg(z); mse = 0.0
        hloss = F.mse_loss(head(z), PO[e, t0])                 # readout: latent -> position
        zr = z
        for k in range(K):                                     # roll the world model K steps
            zr = pred(zr, A[e, t0+k])
            mse = mse + F.mse_loss(zr, enc(stacks_at(FR, e, t0+k+1)).detach())
            hloss = hloss + F.mse_loss(head(zr), PO[e, t0+k+1])
        loss = mse/K + 10.0*hloss/(K+1) + 25.0*var + 1.0*cov
        opt.zero_grad(); loss.backward(); opt.step()
        rec = dec(z.detach()); decloss = F.mse_loss(rec, FR[e, t0])
        optd.zero_grad(); decloss.backward(); optd.step()
        if s % 2000 == 0: print(f"  step {s:5d}  pred {mse.item()/K:.4f}  pos-readout {hloss.item()/(K+1):.4f}")
    return enc.eval(), pred.eval(), dec.eval(), head.eval()

# ----------------------------- planning (CEM / MPC) -----------------------------
@torch.no_grad()
def plan(enc, pred, head, stack, goal_pos, H=15, n=700, iters=6, elite=70):
    z = enc(torch.tensor(stack[None]).to(DEV))
    gp = torch.tensor(goal_pos, device=DEV)
    mu, std = torch.zeros(H, 2, device=DEV), torch.ones(H, 2, device=DEV)
    for _ in range(iters):
        acts = (mu + std*torch.randn(n, H, 2, device=DEV)).clamp(-1, 1)
        zc = z.repeat(n, 1); cost = torch.zeros(n, device=DEV)
        for t in range(H):
            zc = pred(zc, acts[:, t])
            cost = cost + ((head(zc) - gp)**2).sum(-1)         # drive predicted position to goal
        top = torch.topk(-cost, elite).indices
        mu, std = acts[top].mean(0), acts[top].std(0) + 1e-3
    return mu[0].cpu().numpy()

@torch.no_grad()
def evaluate(env, enc, pred, head, episodes=40, max_steps=55, thresh=0.06, gif=None, policy="plan"):
    succ, frames = 0, []
    for ep in range(episodes):
        cur = env.reset(); prev = cur.copy()
        for t in range(max_steps):
            st = stack2(prev, cur)
            a = plan(enc, pred, head, st, env.goal) if policy == "plan" else np.random.uniform(-1, 1, 2)
            prev = cur; cur = env.step(a)
            if gif and ep < 5: frames.append(env.render_rgb())
            if env.dist() < thresh: succ += 1; break
    if gif and frames: imageio.mimsave(gif, frames, fps=15)
    return succ / episodes

@torch.no_grad()
def imagine(env, enc, pred, dec, steps=20, gif=None):
    cur = env.reset(); prev = cur.copy()
    acts = [np.clip(np.random.randn(2)*0.5 + np.array([0.8, 0.5]), -1, 1).astype(np.float32) for _ in range(steps)]
    z = enc(torch.tensor(stack2(prev, cur)[None]).to(DEV)); rows = []
    for a in acts:
        z = pred(z, torch.tensor(a[None]).to(DEV))
        imag = dec(z)[0, 0].cpu().numpy()
        prev = cur; cur = env.step(a); actual = cur[0]
        sep = np.ones((HIMG, 2, 3), np.float32)*0.3
        rows.append(_up((np.concatenate([np.stack([actual]*3, -1), sep, np.stack([imag]*3, -1)], 1)*255).astype(np.uint8)))
    if gif: imageio.mimsave(gif, rows, fps=8)

if __name__ == "__main__":
    t0 = time.time(); env = PointMass()
    print(f"device: {DEV} | {torch.cuda.get_device_name(0) if DEV=='cuda' else 'cpu'}")
    print("1) collecting random play ..."); FR, AC, PO = collect(env); print(f"   {FR.shape[0]} episodes x {AC.shape[1]} steps")
    print("2) training JEPA world model + readout + decoder ..."); enc, pred, dec, head = train(FR, AC, PO)
    out = os.path.expanduser("~/ml/latentmpc"); assets = os.path.join(out, "assets"); os.makedirs(assets, exist_ok=True)
    print("3) evaluating ...")
    rnd  = evaluate(env, enc, pred, head, policy="random")
    plan_sr = evaluate(env, enc, pred, head, policy="plan", gif=os.path.join(assets, "demo.gif"))
    print("4) rendering imagined-vs-actual ..."); imagine(env, enc, pred, dec, gif=os.path.join(assets, "imagine.gif"))
    torch.save({"enc": enc.state_dict(), "pred": pred.state_dict(), "dec": dec.state_dict(), "head": head.state_dict()}, os.path.join(out, "model.pt"))
    print("\n================ RESULT ================")
    print(f"  random-force baseline  : {rnd*100:5.0f}% goals reached")
    print(f"  LATENT-PLANNING agent  : {plan_sr*100:5.0f}% goals reached  <-- never trained on goals")
    print(f"  demo.gif (reaching) + imagine.gif (left=ACTUAL, right=IMAGINED) in {assets}")
    print(f"  total time: {time.time()-t0:.0f}s")
