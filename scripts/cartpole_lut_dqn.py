import argparse
import csv
import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lut_nn import LUTBlock


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    d: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, tr: Transition):
        self.buf.append(tr)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = np.stack([b.s for b in batch], axis=0)
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        ns = np.stack([b.ns for b in batch], axis=0)
        d = np.array([b.d for b in batch], dtype=np.float32)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)


class LUTQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, num_tables: int, num_comparisons: int, num_blocks: int):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(num_blocks):
            layers += [
                LUTBlock(
                    in_features=hidden,
                    out_features=hidden,
                    num_tables=num_tables,
                    num_comparisons=num_comparisons,
                    residual=True,
                ),
                nn.ReLU(),
            ]
        layers += [nn.Linear(hidden, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def select_action(qnet, state, eps, action_dim, device):
    if random.random() < eps:
        return random.randrange(action_dim)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = qnet(s)
        return int(q.argmax(dim=1).item())


def record_episode_gif(qnet, env_id, out_path, device, max_steps=500):
    env = gym.make(env_id, render_mode="rgb_array")
    state, _ = env.reset()
    frames = []
    total_reward = 0.0
    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(qnet(s).argmax(dim=1).item())
        nstate, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = nstate
        if terminated or truncated:
            frame = env.render()
            frames.append(frame)
            break
    env.close()
    imageio.mimsave(out_path, frames, fps=30)
    return total_reward, len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-update", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-tables", type=int, default=8)
    p.add_argument("--num-comparisons", type=int, default=6)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--figures-dir", type=str, default="figures")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.figures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name if args.run_name else f"cartpole_lut_dqn_{stamp}"

    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    qnet = LUTQNet(state_dim, action_dim, args.hidden, args.num_tables, args.num_comparisons, args.num_blocks).to(device)
    qtarget = LUTQNet(state_dim, action_dim, args.hidden, args.num_tables, args.num_comparisons, args.num_blocks).to(device)
    qtarget.load_state_dict(qnet.state_dict())
    qtarget.eval()

    opt = torch.optim.AdamW(qnet.parameters(), lr=args.lr, weight_decay=0.0)
    replay = ReplayBuffer(args.buffer_size)

    first_gif = os.path.join(args.figures_dir, f"{run_name}_first_session.gif")
    first_reward, first_frames = record_episode_gif(qnet, args.env_id, first_gif, device, args.max_steps)

    global_step = 0
    eps = args.eps_start
    eps_decay = (args.eps_start - args.eps_end) / max(args.eps_decay_steps, 1)

    rewards = []
    avg100 = []
    losses = []

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_losses = []

        for _ in range(args.max_steps):
            global_step += 1
            eps = max(args.eps_end, args.eps_start - eps_decay * global_step)

            action = select_action(qnet, state, eps, action_dim, device)
            nstate, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(Transition(state, action, reward, nstate, float(done)))
            state = nstate
            ep_reward += reward

            if len(replay) >= args.batch_size and global_step >= args.warmup_steps:
                s, a, r, ns, d = replay.sample(args.batch_size)
                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                ns = torch.tensor(ns, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                q = qnet(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    next_q = qtarget(ns).max(dim=1).values
                    tgt = r + args.gamma * (1.0 - d) * next_q

                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
                opt.step()
                ep_losses.append(float(loss.item()))

                if global_step % args.target_update == 0:
                    qtarget.load_state_dict(qnet.state_dict())

            if done:
                break

        rewards.append(ep_reward)
        avg100.append(float(np.mean(rewards[-100:])))
        losses.append(float(np.mean(ep_losses)) if ep_losses else np.nan)

        if ep % 20 == 0 or ep == 1 or ep == args.episodes:
            print(
                f"episode={ep} reward={ep_reward:.1f} avg100={avg100[-1]:.2f} "
                f"eps={eps:.3f} loss={losses[-1]:.4f}"
            )

    env.close()

    last_gif = os.path.join(args.figures_dir, f"{run_name}_last_session.gif")
    last_reward, last_frames = record_episode_gif(qnet, args.env_id, last_gif, device, args.max_steps)

    csv_path = os.path.join(args.figures_dir, f"{run_name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "reward", "avg100", "loss", "epsilon"])
        w.writeheader()
        for i in range(len(rewards)):
            w.writerow(
                {
                    "episode": i + 1,
                    "reward": rewards[i],
                    "avg100": avg100[i],
                    "loss": losses[i],
                    "epsilon": max(args.eps_end, args.eps_start - eps_decay * (i + 1)),
                }
            )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ep_x = np.arange(1, len(rewards) + 1)

    axes[0].plot(ep_x, rewards, alpha=0.6, label="episode reward")
    axes[0].plot(ep_x, avg100, linewidth=2, label="avg100")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()

    axes[1].plot(ep_x, losses)
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Huber loss")

    eps_curve = [max(args.eps_end, args.eps_start - eps_decay * i) for i in ep_x]
    axes[2].plot(ep_x, eps_curve)
    axes[2].set_title("Epsilon Schedule")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("epsilon")

    fig.suptitle(
        f"{run_name} | first_reward={first_reward:.1f} last_reward={last_reward:.1f} avg100_final={avg100[-1]:.2f}"
    )
    fig.tight_layout()
    plot_path = os.path.join(args.figures_dir, f"{run_name}_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.join(args.figures_dir, f"{run_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"env={args.env_id}\n")
        f.write(f"episodes={args.episodes}\n")
        f.write(f"first_reward={first_reward}\n")
        f.write(f"last_reward={last_reward}\n")
        f.write(f"final_avg100={avg100[-1]:.4f}\n")
        f.write(f"best_reward={max(rewards):.1f}\n")
        f.write(f"first_gif={first_gif}\n")
        f.write(f"last_gif={last_gif}\n")
        f.write(f"metrics_csv={csv_path}\n")
        f.write(f"curves_png={plot_path}\n")

    print(f"FIRST_GIF={first_gif} frames={first_frames} reward={first_reward:.1f}")
    print(f"LAST_GIF={last_gif} frames={last_frames} reward={last_reward:.1f}")
    print(f"METRICS_CSV={csv_path}")
    print(f"CURVES_PNG={plot_path}")
    print(f"SUMMARY_TXT={summary_path}")


if __name__ == "__main__":
    main()
