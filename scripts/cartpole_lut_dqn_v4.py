import argparse
import csv
import ctypes
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime


def _bootstrap_cuda_libs():
    """Ensure Jetson CUDA aux libs (e.g., cuSPARSELt) are discoverable before importing torch."""
    candidates = [
        os.path.expanduser("~/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib"),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
    ]
    found = [p for p in candidates if os.path.isdir(p)]
    if found:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(":") if p]
        for p in reversed(found):
            if p not in parts:
                parts.insert(0, p)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    # Preload cuSPARSELt if present so torch import won't fail on missing linker path.
    for p in found:
        lib = os.path.join(p, "libcusparseLt.so.0")
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break


_bootstrap_cuda_libs()

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
    d: float  # bootstrap terminal flag: 1 only for true terminal (not time-limit truncation)


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
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int,
        num_tables: int,
        num_comparisons: int,
        num_blocks: int,
        inter_block_norm: str,
        surrogate_topk: int,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(
                LUTBlock(
                    in_features=hidden,
                    out_features=hidden,
                    num_tables=num_tables,
                    num_comparisons=num_comparisons,
                    residual=True,
                    surrogate_topk=surrogate_topk,
                )
            )
            if inter_block_norm == "relu":
                layers.append(nn.ReLU())
            elif inter_block_norm == "layernorm":
                layers.append(nn.LayerNorm(hidden))
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


def evaluate_policy(qnet, env_id, device, episodes=10, max_steps=500, seed=0):
    env = gym.make(env_id)
    ep_returns = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + 10000 + ep)
        total_reward = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(qnet(s).argmax(dim=1).item())
            nstate, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = nstate
            if terminated or truncated:
                break
        ep_returns.append(total_reward)
    env.close()
    return float(np.mean(ep_returns)), float(np.std(ep_returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-update", type=int, default=200, help="hard target copy frequency in env steps")
    p.add_argument("--tau", type=float, default=0.0, help="polyak soft update factor; set <=0 to disable")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-tables", type=int, default=16)
    p.add_argument("--num-comparisons", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-steps", type=int, default=15000)
    p.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ddqn"])
    p.add_argument("--inter-block-norm", type=str, default="layernorm", choices=["none", "relu", "layernorm"], help="transform between LUT blocks; layernorm is recommended for LUT hash stability")
    p.add_argument("--surrogate-topk", type=int, default=1, help="number of closest comparison pairs per table to use in surrogate backward")
    p.add_argument("--compile", action="store_true", help="use torch.compile on qnet/qtarget when available")
    p.add_argument("--train-freq", type=int, default=1, help="learn every N env steps")
    p.add_argument("--grad-steps", type=int, default=1, help="gradient updates per learning step")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--table-lr-mult", type=float, default=3.0, help="learning-rate multiplier for LUT table parameters")
    p.add_argument("--grad-clip", type=float, default=5.0, help="global grad clip norm")
    p.add_argument("--figures-dir", type=str, default="figures")
    p.add_argument("--done-file", type=str, default="", help="optional path to write completion JSON status")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but torch.cuda.is_available() is False")
    print(f"device={device} cuda_available={torch.cuda.is_available()}")

    os.makedirs(args.figures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name if args.run_name else f"cartpole_lut_dqn_v4_{stamp}"

    env = gym.make(args.env_id)
    reward_threshold = getattr(env.spec, "reward_threshold", None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    qnet = LUTQNet(
        state_dim,
        action_dim,
        args.hidden,
        args.num_tables,
        args.num_comparisons,
        args.num_blocks,
        inter_block_norm=args.inter_block_norm,
        surrogate_topk=args.surrogate_topk,
    ).to(device)
    qtarget = LUTQNet(
        state_dim,
        action_dim,
        args.hidden,
        args.num_tables,
        args.num_comparisons,
        args.num_blocks,
        inter_block_norm=args.inter_block_norm,
        surrogate_topk=args.surrogate_topk,
    ).to(device)
    qtarget.load_state_dict(qnet.state_dict())
    qtarget.eval()

    if args.compile and hasattr(torch, "compile"):
        try:
            qnet = torch.compile(qnet)
            qtarget = torch.compile(qtarget)
            print("torch_compile=enabled")
        except Exception as e:
            print(f"torch_compile=failed error={e}")

    table_params = []
    base_params = []
    for name, param in qnet.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".table"):
            table_params.append(param)
        else:
            base_params.append(param)

    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": table_params, "lr": args.lr * args.table_lr_mult},
        ],
        weight_decay=0.0,
    )
    print(f"optimizer=AdamW base_lr={args.lr:.2e} table_lr={args.lr * args.table_lr_mult:.2e} table_params={len(table_params)}")
    replay = ReplayBuffer(args.buffer_size)

    first_gif = os.path.join(args.figures_dir, f"{run_name}_first_session.gif")
    first_reward, first_frames = record_episode_gif(qnet, args.env_id, first_gif, device, args.max_steps)

    global_step = 0
    eps = args.eps_start
    eps_decay = (args.eps_start - args.eps_end) / max(args.eps_decay_steps, 1)

    rewards = []
    avg100 = []
    losses = []
    eval_means = []
    eval_stds = []

    solved_ep = None

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_losses = []

        for _ in range(args.max_steps):
            global_step += 1
            eps = max(args.eps_end, args.eps_start - eps_decay * global_step)

            action = select_action(qnet, state, eps, action_dim, device)
            nstate, reward, terminated, truncated, _ = env.step(action)

            # Use terminated for TD bootstrap cutoff; time-limit truncation should still bootstrap.
            done_for_bootstrap = float(terminated)
            done_for_loop = terminated or truncated

            replay.push(Transition(state, action, reward, nstate, done_for_bootstrap))
            state = nstate
            ep_reward += reward

            if (
                len(replay) >= args.batch_size
                and global_step >= args.warmup_steps
                and (global_step % args.train_freq == 0)
            ):
                for _g in range(args.grad_steps):
                    s, a, r, ns, d = replay.sample(args.batch_size)
                    s = torch.tensor(s, dtype=torch.float32, device=device)
                    a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
                    r = torch.tensor(r, dtype=torch.float32, device=device)
                    ns = torch.tensor(ns, dtype=torch.float32, device=device)
                    d = torch.tensor(d, dtype=torch.float32, device=device)

                    q = qnet(s).gather(1, a).squeeze(1)

                    with torch.no_grad():
                        if args.algo == "ddqn":
                            next_actions = qnet(ns).argmax(dim=1, keepdim=True)
                            next_q = qtarget(ns).gather(1, next_actions).squeeze(1)
                        else:
                            next_q = qtarget(ns).max(dim=1).values
                        tgt = r + args.gamma * (1.0 - d) * next_q

                    loss = F.smooth_l1_loss(q, tgt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.grad_clip)
                    opt.step()
                    ep_losses.append(float(loss.item()))

                    if args.tau > 0.0:
                        with torch.no_grad():
                            for p_t, p in zip(qtarget.parameters(), qnet.parameters()):
                                p_t.data.lerp_(p.data, args.tau)

                if args.target_update > 0 and global_step % args.target_update == 0:
                    qtarget.load_state_dict(qnet.state_dict())

            if done_for_loop:
                break

        rewards.append(ep_reward)
        avg100.append(float(np.mean(rewards[-100:])))
        losses.append(float(np.mean(ep_losses)) if ep_losses else np.nan)

        if ep % args.eval_every == 0 or ep == 1 or ep == args.episodes:
            ev_mean, ev_std = evaluate_policy(
                qnet,
                args.env_id,
                device,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
            )
            eval_means.append(ev_mean)
            eval_stds.append(ev_std)
            print(
                f"episode={ep} reward={ep_reward:.1f} avg100={avg100[-1]:.2f} "
                f"eval={ev_mean:.1f}±{ev_std:.1f} eps={eps:.3f} loss={losses[-1]:.4f}"
            )
        else:
            eval_means.append(np.nan)
            eval_stds.append(np.nan)

        if reward_threshold is not None and solved_ep is None and len(rewards) >= 100 and avg100[-1] >= reward_threshold:
            solved_ep = ep

    env.close()

    last_gif = os.path.join(args.figures_dir, f"{run_name}_last_session.gif")
    last_reward, last_frames = record_episode_gif(qnet, args.env_id, last_gif, device, args.max_steps)

    csv_path = os.path.join(args.figures_dir, f"{run_name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["episode", "reward", "avg100", "loss", "epsilon", "eval_mean", "eval_std"],
        )
        w.writeheader()
        for i in range(len(rewards)):
            w.writerow(
                {
                    "episode": i + 1,
                    "reward": rewards[i],
                    "avg100": avg100[i],
                    "loss": losses[i],
                    "epsilon": max(args.eps_end, args.eps_start - eps_decay * (i + 1)),
                    "eval_mean": eval_means[i],
                    "eval_std": eval_stds[i],
                }
            )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ep_x = np.arange(1, len(rewards) + 1)

    axes[0].plot(ep_x, rewards, alpha=0.45, label="episode reward")
    axes[0].plot(ep_x, avg100, linewidth=2, label="avg100")
    axes[0].axhline(reward_threshold if reward_threshold is not None else 475.0, linestyle="--", alpha=0.7, label="solve threshold")
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

    solved_txt = f"solved_ep={solved_ep}" if solved_ep is not None else "solved_ep=not_reached"
    fig.suptitle(
        f"{run_name} | algo={args.algo} first={first_reward:.1f} last={last_reward:.1f} avg100={avg100[-1]:.2f} {solved_txt}"
    )
    fig.tight_layout()
    plot_path = os.path.join(args.figures_dir, f"{run_name}_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.join(args.figures_dir, f"{run_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"env={args.env_id}\n")
        f.write(f"algo={args.algo}\n")
        f.write(f"inter_block_norm={args.inter_block_norm}\n")
        f.write(f"surrogate_topk={args.surrogate_topk}\n")
        f.write(f"table_lr_mult={args.table_lr_mult}\n")
        f.write(f"grad_clip={args.grad_clip}\n")
        f.write(f"compile={args.compile}\n")
        f.write(f"episodes={args.episodes}\n")
        f.write(f"first_reward={first_reward}\n")
        f.write(f"last_reward={last_reward}\n")
        f.write(f"final_avg100={avg100[-1]:.4f}\n")
        f.write(f"best_reward={max(rewards):.1f}\n")
        f.write(f"reward_threshold={reward_threshold}\n")
        f.write(f"solved_episode={solved_ep}\n")
        f.write(f"first_gif={first_gif}\n")
        f.write(f"last_gif={last_gif}\n")
        f.write(f"metrics_csv={csv_path}\n")
        f.write(f"curves_png={plot_path}\n")

    done_file = args.done_file if args.done_file else os.path.join(args.figures_dir, f"{run_name}_done.json")
    done_payload = {
        "run_name": run_name,
        "status": "completed",
        "device": device,
        "algo": args.algo,
        "episodes": args.episodes,
        "final_avg100": float(avg100[-1]) if avg100 else None,
        "best_reward": float(max(rewards)) if rewards else None,
        "solved_episode": solved_ep,
        "summary_txt": summary_path,
        "metrics_csv": csv_path,
        "curves_png": plot_path,
        "finished_at": datetime.now().isoformat(),
    }
    with open(done_file, "w") as f:
        json.dump(done_payload, f, indent=2)

    print(f"FIRST_GIF={first_gif} frames={first_frames} reward={first_reward:.1f}")
    print(f"LAST_GIF={last_gif} frames={last_frames} reward={last_reward:.1f}")
    print(f"METRICS_CSV={csv_path}")
    print(f"CURVES_PNG={plot_path}")
    print(f"SUMMARY_TXT={summary_path}")
    print(f"DONE_JSON={done_file}")


if __name__ == "__main__":
    main()
