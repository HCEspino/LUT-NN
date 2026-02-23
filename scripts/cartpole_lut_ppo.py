import argparse
import csv
import ctypes
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime


def _bootstrap_cuda_libs():
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
from torch.distributions import Categorical

from lut_nn import LUTBlock


class RunningNorm:
    def __init__(self, shape, eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, num_tables: int, num_comparisons: int, num_blocks: int, inter_block_norm: str, surrogate_topk: int):
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
        self.trunk = nn.Sequential(*layers)
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor
    val: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor


def record_episode_gif(model, env_id, out_path, device, obs_norm=None, max_steps=500):
    env = gym.make(env_id, render_mode="rgb_array")
    state, _ = env.reset()
    if obs_norm is not None:
        state = obs_norm.normalize(state)
    frames = []
    total_reward = 0.0
    for _ in range(max_steps):
        frames.append(env.render())
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(s)
            action = int(torch.argmax(logits, dim=1).item())
        nstate, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = obs_norm.normalize(nstate) if obs_norm is not None else nstate
        if terminated or truncated:
            frames.append(env.render())
            break
    env.close()
    imageio.mimsave(out_path, frames, fps=30)
    return total_reward, len(frames)


def evaluate_policy(model, env_id, device, obs_norm=None, episodes=10, max_steps=500, seed=0):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + 10000 + ep)
        if obs_norm is not None:
            s = obs_norm.normalize(s)
        total = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(t)
                a = int(torch.argmax(logits, dim=1).item())
            ns, r, term, trunc, _ = env.step(a)
            total += r
            s = obs_norm.normalize(ns) if obs_norm is not None else ns
            if term or trunc:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--num-tables", type=int, default=8)
    p.add_argument("--num-comparisons", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--inter-block-norm", type=str, default="none", choices=["none", "relu", "layernorm"])
    p.add_argument("--surrogate-topk", type=int, default=2)
    p.add_argument("--base-optim", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--table-optim", type=str, default="sgd", choices=["adamw", "sgd"])
    p.add_argument("--table-lr-mult", type=float, default=3.0)
    p.add_argument("--table-momentum", type=float, default=0.0)
    p.add_argument("--obs-norm", dest="obs_norm", action="store_true")
    p.add_argument("--no-obs-norm", dest="obs_norm", action="store_false")
    p.set_defaults(obs_norm=True)
    p.add_argument("--eval-every-updates", type=int, default=2)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--figures-dir", type=str, default="figures")
    p.add_argument("--done-file", type=str, default="")
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
    run_name = args.run_name if args.run_name else f"cartpole_lut_ppo_{stamp}"

    env = gym.make(args.env_id)
    reward_threshold = getattr(env.spec, "reward_threshold", None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = PPOActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=args.hidden,
        num_tables=args.num_tables,
        num_comparisons=args.num_comparisons,
        num_blocks=args.num_blocks,
        inter_block_norm=args.inter_block_norm,
        surrogate_topk=args.surrogate_topk,
    ).to(device)

    table_params, base_params = [], []
    for n, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        if n.endswith(".table"):
            table_params.append(p_)
        else:
            base_params.append(p_)

    base_lr = args.lr
    table_lr = args.lr * args.table_lr_mult

    if args.base_optim == "adamw":
        opt_base = torch.optim.AdamW(base_params, lr=base_lr, weight_decay=0.0) if base_params else None
    else:
        opt_base = torch.optim.SGD(base_params, lr=base_lr, momentum=0.9) if base_params else None

    if args.table_optim == "sgd":
        opt_table = torch.optim.SGD(table_params, lr=table_lr, momentum=args.table_momentum) if table_params else None
    else:
        opt_table = torch.optim.AdamW(table_params, lr=table_lr, weight_decay=0.0) if table_params else None

    print(f"optimizer base={args.base_optim} table={args.table_optim} base_lr={base_lr:.2e} table_lr={table_lr:.2e} table_params={len(table_params)}")

    obs_norm = RunningNorm((state_dim,)) if args.obs_norm else None

    first_gif = os.path.join(args.figures_dir, f"{run_name}_first_session.gif")
    first_reward, first_frames = record_episode_gif(model, args.env_id, first_gif, device, obs_norm=obs_norm)

    obs, _ = env.reset(seed=args.seed)
    if obs_norm is not None:
        obs_norm.update(obs)
        obs = obs_norm.normalize(obs)

    updates = max(1, args.total_steps // args.rollout_steps)

    ep_rewards = []
    episode_return = 0.0
    ppo_losses, value_losses, policy_losses, entropies = [], [], [], []
    eval_means, eval_stds = [], []
    solved_update = None

    for update in range(1, updates + 1):
        obs_buf = np.zeros((args.rollout_steps, state_dim), dtype=np.float32)
        act_buf = np.zeros((args.rollout_steps,), dtype=np.int64)
        logp_buf = np.zeros((args.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((args.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((args.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((args.rollout_steps,), dtype=np.float32)

        for t in range(args.rollout_steps):
            obs_buf[t] = obs
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = model(obs_t)
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
            act = int(a.item())

            nobs_raw, rew, term, trunc, _ = env.step(act)
            done = term or trunc

            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(rew)
            done_buf[t] = float(done)
            val_buf[t] = float(v.item())

            episode_return += rew
            if done:
                ep_rewards.append(episode_return)
                episode_return = 0.0
                nobs_raw, _ = env.reset()

            if obs_norm is not None:
                obs_norm.update(nobs_raw)
                obs = obs_norm.normalize(nobs_raw)
            else:
                obs = nobs_raw

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_v = model(obs_t)
            next_v = float(next_v.item())

        adv = np.zeros_like(rew_buf)
        lastgaelam = 0.0
        for t in reversed(range(args.rollout_steps)):
            if t == args.rollout_steps - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_value = next_v
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_value = val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_value * next_nonterminal - val_buf[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val_buf

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch = RolloutBatch(
            obs=torch.tensor(obs_buf, dtype=torch.float32, device=device),
            act=torch.tensor(act_buf, dtype=torch.long, device=device),
            logp=torch.tensor(logp_buf, dtype=torch.float32, device=device),
            rew=torch.tensor(rew_buf, dtype=torch.float32, device=device),
            done=torch.tensor(done_buf, dtype=torch.float32, device=device),
            val=torch.tensor(val_buf, dtype=torch.float32, device=device),
            adv=torch.tensor(adv, dtype=torch.float32, device=device),
            ret=torch.tensor(ret, dtype=torch.float32, device=device),
        )

        idx = np.arange(args.rollout_steps)
        pl_epoch, vl_epoch, ent_epoch, loss_epoch = [], [], [], []

        for _ in range(args.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, args.rollout_steps, args.minibatch_size):
                mb_idx = idx[start : start + args.minibatch_size]
                mb_obs = batch.obs[mb_idx]
                mb_act = batch.act[mb_idx]
                mb_logp_old = batch.logp[mb_idx]
                mb_adv = batch.adv[mb_idx]
                mb_ret = batch.ret[mb_idx]

                logits, v_pred = model(mb_obs)
                dist = Categorical(logits=logits)
                mb_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(v_pred, mb_ret)
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                if opt_base is not None:
                    opt_base.zero_grad(set_to_none=True)
                if opt_table is not None:
                    opt_table.zero_grad(set_to_none=True)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if opt_base is not None:
                    opt_base.step()
                if opt_table is not None:
                    opt_table.step()

                pl_epoch.append(float(policy_loss.item()))
                vl_epoch.append(float(value_loss.item()))
                ent_epoch.append(float(entropy.item()))
                loss_epoch.append(float(loss.item()))

        ppo_losses.append(float(np.mean(loss_epoch)))
        value_losses.append(float(np.mean(vl_epoch)))
        policy_losses.append(float(np.mean(pl_epoch)))
        entropies.append(float(np.mean(ent_epoch)))

        avg100 = float(np.mean(ep_rewards[-100:])) if ep_rewards else np.nan

        do_eval = (update % args.eval_every_updates == 0) or (update == 1) or (update == updates)
        if do_eval:
            ev_mean, ev_std = evaluate_policy(model, args.env_id, device, obs_norm=obs_norm, episodes=args.eval_episodes, seed=args.seed)
            eval_means.append(ev_mean)
            eval_stds.append(ev_std)
            print(
                f"update={update}/{updates} steps={update*args.rollout_steps} "
                f"avg100={avg100:.2f} eval={ev_mean:.1f}±{ev_std:.1f} "
                f"loss={ppo_losses[-1]:.4f} pi={policy_losses[-1]:.4f} v={value_losses[-1]:.4f} ent={entropies[-1]:.4f}"
            )
        else:
            eval_means.append(np.nan)
            eval_stds.append(np.nan)

        if reward_threshold is not None and solved_update is None and len(ep_rewards) >= 100 and avg100 >= reward_threshold:
            solved_update = update

    env.close()

    last_gif = os.path.join(args.figures_dir, f"{run_name}_last_session.gif")
    last_reward, last_frames = record_episode_gif(model, args.env_id, last_gif, device, obs_norm=obs_norm)

    csv_path = os.path.join(args.figures_dir, f"{run_name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["update", "steps", "avg100", "ppo_loss", "policy_loss", "value_loss", "entropy", "eval_mean", "eval_std"],
        )
        w.writeheader()
        for i in range(updates):
            avg100_i = float(np.mean(ep_rewards[: min(len(ep_rewards), (i + 1) * 4)][-100:])) if ep_rewards else np.nan
            w.writerow(
                {
                    "update": i + 1,
                    "steps": (i + 1) * args.rollout_steps,
                    "avg100": avg100_i,
                    "ppo_loss": ppo_losses[i],
                    "policy_loss": policy_losses[i],
                    "value_loss": value_losses[i],
                    "entropy": entropies[i],
                    "eval_mean": eval_means[i],
                    "eval_std": eval_stds[i],
                }
            )

    x = np.arange(1, updates + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(x, ppo_losses, label="ppo loss")
    axes[0].plot(x, value_losses, label="value loss", alpha=0.7)
    axes[0].plot(x, policy_losses, label="policy loss", alpha=0.7)
    axes[0].set_title("Losses")
    axes[0].legend()

    axes[1].plot(x, eval_means, label="eval mean")
    axes[1].set_title("Evaluation")
    axes[1].legend()

    running_avg100 = []
    for i in range(1, updates + 1):
        upto = min(len(ep_rewards), i * 4)
        running_avg100.append(float(np.mean(ep_rewards[:upto][-100:])) if upto > 0 else np.nan)
    axes[2].plot(x, running_avg100, label="avg100")
    if reward_threshold is not None:
        axes[2].axhline(reward_threshold, linestyle="--", alpha=0.7, label="solve threshold")
    axes[2].set_title("Episode Return (avg100)")
    axes[2].legend()

    fig.tight_layout()
    plot_path = os.path.join(args.figures_dir, f"{run_name}_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.join(args.figures_dir, f"{run_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"env={args.env_id}\n")
        f.write(f"algo=ppo\n")
        f.write(f"episodes_observed={len(ep_rewards)}\n")
        f.write(f"total_steps={args.total_steps}\n")
        f.write(f"rollout_steps={args.rollout_steps}\n")
        f.write(f"hidden={args.hidden}\n")
        f.write(f"num_tables={args.num_tables}\n")
        f.write(f"num_comparisons={args.num_comparisons}\n")
        f.write(f"surrogate_topk={args.surrogate_topk}\n")
        f.write(f"first_reward={first_reward}\n")
        f.write(f"last_reward={last_reward}\n")
        f.write(f"final_avg100={running_avg100[-1] if running_avg100 else np.nan}\n")
        f.write(f"best_reward={max(ep_rewards) if ep_rewards else np.nan}\n")
        f.write(f"reward_threshold={reward_threshold}\n")
        f.write(f"solved_update={solved_update}\n")
        f.write(f"first_gif={first_gif}\n")
        f.write(f"last_gif={last_gif}\n")
        f.write(f"metrics_csv={csv_path}\n")
        f.write(f"curves_png={plot_path}\n")

    done_file = args.done_file if args.done_file else os.path.join(args.figures_dir, f"{run_name}_done.json")
    done_payload = {
        "run_name": run_name,
        "status": "completed",
        "device": device,
        "algo": "ppo",
        "total_steps": args.total_steps,
        "final_avg100": float(running_avg100[-1]) if running_avg100 else None,
        "best_reward": float(max(ep_rewards)) if ep_rewards else None,
        "solved_update": solved_update,
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
