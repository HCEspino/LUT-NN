import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path):
    rows = list(csv.DictReader(open(path)))
    ep = np.array([int(r['episode']) for r in rows])
    reward = np.array([float(r['reward']) for r in rows])
    avg100 = np.array([float(r['avg100']) for r in rows])
    eval_mean = np.array([float(r['eval_mean']) if r['eval_mean'] not in ('', 'nan', 'NaN') else np.nan for r in rows])
    return ep, reward, avg100, eval_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--figures-dir', default='figures')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for run in args.runs:
        metrics = os.path.join(args.figures_dir, f"{run}_metrics.csv")
        ep, reward, avg100, eval_mean = load_metrics(metrics)

        ax1.plot(ep, avg100, label=f"{run} avg100", linewidth=2)
        ax1.plot(ep, reward, alpha=0.20)

        ax2.plot(ep, eval_mean, label=f"{run} eval_mean", linewidth=2)

    ax1.set_title('CartPole LUT-NN: v3 params, no ReLU, top-k surrogate comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward / avg100')
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2.set_title('Evaluation mean')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Eval return')
    ax2.legend()
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == '__main__':
    main()
