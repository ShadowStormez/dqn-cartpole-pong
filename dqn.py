"""DQN for CartPole-v1 and Pong (AI course home assignment)."""
import os, csv, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- replay buffer
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, obs_dtype=np.float32):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        ids = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[ids], device=device).float(),
            torch.as_tensor(self.actions[ids], device=device),
            torch.as_tensor(self.rewards[ids], device=device),
            torch.as_tensor(self.next_obs[ids], device=device).float(),
            torch.as_tensor(self.dones[ids], device=device),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------- networks
class MLPQNet(nn.Module):
    """Q-network for CartPole: state vector in, one Q-value per action out."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class CNNQNet(nn.Module):
    """Q-network for Atari (the architecture from the DQN paper)."""
    def __init__(self, n_actions, in_frames=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_frames, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.fc(self.conv(x / 255.0))


# ---------------------------------------------------------------- acting
def epsilon_by_step(step, start=1.0, end=0.05, decay_steps=10_000):
    frac = min(step / decay_steps, 1.0)
    return start + frac * (end - start)


def select_action(net, obs, epsilon, n_actions, device):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    with torch.no_grad():
        x = torch.as_tensor(obs, device=device).float().unsqueeze(0)
        return int(net(x).argmax(dim=1).item())


# ---------------------------------------------------------------- learning step
def compute_targets(target_net, rewards, next_obs, dones, gamma):
    with torch.no_grad():
        q_next = target_net(next_obs).max(dim=1).values
        return rewards + gamma * (1.0 - dones) * q_next


def dqn_update(net, target_net, optimizer, batch, gamma, max_grad_norm=10.0):
    obs, actions, rewards, next_obs, dones = batch

    q = net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    y = compute_targets(target_net, rewards, next_obs, dones, gamma)

    loss = F.smooth_l1_loss(q, y)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


def sync_target(net, target_net):
    target_net.load_state_dict(net.state_dict())


# ---------------------------------------------------------------- CartPole
CARTPOLE_CFG = dict(
    env_id="CartPole-v1", total_steps=50_000, buffer_size=10_000, batch_size=64,
    gamma=0.99, lr=1e-3, eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000,
    learning_starts=1_000, train_freq=1, target_update=500, seed=0,
)


def train_cartpole(cfg, log_path="cartpole_log.csv", model_path="cartpole_dqn.pt", print_every=5000):
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    env = gym.make(cfg["env_id"])
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = MLPQNet(obs_dim, n_actions).to(device)
    target_net = MLPQNet(obs_dim, n_actions).to(device)
    sync_target(net, target_net)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer_size"], (obs_dim,))

    obs, _ = env.reset(seed=cfg["seed"])
    ep_reward, ep_count = 0.0, 0
    rows, losses = [], []

    for step in range(1, cfg["total_steps"] + 1):
        eps = epsilon_by_step(step, cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])
        action = select_action(net, obs, eps, n_actions, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_reward += reward

        if terminated or truncated:
            ep_count += 1
            rows.append((step, ep_count, ep_reward, eps))
            obs, _ = env.reset()
            ep_reward = 0.0

        if step > cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            batch = buffer.sample(cfg["batch_size"], device)
            losses.append(dqn_update(net, target_net, optimizer, batch, cfg["gamma"]))

        if step % cfg["target_update"] == 0:
            sync_target(net, target_net)

        if step % print_every == 0:
            recent = np.mean([r[2] for r in rows[-10:]]) if rows else float("nan")
            mean_loss = np.mean(losses[-1000:]) if losses else float("nan")
            print(f"step {step:>7} | eps {eps:.3f} | episodes {ep_count:>4} | "
                  f"avg reward (last 10) {recent:7.1f} | avg loss {mean_loss:.4f}")

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "episode", "reward", "epsilon"])
        w.writerows(rows)
    torch.save(net.state_dict(), model_path)
    return net, rows, losses


def evaluate(net, env_id, episodes=10, epsilon=0.05, seed=1000):
    env = gym.make(env_id)
    n_actions = env.action_space.n
    totals = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action = select_action(net, obs, epsilon, n_actions, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        totals.append(total)
    return totals


# ---------------------------------------------------------------- plotting
def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    window = min(window, len(x))
    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_log(path):
    data = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))
    return data[:, 0], data[:, 2]          # steps, episode rewards


def plot_learning_curve(log_path, title, out_path, window=20):
    steps, rewards = load_log(log_path)
    window = min(window, len(rewards))
    avg = moving_average(rewards, window)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, rewards, alpha=0.3, label="episode reward")
    ax.plot(steps[window - 1:], avg, linewidth=2, label=f"moving average ({window} episodes)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------- Atari / Pong
def make_atari_env(env_id="PongNoFrameskip-v4"):
    env = gym.make(env_id)
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


PONG_CFG = dict(
    env_id="PongNoFrameskip-v4", total_steps=4_000_000, buffer_size=50_000, batch_size=32,
    gamma=0.99, lr=1e-4, eps_start=1.0, eps_end=0.02, eps_decay_steps=250_000,
    learning_starts=10_000, train_freq=4, target_update=1_000, checkpoint_every=50_000, seed=0,
)


def save_checkpoint(path, net, target_net, optimizer, step, ep_count, rows):
    tmp = path + ".tmp"
    torch.save({"net": net.state_dict(), "target": target_net.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step,
                "ep_count": ep_count, "rows": rows}, tmp)
    os.replace(tmp, path)


def write_log(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "episode", "reward", "epsilon"])
        w.writerows(rows)


def train_atari(cfg, out_dir="pong_run", resume=True, print_every=10_000):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    log_path = os.path.join(out_dir, "log.csv")

    env = make_atari_env(cfg["env_id"])
    n_actions = env.action_space.n
    net = CNNQNet(n_actions).to(device)
    target_net = CNNQNet(n_actions).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer_size"], (4, 84, 84), obs_dtype=np.uint8)

    start_step, ep_count, rows = 0, 0, []
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        target_net.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, ep_count, rows = ckpt["step"], ckpt["ep_count"], ckpt["rows"]
        print(f"resuming from step {start_step}")
    else:
        sync_target(net, target_net)

    np.random.seed(cfg["seed"] + start_step)
    torch.manual_seed(cfg["seed"] + start_step)

    obs, _ = env.reset(seed=cfg["seed"] + start_step)
    obs = np.array(obs)
    ep_reward, losses = 0.0, []
    t0 = time.time()

    for step in range(start_step + 1, cfg["total_steps"] + 1):
        eps = epsilon_by_step(step, cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])
        action = select_action(net, obs, eps, n_actions, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.array(next_obs)
        buffer.add(obs, action, np.sign(reward), next_obs, float(terminated))
        obs = next_obs
        ep_reward += reward

        if terminated or truncated:
            ep_count += 1
            rows.append((step, ep_count, float(ep_reward), float(eps)))
            obs, _ = env.reset()
            obs = np.array(obs)
            ep_reward = 0.0

        if len(buffer) >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            batch = buffer.sample(cfg["batch_size"], device)
            losses.append(dqn_update(net, target_net, optimizer, batch, cfg["gamma"]))

        if step % cfg["target_update"] == 0:
            sync_target(net, target_net)

        if step % print_every == 0:
            recent = np.mean([r[2] for r in rows[-10:]]) if rows else float("nan")
            mean_loss = np.mean(losses[-1000:]) if losses else float("nan")
            speed = (step - start_step) / (time.time() - t0)
            print(f"step {step:>8} | eps {eps:.3f} | episodes {ep_count:>5} | "
                  f"avg reward (last 10) {recent:6.1f} | loss {mean_loss:.4f} | {speed:.0f} steps/s")

        if step % cfg["checkpoint_every"] == 0:
            save_checkpoint(ckpt_path, net, target_net, optimizer, step, ep_count, rows)
            write_log(log_path, rows)

    save_checkpoint(ckpt_path, net, target_net, optimizer, cfg["total_steps"], ep_count, rows)
    write_log(log_path, rows)
    torch.save(net.state_dict(), os.path.join(out_dir, "pong_dqn.pt"))
    return net, rows, losses, buffer


def evaluate_atari(net, env_id="PongNoFrameskip-v4", episodes=5, epsilon=0.05, seed=1000):
    env = make_atari_env(env_id)
    n_actions = env.action_space.n
    totals = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action = select_action(net, np.array(obs), epsilon, n_actions, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        totals.append(total)
    return totals
if __name__ == "__main__":
    train_atari(PONG_CFG)
