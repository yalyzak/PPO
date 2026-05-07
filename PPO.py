"""
Multi-agent PPO for a pure-Python game engine.

Design:
- Trainer owns the shared ActorCritic model, rollout buffers, optimizer, and learning loop.
- Agent is the component you attach to each game object.
- Many Agent instances can share one Trainer, so they share one model and learn together.
- Agents may request actions independently; the trainer keeps separate trajectories per agent.
- Supports both continuous and discrete actions:
    - action_dim_continuous: number of continuous action values.
    - action_dim_discrete: number of discrete action branches/classes.

Expected engine flow per agent:
    agent.OnEpisodeBegin()
    while episode running:
        obs = np.ndarray shape [obs_dim]
        action = agent.get_actions(obs)
        # or:
        continuous_action, discrete_action = agent.get_mixed_actions(obs)
        # apply action to game object
        agent.add_reward(reward_delta)
        # when terminal:
        agent.end_episode()

Training flow:
    stats = trainer.learn_if_ready()

Call trainer.learn_if_ready() once per engine frame/tick, or after some number of agent actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal


@dataclass
class Config:
    obs_dim: int
    action_dim_continuous: int = 0
    action_dim_discrete: int = 0
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 5
    minibatch_size: int = 256
    rollout_steps: int = 2048
    continuous_action_low: float = -1.0
    continuous_action_high: float = 1.0
    device: str = "cpu"
    performance_window: int = 100
    best_model_path: Optional[str] = None


    def __post_init__(self) -> None:
        if self.action_dim_continuous < 0:
            raise ValueError("action_dim_continuous must be >= 0")
        if self.action_dim_discrete < 0:
            raise ValueError("action_dim_discrete must be >= 0")
        if self.action_dim_continuous == 0 and self.action_dim_discrete == 0:
            raise ValueError("At least one of action_dim_continuous or action_dim_discrete must be > 0")


class ActorCritic(nn.Module):
    """Shared policy/value network for continuous, discrete, or mixed actions."""

    def __init__(self, obs_dim: int, action_dim_continuous: int, action_dim_discrete: int, hidden_size: int):
        super().__init__()
        self.action_dim_continuous = action_dim_continuous
        self.action_dim_discrete = action_dim_discrete

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.critic = nn.Linear(hidden_size, 1)

        if action_dim_continuous > 0:
            self.actor_mean = nn.Linear(hidden_size, action_dim_continuous)
            self.log_std = nn.Parameter(torch.zeros(action_dim_continuous))
        else:
            self.actor_mean = None
            self.log_std = None

        if action_dim_discrete > 0:
            self.actor_logits = nn.Linear(hidden_size, action_dim_discrete)
        else:
            self.actor_logits = None

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone(obs)
        out: Dict[str, torch.Tensor] = {"value": self.critic(x).squeeze(-1)}

        if self.action_dim_continuous > 0:
            mean = self.actor_mean(x)
            std = torch.exp(self.log_std).expand_as(mean)
            out["continuous_mean"] = mean
            out["continuous_std"] = std

        if self.action_dim_discrete > 0:
            out["discrete_logits"] = self.actor_logits(x)

        return out

    def distributions_and_value(self, obs: torch.Tensor) -> Tuple[
        Optional[Normal], Optional[Categorical], torch.Tensor]:
        out = self.forward(obs)

        continuous_dist = None
        discrete_dist = None

        if self.action_dim_continuous > 0:
            continuous_dist = Normal(out["continuous_mean"], out["continuous_std"])

        if self.action_dim_discrete > 0:
            discrete_dist = Categorical(logits=out["discrete_logits"])

        return continuous_dist, discrete_dist, out["value"]


class RolloutBuffer:
    """Stores transitions from all agents, then flattens them for PPO learning."""

    def __init__(self):
        self.agent_trajectories: Dict[Union[int, str], List[Dict[str, object]]] = {}
        self.total_steps: int = 0

    def add(self, agent_id: Union[int, str], transition: Dict[str, object]) -> None:
        if agent_id not in self.agent_trajectories:
            self.agent_trajectories[agent_id] = []
        self.agent_trajectories[agent_id].append(transition)
        self.total_steps += 1

    def __len__(self) -> int:
        return self.total_steps

    def clear(self) -> None:
        self.agent_trajectories.clear()
        self.total_steps = 0


class Trainer:
    """Owns the shared model and trains it from all agents' experiences."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ActorCritic(
            obs_dim=config.obs_dim,
            action_dim_continuous=config.action_dim_continuous,
            action_dim_discrete=config.action_dim_discrete,
            hidden_size=config.hidden_size,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer()

        # Performance tracking.
        self.training_updates: int = 0
        self.total_environment_steps: int = 0
        self.completed_episodes: int = 0
        self.episode_rewards: Deque[float] = deque(maxlen=config.performance_window)
        self.episode_lengths: Deque[int] = deque(maxlen=config.performance_window)
        self.best_average_reward: float = float("-inf")
        self.last_stats: Dict[str, float] = {}

    def _to_obs_tensor(self, observation: np.ndarray) -> torch.Tensor:
        if not isinstance(observation, np.ndarray):
            observation = np.asarray(observation, dtype=np.float32)
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.config.obs_dim:
            raise ValueError(f"Expected observation size {self.config.obs_dim}, got shape {tuple(obs.shape)}")
        return obs

    @torch.no_grad()
    def act(
            self,
            observation: np.ndarray,
            deterministic: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Dict[str, torch.Tensor]]:
        """
        Samples mixed actions from the shared policy.

        Returns:
            continuous_action:
                np.ndarray shape [action_dim_continuous], or None if no continuous branch.
            discrete_action:
                int in [0, action_dim_discrete - 1], or None if no discrete branch.
            ppo_data:
                tensors needed later for PPO storage.
        """
        obs = self._to_obs_tensor(observation)
        continuous_dist, discrete_dist, value = self.model.distributions_and_value(obs)

        log_prob_parts: List[torch.Tensor] = []
        entropy_parts: List[torch.Tensor] = []
        ppo_data: Dict[str, torch.Tensor] = {"value": value.squeeze(0)}

        continuous_action_np: Optional[np.ndarray] = None
        discrete_action_int: Optional[int] = None

        if continuous_dist is not None:
            raw_continuous = continuous_dist.mean if deterministic else continuous_dist.sample()
            continuous_log_prob = continuous_dist.log_prob(raw_continuous).sum(dim=-1)
            continuous_entropy = continuous_dist.entropy().sum(dim=-1)

            clipped_continuous = torch.clamp(
                raw_continuous,
                self.config.continuous_action_low,
                self.config.continuous_action_high,
            )
            continuous_action_np = clipped_continuous.squeeze(0).cpu().numpy()

            ppo_data["continuous_action"] = raw_continuous.squeeze(0)
            log_prob_parts.append(continuous_log_prob)
            entropy_parts.append(continuous_entropy)

        if discrete_dist is not None:
            discrete_action = torch.argmax(discrete_dist.probs, dim=-1) if deterministic else discrete_dist.sample()
            discrete_log_prob = discrete_dist.log_prob(discrete_action)
            discrete_entropy = discrete_dist.entropy()

            discrete_action_int = int(discrete_action.squeeze(0).cpu().item())

            ppo_data["discrete_action"] = discrete_action.squeeze(0)
            log_prob_parts.append(discrete_log_prob)
            entropy_parts.append(discrete_entropy)

        ppo_data["log_prob"] = torch.stack([x.squeeze(0) for x in log_prob_parts]).sum()
        ppo_data["entropy"] = torch.stack([x.squeeze(0) for x in entropy_parts]).sum()

        return continuous_action_np, discrete_action_int, ppo_data

    def store_transition(
            self,
            agent_id: Union[int, str],
            observation: np.ndarray,
            ppo_data: Dict[str, torch.Tensor],
            reward: float,
            done: bool,
            next_observation: Optional[np.ndarray],
    ) -> None:
        obs_tensor = self._to_obs_tensor(observation).squeeze(0).detach().cpu()

        # If there is no final observation, PPO treats terminal next value as 0.
        next_obs_tensor = None
        if next_observation is not None:
            next_obs_tensor = self._to_obs_tensor(next_observation).squeeze(0).detach().cpu()

        transition: Dict[str, object] = {
            "observation": obs_tensor,
            "reward": float(reward),
            "done": bool(done),
            "next_observation": next_obs_tensor,
            "value": ppo_data["value"].detach().cpu(),
            "log_prob": ppo_data["log_prob"].detach().cpu(),
        }

        if self.config.action_dim_continuous > 0:
            transition["continuous_action"] = ppo_data["continuous_action"].detach().cpu()

        if self.config.action_dim_discrete > 0:
            transition["discrete_action"] = ppo_data["discrete_action"].detach().cpu()

        self.buffer.add(agent_id, transition)
        self.total_environment_steps += 1

    @torch.no_grad()
    def _value_of_next_obs(self, next_observation: Optional[torch.Tensor], done: bool) -> torch.Tensor:
        if done or next_observation is None:
            return torch.tensor(0.0)
        next_observation = next_observation.to(self.device).unsqueeze(0)
        _, _, value = self.model.distributions_and_value(next_observation)
        return value.squeeze(0).detach().cpu()

    def _flatten_with_gae(self) -> Dict[str, torch.Tensor]:
        """
        Computes GAE separately for each agent trajectory, then flattens all agents.
        This is important because agents may request actions independently.
        """
        flat_obs: List[torch.Tensor] = []
        flat_log_probs: List[torch.Tensor] = []
        flat_values: List[torch.Tensor] = []
        flat_returns: List[torch.Tensor] = []
        flat_advantages: List[torch.Tensor] = []
        flat_continuous_actions: List[torch.Tensor] = []
        flat_discrete_actions: List[torch.Tensor] = []

        for trajectory in self.buffer.agent_trajectories.values():
            if not trajectory:
                continue

            rewards = torch.tensor([t["reward"] for t in trajectory], dtype=torch.float32)
            dones = torch.tensor([float(t["done"]) for t in trajectory], dtype=torch.float32)
            values = torch.stack([t["value"] for t in trajectory]).float()

            advantages = torch.zeros_like(rewards)
            last_gae = torch.tensor(0.0)

            for i in reversed(range(len(trajectory))):
                if i == len(trajectory) - 1:
                    next_value = self._value_of_next_obs(
                        trajectory[i]["next_observation"],
                        bool(trajectory[i]["done"]),
                    )
                else:
                    next_value = values[i + 1]

                next_non_terminal = 1.0 - dones[i]
                delta = rewards[i] + self.config.gamma * next_value * next_non_terminal - values[i]
                last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
                advantages[i] = last_gae

            returns = advantages + values

            for i, transition in enumerate(trajectory):
                flat_obs.append(transition["observation"])
                flat_log_probs.append(transition["log_prob"])
                flat_values.append(transition["value"])
                flat_advantages.append(advantages[i])
                flat_returns.append(returns[i])

                if self.config.action_dim_continuous > 0:
                    flat_continuous_actions.append(transition["continuous_action"])
                if self.config.action_dim_discrete > 0:
                    flat_discrete_actions.append(transition["discrete_action"])

        batch: Dict[str, torch.Tensor] = {
            "observations": torch.stack(flat_obs).float().to(self.device),
            "old_log_probs": torch.stack(flat_log_probs).float().to(self.device),
            "advantages": torch.stack(flat_advantages).float().to(self.device),
            "returns": torch.stack(flat_returns).float().to(self.device),
        }

        if self.config.action_dim_continuous > 0:
            batch["continuous_actions"] = torch.stack(flat_continuous_actions).float().to(self.device)

        if self.config.action_dim_discrete > 0:
            batch["discrete_actions"] = torch.stack(flat_discrete_actions).long().to(self.device)

        return batch

    def learn_if_ready(self, force: bool = False) -> Optional[Dict[str, float]]:
        """Train once when enough shared rollout data exists."""
        if len(self.buffer) < self.config.rollout_steps and not force:
            return None
        if len(self.buffer) == 0:
            return None
        stats = self.learn()
        self.last_stats = stats
        return stats

    def record_episode_result(self, episode_reward: float, episode_length: int) -> None:
        """
        Called by Agent.end_episode().

        Tracks recent performance and saves the shared model whenever the average
        episode reward over the recent window is the best so far.
        """
        self.completed_episodes += 1
        self.episode_rewards.append(float(episode_reward))
        self.episode_lengths.append(int(episode_length))

        average_reward = self.get_average_reward()
        if average_reward is not None and average_reward > self.best_average_reward:
            self.best_average_reward = average_reward

            if self.config.best_model_path is not None:
                self.save(self.config.best_model_path)

    def get_average_reward(self) -> Optional[float]:
        if len(self.episode_rewards) == 0:
            return None
        return float(np.mean(self.episode_rewards))

    def get_average_episode_length(self) -> Optional[float]:
        if len(self.episode_lengths) == 0:
            return None
        return float(np.mean(self.episode_lengths))

    def print_performance(self) -> None:
        """
        Print a simple report showing how the shared PPO model is doing.

        You can call this every few seconds, every N frames, or after trainer.learn_if_ready().
        """
        avg_reward = self.get_average_reward()
        avg_length = self.get_average_episode_length()

        avg_reward_text = "N/A" if avg_reward is None else f"{avg_reward:.3f}"
        avg_length_text = "N/A" if avg_length is None else f"{avg_length:.1f}"
        best_reward_text = "N/A" if self.best_average_reward == float("-inf") else f"{self.best_average_reward:.3f}"

        print("=" * 48)
        print("PPO Performance")
        print(f"Updates:              {self.training_updates}")
        print(f"Environment steps:    {self.total_environment_steps}")
        print(f"Completed episodes:   {self.completed_episodes}")
        print(f"Avg reward/window:    {avg_reward_text}")
        print(f"Best avg reward:      {best_reward_text}")
        print(f"Avg episode length:   {avg_length_text}")
        print(f"Best model path:      {self.config.best_model_path}")

        if self.last_stats:
            print(f"Policy loss:          {self.last_stats.get('policy_loss', 0.0):.5f}")
            print(f"Value loss:           {self.last_stats.get('value_loss', 0.0):.5f}")
            print(f"Entropy:              {self.last_stats.get('entropy', 0.0):.5f}")
            print(f"Last trained steps:   {self.last_stats.get('trained_steps', 0.0):.0f}")

        print("=" * 48)

    def learn(self) -> Dict[str, float]:
        cfg = self.config
        batch = self._flatten_with_gae()

        observations = batch["observations"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = observations.shape[0]
        indices = np.arange(batch_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, cfg.minibatch_size):
                mb_idx = indices[start: start + cfg.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_obs = observations[mb_idx_t]
                mb_old_log_probs = old_log_probs[mb_idx_t]
                mb_advantages = advantages[mb_idx_t]
                mb_returns = returns[mb_idx_t]

                continuous_dist, discrete_dist, values = self.model.distributions_and_value(mb_obs)

                new_log_prob_parts: List[torch.Tensor] = []
                entropy_parts: List[torch.Tensor] = []

                if continuous_dist is not None:
                    mb_continuous_actions = batch["continuous_actions"][mb_idx_t]
                    continuous_log_probs = continuous_dist.log_prob(mb_continuous_actions).sum(dim=-1)
                    continuous_entropy = continuous_dist.entropy().sum(dim=-1)
                    new_log_prob_parts.append(continuous_log_probs)
                    entropy_parts.append(continuous_entropy)

                if discrete_dist is not None:
                    mb_discrete_actions = batch["discrete_actions"][mb_idx_t]
                    discrete_log_probs = discrete_dist.log_prob(mb_discrete_actions)
                    discrete_entropy = discrete_dist.entropy()
                    new_log_prob_parts.append(discrete_log_probs)
                    entropy_parts.append(discrete_entropy)

                new_log_probs = torch.stack(new_log_prob_parts, dim=0).sum(dim=0)
                entropy = torch.stack(entropy_parts, dim=0).sum(dim=0).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = nn.functional.mse_loss(values, mb_returns)
                loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_entropy += float(entropy.detach().cpu())
                updates += 1

        self.buffer.clear()
        self.training_updates += 1

        return {
            "policy_loss": total_policy_loss / max(updates, 1),
            "value_loss": total_value_loss / max(updates, 1),
            "entropy": total_entropy / max(updates, 1),
            "trained_steps": float(batch_size),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class Agent:
    """
    Game-object component class.

    Attach one Agent to each game object.
    All agents may share the same Trainer to learn one shared policy.
    """
    ID = 0
    def __init__(self, trainer: Trainer, agent_id: Union[int, str]):
        self.trainer = trainer
        self.agent_id = Agent.ID
        Agent.ID += 1

        self.episode_reward: float = 0.0
        self.pending_reward: float = 0.0
        self.episode_step: int = 0

        self.last_observation: Optional[np.ndarray] = None
        self.last_ppo_data: Optional[Dict[str, torch.Tensor]] = None
        self.has_active_action: bool = False

    def Start(self):
        self.OnEpisodeBegin()

    def OnEpisodeBegin(self) -> None:
        """
        Call this from the engine when this agent/game object starts or resets an episode.
        Unity-style capitalization is kept because you requested OnEpisodeBegin.
        """
        self.episode_reward = 0.0
        self.pending_reward = 0.0
        self.episode_step = 0
        self.last_observation = None
        self.last_ppo_data = None
        self.has_active_action = False
        if hasattr(self, "parent"):
            for component in self.parent.components.values():
                if hasattr(component, 'OnEpisodeBegin') and component != self:
                    component.OnEpisodeBegin()

    def get_continuous_actions(self, observations: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Required function: returns only the continuous action branch.

        Use this if your agent has continuous actions only, or if the engine separately asks
        for continuous actions. If action_dim_continuous is 0, this raises an error.
        """
        continuous_action, _ = self.get_mixed_actions(observations, deterministic=deterministic)
        if continuous_action is None:
            raise RuntimeError("This Agent has no continuous action branch. Set action_dim_continuous > 0.")
        return continuous_action

    def get_discrete_action(self, observations: np.ndarray, deterministic: bool = False) -> int:
        """
        Returns only the discrete action branch.

        Use this if your agent has discrete actions only, or if the engine separately asks
        for discrete actions. If action_dim_discrete is 0, this raises an error.
        """
        _, discrete_action = self.get_mixed_actions(observations, deterministic=deterministic)
        if discrete_action is None:
            raise RuntimeError("This Agent has no discrete action branch. Set action_dim_discrete > 0.")
        return discrete_action

    def get_actions(self, observations: np.ndarray, deterministic: bool = False) -> Dict[str, object]:
        """
        Engine-friendly action getter.

        Returns:
            {
                "continuous": np.ndarray or None,
                "discrete": int or None,
            }
        """
        continuous, discrete = self.get_mixed_actions(observations, deterministic=deterministic)
        return {"continuous": continuous, "discrete": discrete}

    def get_mixed_actions(
            self,
            observations: np.ndarray,
            deterministic: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Gets both continuous and discrete action branches from the shared model.

        Important independent-agent pattern:
        - If this agent already had a previous action, this call stores that previous
          transition using the current observation as next_observation.
        - Because this is per-agent state, agents can request actions independently.
        """
        observations = np.asarray(observations, dtype=np.float32)

        if self.has_active_action and self.last_observation is not None and self.last_ppo_data is not None:
            self.trainer.store_transition(
                agent_id=self.agent_id,
                observation=self.last_observation,
                ppo_data=self.last_ppo_data,
                reward=self.pending_reward,
                done=False,
                next_observation=observations,
            )
            self.pending_reward = 0.0

        continuous_action, discrete_action, ppo_data = self.trainer.act(observations, deterministic=deterministic)

        self.last_observation = observations
        self.last_ppo_data = ppo_data
        self.has_active_action = True
        self.episode_step += 1

        return continuous_action, discrete_action

    def add_reward(self, reward: float) -> None:
        """Add reward to this agent's current step and episode total."""
        r = float(reward)
        self.pending_reward += r
        self.episode_reward += r

    def set_reward(self, reward: float) -> None:
        """Set this step's reward value, replacing any pending reward for the current step."""
        old_pending = self.pending_reward
        self.pending_reward = float(reward)
        self.episode_reward += self.pending_reward - old_pending

    def end_episode(self) -> None:
        """
        Call when this agent's episode ends.

        You said the engine will not provide a final observation. That is supported:
        the final transition is stored with next_observation=None and done=True,
        so the bootstrap value is treated as 0.
        """
        final_episode_reward = self.episode_reward
        final_episode_length = self.episode_step

        if self.has_active_action and self.last_observation is not None and self.last_ppo_data is not None:
            self.trainer.store_transition(
                agent_id=self.agent_id,
                observation=self.last_observation,
                ppo_data=self.last_ppo_data,
                reward=self.pending_reward,
                done=True,
                next_observation=None,
            )

        self.trainer.record_episode_result(final_episode_reward, final_episode_length)
        self.OnEpisodeBegin()


# Example pure-Python engine setup:
if __name__ == "__main__":
    config = Config(
        obs_dim=12,
        action_dim_continuous=3,
        action_dim_discrete=5,
        rollout_steps=1024,
        device="cpu",
    )
    trainer = Trainer(config)

    agents = [Agent(trainer, agent_id=i) for i in range(4)]

    for agent in agents:
        agent.OnEpisodeBegin()

    for frame in range(5000):
        # Independent stepping example: not every agent needs to act every frame.
        for agent in agents:
            if np.random.random() < 0.75:
                obs = np.random.randn(config.obs_dim).astype(np.float32)
                actions = agent.get_actions(obs)

                continuous_action = actions["continuous"]
                discrete_action = actions["discrete"]

                # Your engine applies actions here.
                # Example:
                # game_object.apply_continuous_action(continuous_action)
                # game_object.apply_discrete_action(discrete_action)

                reward = np.random.randn() * 0.01
                agent.add_reward(reward)

            if np.random.random() < 0.005:
                agent.end_episode()

        stats = trainer.learn_if_ready()
        if stats is not None:
            print("PPO update:", stats)
            trainer.print_performance()
