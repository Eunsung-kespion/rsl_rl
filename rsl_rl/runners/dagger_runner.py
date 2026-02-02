# rsl_rl/rsl_rl/runners/dagger_runner.py

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import statistics
from collections import deque

import torch
from tensordict import TensorDict


import rsl_rl
from rsl_rl.algorithms import DAgger
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state


class LinearBetaSchedule:
    """A schedule that linearly anneals beta from 1.0 to 0.0 over a fixed number of rounds."""

    def __init__(self, total_rounds: int):
        self.total_rounds = total_rounds

    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round."""
        assert round_num >= 0
        return max(0.0, (self.total_rounds - round_num) / self.total_rounds)

class InverseDecaySchedule:
    """The original schedule from the DAgger paper, decaying beta as 1/i."""

    def __init__(self):
        pass

    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round."""
        assert round_num >= 0
        return 1.0 / (round_num + 1)

class ExponentialBetaSchedule:
    """A schedule that exponentially anneals beta."""

    def __init__(self, initial_beta: float = 1.0, decay_rate: float = 0.95):
        self.initial_beta = initial_beta
        self.decay_rate = decay_rate

    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round."""
        assert round_num >= 0
        return self.initial_beta * (self.decay_rate ** round_num)

class LinearBetaScheduleWithWarmup:
    """
    A schedule with warmup period followed by linear decay.
    During warmup, beta=1.0 (pure expert), allowing the student to learn from 
    stable expert demonstrations before gradually transitioning to its own policy.
    """

    def __init__(self, total_rounds: int, warmup_rounds: int = 30):
        """
        Args:
            total_rounds: Total number of training rounds
            warmup_rounds: Number of initial rounds with beta=1.0 (pure expert)
        """
        self.total_rounds = total_rounds
        self.warmup_rounds = warmup_rounds

    def __call__(self, round_num: int) -> float:
        """
        Computes the value of beta for the current round.
        
        Args:
            round_num: Current round number (0-indexed)
            
        Returns:
            beta value between 0.0 and 1.0
        """
        assert round_num >= 0
        
        # Warmup phase: use pure expert
        if round_num < self.warmup_rounds:
            return 1.0
        
        # Linear decay phase after warmup
        adjusted_round = round_num - self.warmup_rounds
        adjusted_total = self.total_rounds - self.warmup_rounds
        return max(0.0, (adjusted_total - adjusted_round) / adjusted_total)


class DAggerRunner(OnPolicyRunner):
    """
    On-policy runner for the DAgger (Dataset Aggregation) algorithm.
    This runner leverages the intelligent `load_state_dict` method of the
    `StudentTeacher` policy to load an expert teacher from a checkpoint.
    
    It has been modified to support Isaac Lab environments where the student's
    observation space (split into groups like 'policy' and 'privileged') differs
    from the teacher's flat, ordered observation vector.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        # --- Standard Initialization ---
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Configure multi-GPU setup if enabled
        self._configure_multi_gpu()

        # Training parameters
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Get initial observations to setup policy
        obs = self.env.get_observations()
        # Resolve observation groups, ensuring the 'teacher' group is handled for the policy
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg.get("obs_groups", {}), default_sets=["teacher"])

        # Store teacher_obs_order for policy construction
        self.teacher_obs_order = self.cfg.get("teacher_obs_order", None)

        # Construct the DAgger algorithm. The expert teacher will be loaded later via runner.load()
        self.alg: DAgger = self._construct_algorithm(obs)

        # Disable logging for non-master processes in a multi-GPU setup
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_round = 0  # DAgger uses rounds instead of iterations
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]


    def learn(self, num_rounds: int, init_at_random_ep_len: bool = False):
        """Main training loop for DAgger."""
        # Initialize the logger (e.g., TensorBoard).
        self._prepare_logging_writer()
        
        # Ensure the teacher model was loaded before starting training.
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model not loaded. Call runner.load(expert_path) before learn().")

        # DAgger uses a beta schedule to mix student and teacher actions during collection.
        # beta_schedule = LinearBetaSchedule(num_rounds)
        beta_schedule = InverseDecaySchedule()
        # beta_schedule = LinearBetaScheduleWithWarmup(total_rounds=num_rounds, warmup_rounds=num_rounds//3)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        
        # Move all observation tensors to the correct device.
        if isinstance(obs, TensorDict):
            obs = obs.to(self.device)
            
        elif isinstance(obs, dict):
             obs = {k: v.to(self.device) for k, v in obs.items() if isinstance(v, torch.Tensor)}
        else:
             obs = obs.to(self.device)

        self.alg.train()
        
        # Reset hidden states at the start of training for recurrent policies
        if hasattr(self.alg.policy, 'is_recurrent') and self.alg.policy.is_recurrent:
            self.alg.policy.reset()

        # Bookkeeping for logs
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        
        # Synchronize model parameters across GPUs if in distributed mode.
        if self.is_distributed:
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_rounds
        num_learning_iterations = num_rounds # for logging compatibility
        
        for it in range(start_iter, tot_iter):
            self.current_round = it
            start = time.time()
            
            # Update beta for the current round.
            current_beta = beta_schedule(it)
            self.alg.set_beta(current_beta)
            if not self.disable_logs:
                print(f"Round {it + 1}/{tot_iter} | Beta: {current_beta:.4f}")

            # Reset hidden states at the start of each round for recurrent policies
            if hasattr(self.alg.policy, 'is_recurrent') and self.alg.policy.is_recurrent:
                self.alg.policy.reset()

            # Data Collection (Rollout Phase)
            with torch.inference_mode():
                cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

                for _ in range(self.num_steps_per_env):
                    # Get actions from the DAgger policy. `alg.act` should be able to handle the obs dict
                    # and pass the correct observations to the student and teacher.
                    actions = self.alg.act(obs)
                    
                    # Step the environment.
                    next_obs_dict, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    
                    # Move observations to device
                    if isinstance(next_obs_dict, TensorDict):
                        next_obs_dict = next_obs_dict.to(self.device)
                    elif isinstance(next_obs_dict, dict):
                        next_obs_dict = {k: v.to(self.device) for k, v in next_obs_dict.items() if isinstance(v, torch.Tensor)}
                    else:
                        next_obs_dict = next_obs_dict.to(self.device)

                    # Update the current observation for the next loop iteration.
                    obs = next_obs_dict
                    # print(f"Obs: {obs}")
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    
                    # Process the step data for the algorithm's buffer.
                    # Note: process_env_step will handle hidden state reset for done episodes
                    self.alg.process_env_step(obs, rewards, dones)
                    
                    # Detach hidden states to prevent backprop through entire episode history
                    if hasattr(self.alg.policy, 'is_recurrent') and self.alg.policy.is_recurrent:
                        self.alg.policy.detach_hidden_states()
                    
                    # Log episode statistics.
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            collection_time = time.time() - start

            # Learning Phase: Update the student policy using the collected data.
            loss_dict = self.alg.update()
            
            learn_time = time.time() - (collection_time + start)
            self.current_learning_iteration = it
            
            # Log training statistics and save model checkpoints.
            if self.log_dir is not None and not self.disable_logs:
                log_locals = locals()
                log_locals['mean_reward'] = statistics.mean(rewbuffer) if rewbuffer else 0
                log_locals['mean_episode_length'] = statistics.mean(lenbuffer) if lenbuffer else 0
                self.log(log_locals)

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_round_{it}.pt"))
            
            ep_infos.clear()

            # At the end of training, save the state of the codebase.
            if it == tot_iter - 1 and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model.
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_round_{it}.pt"))

    def _construct_algorithm(self, obs) -> DAgger:
        """Constructs the DAgger algorithm."""
        policy_class = eval(self.policy_cfg.pop("class_name"))
        
        # The policy object contains both student and teacher networks.
        # Pass env and teacher_obs_order for automatic observation reconstruction
        policy: StudentTeacher | StudentTeacherRecurrent = policy_class(
            obs, 
            self.cfg["obs_groups"], 
            self.env.num_actions, 
            teacher_obs_order=self.teacher_obs_order,
            env=self.env,
            **self.policy_cfg
        ).to(self.device)
        print(f"policy: {policy}")
        alg_class = eval(self.alg_cfg.pop("class_name"))

        # The DAgger algorithm takes a single policy object that manages both networks.
        alg: DAgger = alg_class(
            policy,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg
        )

        alg.init_storage(
            "dagger",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg