# rsl_rl/rsl_rl/algorithms/dagger.py -> for recurrent policy

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer

# Access the Transition class as a nested class of RolloutStorage
Transition = RolloutStorage.Transition


class DAgger:
    """
    Implementation of the DAgger (Dataset Aggregation) algorithm.
    This version manages data aggregation internally without requiring modifications
    to the original RolloutStorage class.
    """
    policy: StudentTeacher | StudentTeacherRecurrent
    
    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        learning_rate=1e-3,
        batch_size=1024,
        max_grad_norm=1.0,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        multi_gpu_cfg: dict | None = None,
        action_smoothness_weight=0.3,
    ):
        print(f"[DAgger] Initializing DAgger with policy: {policy}")
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
            
        self.policy = policy
        self.policy.to(self.device)
        
        # Action smoothness regularization weight
        self.action_smoothness_weight = action_smoothness_weight

        # Check if policy is recurrent
        self.is_recurrent = hasattr(self.policy, 'is_recurrent') and self.policy.is_recurrent

        # The optimizer updates the parameters of the student model (MLP + RNN if recurrent)
        if self.is_recurrent:
            student_params = list(self.policy.student.parameters()) + list(self.policy.memory_s.parameters())
            self.optimizer = resolve_optimizer(optimizer)(student_params, lr=learning_rate)
            print(f"[DAgger] Recurrent policy detected. Optimizer includes Student MLP + RNN parameters.")
        else:
            self.optimizer = resolve_optimizer(optimizer)(self.policy.student.parameters(), lr=learning_rate)
            print(f"[DAgger] Feedforward policy detected. Optimizer includes Student MLP parameters only.")

        # Storage buffers
        self.temp_storage: RolloutStorage | None = None
        self.aggregated_observations = []
        self.aggregated_actions = []
        self.aggregated_dataset = []

        self.transition = Transition()
        self.beta = 1.0  # Probability of using the expert's action

        # Learning parameters
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        loss_fn_dict = {"mse": nn.functional.mse_loss, "huber": nn.functional.huber_loss}
        self.loss_fn = loss_fn_dict.get(loss_type, nn.functional.mse_loss)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.temp_storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )

    def set_beta(self, beta: float):
        """Sets the beta value for the current round."""
        self.beta = beta
        
    def train(self):
        """Sets the student policy to training mode (e.g., for dropout)."""
        self.policy.train()

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Determines the action to take in the environment based on beta.
        The expert's action is always stored for the dataset.
        """
        student_actions = self.policy.act(obs).detach()
        expert_actions = self.policy.evaluate(obs).detach()
        
        # Record the observation and the corresponding expert action for the dataset
        self.transition.observations = obs
        self.transition.actions = expert_actions

        # Mix actions for execution in the environment
        use_expert_mask = (torch.rand(student_actions.shape[0], device=self.device) < self.beta).unsqueeze(-1)
        actions_to_take = torch.where(use_expert_mask, expert_actions, student_actions)
        
        return actions_to_take

    def process_env_step(self, obs, rewards, dones):
        """Stores the transition data from the environment step into the temporary buffer."""
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.temp_storage.add_transitions(self.transition)
        self.transition.clear()
        
        # Reset hidden states for done environments in recurrent policies
        if self.is_recurrent:
            self.policy.reset(dones=dones)

    def _add_storage(self, source: RolloutStorage, destination: RolloutStorage):
        """
        Helper function to efficiently copy transitions from a source to a destination storage.
        """
        num_new_steps = source.step
        if destination.step + num_new_steps > destination.num_transitions_per_env:
            raise OverflowError(
                f"Aggregated buffer overflow! Current size: {destination.step}, "
                f"trying to add {num_new_steps}, capacity: {destination.num_transitions_per_env}"
            )

        start = destination.step
        end = destination.step + num_new_steps
        
        destination.observations[start:end].copy_(source.observations[0:num_new_steps])
        destination.actions[start:end].copy_(source.actions[0:num_new_steps])
        destination.rewards[start:end].copy_(source.rewards[0:num_new_steps])
        destination.dones[start:end].copy_(source.dones[0:num_new_steps])
        
        destination.step += num_new_steps

    
    def update(self) -> dict[str, float]:
        """
        Updates the student policy on the entire aggregated dataset.
        For recurrent policies, preserves temporal sequences. For feedforward policies, uses flat batches.
        """
        if self.temp_storage.step > 0:
            if self.is_recurrent:
                # Recurrent policy: preserve temporal sequences using trajectory-based approach
                from rsl_rl.utils import split_and_pad_trajectories
                
                # Split trajectories by episode boundaries and pad them to the same length
                padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
                    self.temp_storage.observations[0:self.temp_storage.step], 
                    self.temp_storage.dones[0:self.temp_storage.step]
                )
                # padded_obs_trajectories: TensorDict with [T_max, num_trajs, obs_dim] for each key
                # trajectory_masks: [T_max, num_trajs] - 1 for valid timesteps, 0 for padding
                
                # Process actions similarly
                padded_actions, _ = split_and_pad_trajectories(
                    self.temp_storage.actions[0:self.temp_storage.step],
                    self.temp_storage.dones[0:self.temp_storage.step]
                )
                # padded_actions: [T_max, num_trajs, action_dim]
                
                # Transpose to batch-first format: [num_trajs, T_max, dim]
                new_obs_tensordict = {}
                for key in padded_obs_trajectories.keys():
                    new_obs_tensordict[key] = padded_obs_trajectories[key].transpose(0, 1).clone().to('cpu')
                
                new_actions = padded_actions.transpose(0, 1).clone().to('cpu')
                new_masks = trajectory_masks.transpose(0, 1).clone().to('cpu')
                
                # Create TensorDataset with sequences preserved
                obs_tensors = tuple(new_obs_tensordict[key] for key in sorted(new_obs_tensordict.keys()))
                new_dataset = TensorDataset(*obs_tensors, new_actions, new_masks)
                
                print(f"[DAgger Update] Created trajectory dataset: {len(new_dataset)} trajectories, "
                      f"max length {new_actions.shape[1]}")
            else:
                # Feedforward policy: flatten as before
                new_obs_tensordict = self.temp_storage.observations[0:self.temp_storage.step].clone().to('cpu')
                new_actions = self.temp_storage.actions[0:self.temp_storage.step].clone().to('cpu').flatten(0, 1)

                obs_tensors = tuple(new_obs_tensordict[key].flatten(0, 1) for key in sorted(new_obs_tensordict.keys()))
                new_dataset = TensorDataset(*obs_tensors, new_actions)
                
                print(f"[DAgger Update] Created flat dataset: {len(new_dataset)} samples")
            
            # Add the newly created dataset to the list of aggregated datasets
            self.aggregated_dataset.append(new_dataset)
        
        self.temp_storage.clear()
        
        if not self.aggregated_dataset:
            return {"action_loss": 0, "latent_loss": 0, "smoothness_loss": 0, "total_loss": 0, "dataset_size": 0}

        # Use ConcatDataset to efficiently concatenate all datasets in memory
        full_dataset = ConcatDataset(self.aggregated_dataset)
        data_loader = DataLoader(full_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # Get observation keys from temp_storage
        obs_keys = sorted(self.temp_storage.observations.keys())

        mean_action_loss = 0.0
        mean_latent_loss = 0.0
        mean_smoothness_loss = 0.0
        updates = 0
 
        for epoch in range(self.num_learning_epochs):
            for batch_data in data_loader:
                if self.is_recurrent:
                    # batch_data: (*obs_tensors, actions, masks)
                    # Each tensor: [batch, T_max, dim]
                    trajectory_masks = batch_data[-1].to(self.device)  # [batch, T_max]
                    actions_batch = batch_data[-2].to(self.device)      # [batch, T_max, action_dim]
                    obs_batch_list = [t.to(self.device) for t in batch_data[:-2]]
                    
                    # Reconstruct observation dictionary
                    obs_batch = {key: tensor for key, tensor in zip(obs_keys, obs_batch_list)}
                    
                    # Forward pass with trajectory sequences, passing masks for RNN batch mode
                    l_s, a_s, l_t_bar, a_t = self.policy.forward_train(
                        obs_batch, reset_state=True, masks=trajectory_masks, hidden_states=None
                    )
                    
                    # Apply mask to ignore padding positions in loss
                    mask_expanded_action = trajectory_masks.unsqueeze(-1)  # [batch, T_max, 1]
                    mask_expanded_latent = trajectory_masks.unsqueeze(-1)  # [batch, T_max, 1]
                    
                    # Compute masked loss
                    loss_action = self.loss_fn(
                        a_s * mask_expanded_action, 
                        a_t * mask_expanded_action
                    )
                    loss_latent = self.loss_fn(
                        l_s * mask_expanded_latent, 
                        l_t_bar * mask_expanded_latent
                    )
                    
                    # Temporal Smoothness Regularization for recurrent policies
                    # Encourages smooth action transitions over time
                    if self.action_smoothness_weight > 0 and a_s.size(1) > 1:
                        # Compute temporal differences: a_t - a_{t-1}
                        action_diff_student = a_s[:, 1:, :] - a_s[:, :-1, :]  # [batch, T_max-1, action_dim]
                        action_diff_teacher = a_t[:, 1:, :] - a_t[:, :-1, :]  # [batch, T_max-1, action_dim]
                        
                        # Apply mask to differences (exclude transitions after done)
                        mask_diff = trajectory_masks[:, 1:].unsqueeze(-1)  # [batch, T_max-1, 1]
                        
                        # Student should have similar temporal smoothness as teacher
                        loss_smoothness = self.loss_fn(
                            action_diff_student * mask_diff,
                            action_diff_teacher * mask_diff
                        )
                    else:
                        loss_smoothness = torch.tensor(0.0, device=self.device)
                else:
                    # Feedforward: batch_data contains flattened samples
                    actions_batch = batch_data[-1].to(self.device)
                    obs_batch_list = [t.to(self.device) for t in batch_data[:-1]]
                    
                    obs_batch = {key: tensor for key, tensor in zip(obs_keys, obs_batch_list)}
                    
                    # Forward pass (no trajectory structure needed)
                    l_s, a_s, l_t_bar, a_t = self.policy.forward_train(
                        obs_batch, reset_state=True, masks=None, hidden_states=None
                    )
                    
                    loss_action = self.loss_fn(a_s, a_t)
                    loss_latent = self.loss_fn(l_s, l_t_bar)
                    loss_smoothness = torch.tensor(0.0, device=self.device)  # No smoothness for feedforward
                
                # Combine all losses
                total_loss = loss_action + loss_latent + self.action_smoothness_weight * loss_smoothness
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                if self.is_multi_gpu:
                    self.reduce_student_parameters()
                    
                if self.max_grad_norm:
                    # Clip gradients for student MLP + RNN (if recurrent)
                    if self.is_recurrent:
                        student_params = list(self.policy.student.parameters()) + list(self.policy.memory_s.parameters())
                        nn.utils.clip_grad_norm_(student_params, self.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                        
                self.optimizer.step()
                
                mean_action_loss += loss_action.item()
                mean_latent_loss += loss_latent.item()
                mean_smoothness_loss += loss_smoothness.item()
                updates += 1
        
        if updates > 0:
            mean_action_loss /= updates
            mean_latent_loss /= updates
            mean_smoothness_loss /= updates

        mean_total_loss = mean_action_loss + mean_latent_loss + self.action_smoothness_weight * mean_smoothness_loss
        
        return {
            "action_loss": mean_action_loss,
            "latent_loss": mean_latent_loss,
            "smoothness_loss": mean_smoothness_loss,
            "total_loss": mean_total_loss,
            "dataset_size": len(full_dataset)
        }
    """
    Helper functions for distributed training
    """
    def broadcast_parameters(self):
        """Broadcasts the student model parameters (MLP + RNN if recurrent) from rank 0 to all other GPUs."""
        if self.is_recurrent:
            model_params = [{
                'student': self.policy.student.state_dict(),
                'memory_s': self.policy.memory_s.state_dict()
            }]
            torch.distributed.broadcast_object_list(model_params, src=0)
            self.policy.student.load_state_dict(model_params[0]['student'])
            self.policy.memory_s.load_state_dict(model_params[0]['memory_s'])
        else:
            model_params = [self.policy.student.state_dict()]
            torch.distributed.broadcast_object_list(model_params, src=0)
            self.policy.student.load_state_dict(model_params[0])

    def reduce_student_parameters(self):
        """Averages gradients of the student model (MLP + RNN if recurrent) from all GPUs."""
        # Get all student parameters (MLP + RNN if recurrent)
        if self.is_recurrent:
            student_params = list(self.policy.student.parameters()) + list(self.policy.memory_s.parameters())
        else:
            student_params = list(self.policy.student.parameters())
        
        grads = [param.grad.view(-1) for param in student_params if param.grad is not None]
        if not grads:
            return
            
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in student_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel