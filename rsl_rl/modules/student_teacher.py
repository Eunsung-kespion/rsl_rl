# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class StudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
        teacher_obs_order=None,
        env=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False  # indicates if teacher has been loaded

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        # student
        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)

        # student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        print(f"Student MLP: {self.student}")

        # teacher
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()

        # teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        
        # --- Setup for teacher observation reconstruction ---
        self.teacher_obs_order = teacher_obs_order
        self.teacher_obs_map = None
        self.env = env
        
        if self.teacher_obs_order is not None:
            if env is not None and hasattr(env, "unwrapped") and hasattr(env.unwrapped, "observation_manager"):
                self._setup_teacher_obs_map()
            else:
                print("Warning: 'teacher_obs_order' is configured, but environment does not have an observation manager. "
                      "Skipping automatic reconstruction. Will use simple concatenation instead.")

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.student(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs):
        obs = self.get_student_obs(obs)
        # print(f"shape of Student obs: {obs.shape}")
        obs = self.student_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self.student(obs)

    def evaluate(self, obs):
        obs = self.get_teacher_obs(obs)
        # print(f"shape of Teacher obs: {obs.shape}")
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)
    
    def _setup_teacher_obs_map(self):
        """
        Pre-computes a map to efficiently reconstruct the teacher's observation vector
        from the student's observation groups (e.g., 'policy', 'privileged').
        
        This method inspects the Isaac Lab environment's observation manager to find out
        where each required observation term is located (which group and which slice)
        and stores this information for fast access during the training loop.
        """
        try:
            obs_manager = self.env.unwrapped.observation_manager
        except AttributeError:
            print("ERROR: Could not find 'observation_manager' in the unwrapped environment. Cannot build teacher obs map.")
            return

        term_locations = {}
        
        # Iterate through the student's observation groups that we care about.
        for group_name in ["policy", "privileged"]:
            if group_name not in obs_manager._group_obs_term_names:
                continue
            
            # Get the parallel lists of term names and their dimensions for the current group.
            term_names_in_group = obs_manager._group_obs_term_names[group_name]
            term_dims_in_group = obs_manager._group_obs_term_dim[group_name]

            current_idx = 0
            for i, term_name in enumerate(term_names_in_group):
                try:
                    # Get the dimension tuple using the index 'i'.
                    dim_tuple = term_dims_in_group[i]
                    # The dimension is the first element of the tuple, e.g., (3,) -> 3.
                    term_dim = dim_tuple[0]
                except (IndexError, TypeError) as e:
                    raise RuntimeError(f"Could not correctly parse the dimension for term '{term_name}' in group '{group_name}'."
                                       f" Mismatch between term names and dimensions lists? Error: {e}")

                # Store the term's location: (group_name, start_index, end_index)
                term_locations[term_name] = (group_name, current_idx, current_idx + term_dim)
                current_idx += term_dim

        # Build the final reconstruction map in the exact order the teacher expects.
        self.teacher_obs_map = []
        for term_name in self.teacher_obs_order:
            if term_name in term_locations:
                self.teacher_obs_map.append(term_locations[term_name])
            else:
                # Provide a more helpful error message, showing available terms.
                available_terms = list(term_locations.keys())
                raise ValueError(f"Teacher observation term '{term_name}' from 'teacher_obs_order' was not found"
                                 f" in the student's observation groups.\n  > Available terms are: {available_terms}")
        
        print(f"[StudentTeacher] Teacher observation reconstruction map successfully created.")
        print(f"  > Teacher expects terms in this order: {self.teacher_obs_order}")
    
    def _reconstruct_teacher_obs(self, obs_dict):
        """
        Reconstructs the flat teacher observation tensor from the student's
        observation dictionary using the pre-computed map.
        
        Args:
            obs_dict: A dictionary of observation tensors, e.g., {"policy": tensor_A, "privileged": tensor_B}.

        Returns:
            A single, flat tensor ordered according to the teacher's requirements.
        """
        if self.teacher_obs_map is None:
            # If no map is configured, fall back to simple concatenation
            return None

        # List to hold the slices of tensors that will be concatenated.
        parts = []
        # Iterate through the ordered map.
        for group_name, start_idx, end_idx in self.teacher_obs_map:
            # Slice the required part from the correct group tensor.
            parts.append(obs_dict[group_name][..., start_idx:end_idx])
        
        # Concatenate all parts along the last dimension to form the final flat tensor.
        return torch.cat(parts, dim=-1)

    def get_teacher_obs(self, obs):
        # If teacher_obs_map is configured, use reconstruction
        if self.teacher_obs_map is not None:
            return self._reconstruct_teacher_obs(obs)
        
        # Otherwise, fall back to simple concatenation
        obs_list = []
        for obs_group in self.obs_groups["teacher"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        # make sure teacher is in eval mode
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            print(f"teacher_state_dict: {teacher_state_dict.keys()}")
            for key, value in teacher_state_dict.items():
                print(f"key: {key}, value: {value.shape}")
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # training does not resume
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # training resumes
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
