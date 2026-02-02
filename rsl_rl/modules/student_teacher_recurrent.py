# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, Memory


class StudentTeacherRecurrent(nn.Module):
    is_recurrent = True

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
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
        teacher_obs_order=None,
        env=None,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "StudentTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )
        super().__init__()

        self.loaded_teacher = False  # indicates if teacher has been loaded
        self.teacher_recurrent = teacher_recurrent  # indicates if teacher is recurrent too
        self.num_actions = num_actions
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
        self.memory_s = Memory(num_student_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.student = MLP(rnn_hidden_dim, num_actions, student_hidden_dims, activation)

        # student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        print(f"Student RNN: {self.memory_s}")
        print(f"Student MLP: {self.student}")

        # teacher
        if self.teacher_recurrent:
            self.memory_t = Memory(
                num_teacher_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
            )
            num_teacher_obs = rnn_hidden_dim
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()
        
        # teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        if self.teacher_recurrent:
            print(f"Teacher RNN: {self.memory_t}")
        print(f"Teacher MLP: {self.teacher}")

        self._teacher_penultimate = {}

        last_linear = None
        
        for m in reversed(list(self.teacher.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError("Teacher MLP has no nn.Linear layer.")

        def _pre_hook(module, inputs):
            # inputs is a tuple; inputs[0] == penultimate latent [B*?, hidden]
            self._teacher_penultimate["feat"] = inputs[0].detach()

        self._teacher_hook = last_linear.register_forward_pre_hook(_pre_hook)

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
    
    def _to_batch_first(self, x: torch.Tensor, *, batch_size: int | None = None, like: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [T,B,H] or [B,T,H] or [B,H] -> normalize to [B,T,H]
        like: a tensor with shape [B,T,...] (to infer dims)
        """
        if x.dim() == 2:                 # [B,H] 또는 [T,H]
        # [B,H]로 보고 시간축 1 추가
            return x.unsqueeze(1).contiguous()

        if x.dim() == 3:
            B_from_arg = batch_size
            if B_from_arg is None and like is not None:
                B_from_arg = like.shape[0]

            if B_from_arg is not None:
                # [T,B,H] → [B,T,H]
                if x.shape[1] == B_from_arg and x.shape[0] != B_from_arg:
                    return x.transpose(0, 1).contiguous()
                # 이미 [B,T,H]로 가정
                return x.contiguous()

            # fallback: like가 [B,T,...]면 그와 비교
            if like is not None and x.shape[0] == like.shape[1] and x.shape[1] == like.shape[0]:
                return x.transpose(0, 1).contiguous()

            return x.contiguous()
        return x

    def _squeeze_time(self, x: torch.Tensor, *, batch_size: int | None = None) -> torch.Tensor:
        # [B,1,D] -> [B,D], otherwise keep
        if x.dim() == 3:
            if batch_size is not None and x.shape[0] == 1 and x.shape[1] == batch_size:
                x = x.transpose(0, 1).contiguous()
            if x.size(1) == 1:
                return x[:, 0, :].contiguous()
        return x.contiguous()

    def _assert_action_shape(self, a: torch.Tensor, where: str):
        if a.dim() != 2:
            raise RuntimeError(f"{where}: action must be [B, {self.num_actions}], got {tuple(a.shape)}")
        if a.shape[1] != self.num_actions:
            if a.shape[0] == self.num_actions and a.shape[1] != self.num_actions:
                a_t = a.t().contiguous()
                raise RuntimeError(
                    f"{where}: got transposed action [na,B]={tuple(a.shape)}, expected [B,na]. "
                    f"Check RNN/MLP axis handling."
                )
            raise RuntimeError(f"{where}: expected second dim {self.num_actions}, got {a.shape[1]}")

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        self.memory_s.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])

    def forward(self):
        raise NotImplementedError

    def forward_train(self, obs, *, reset_state: bool = True, masks=None, hidden_states=None):
        """
        Return all the latent and action vectors for student and teacher.
        Args:
            obs: {"policy": ..., "teacher": ...}
            reset_state: Whether to reset RNN hidden states before forward pass
            masks: Trajectory masks for batch mode training [T, B] or [B, T]. 1 for valid, 0 for padding.
            hidden_states: Optional initial hidden states for RNN
        Returns:
            l_s: student latent vector
            a_s: student action
            l_t_bar: teacher latent vector
            a_t: teacher action
        Raises:
            RuntimeError: if the teacher MLP has unexpected structure; need at least 2 modules.
            ValueError: if the standard deviation type is unknown.
        """
        # Student
        s = self.get_student_obs(obs)
        s = self.student_obs_normalizer(s)
        
        # Initialize hidden states for batch mode if needed
        if masks is not None and s.dim() == 3:
            # Batch mode: need to initialize hidden states
            batch_size = s.shape[0]
            
            # Initialize student hidden states
            if hidden_states is None or hidden_states[0] is None:
                # Create zero initial hidden states for student
                if isinstance(self.memory_s.rnn, torch.nn.LSTM):
                    h_s = torch.zeros(self.memory_s.rnn.num_layers, batch_size, 
                                     self.memory_s.rnn.hidden_size, device=s.device)
                    c_s = torch.zeros(self.memory_s.rnn.num_layers, batch_size, 
                                     self.memory_s.rnn.hidden_size, device=s.device)
                    student_hidden = (h_s, c_s)
                else:  # GRU
                    student_hidden = torch.zeros(self.memory_s.rnn.num_layers, batch_size, 
                                                self.memory_s.rnn.hidden_size, device=s.device)
            else:
                student_hidden = hidden_states[0]
            
            # s is [B, T, D], transpose to [T, B, D]
            s_rnn = s.transpose(0, 1)
            masks_rnn = masks.transpose(0, 1) if masks.dim() == 2 else masks
            mem_s = self.memory_s(s_rnn, masks=masks_rnn, hidden_states=student_hidden)
            # mem_s is [T, B, H], transpose back to [B, T, H]
            l_s = mem_s.transpose(0, 1)
        else:
            # Inference mode or already correct shape
            if reset_state:
                self.memory_s.reset(hidden_states=hidden_states[0] if hidden_states else None)
                if self.teacher_recurrent:
                    self.memory_t.reset(hidden_states=hidden_states[1] if hidden_states else None)
            mem_s = self.memory_s(s)
            l_s = self._to_batch_first(mem_s, like=s)    # [B,T,H] or [B,H]
        
        a_s = self.student(l_s)                      # [B,T,na] or [B,na]

        # Teacher (latent+action) → [B,T,*]
        l_t, a_t = self.evaluate_with_latent(obs, masks=masks, hidden_states=hidden_states)
        return l_s, a_s, l_t, a_t

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
        s = self.get_student_obs(obs)               # [B,Ds] or [B,T,Ds]
        s = self.student_obs_normalizer(s)
        B = s.shape[0]

        mem = self.memory_s(s)                      # [T,B,H] or [B,T,H] or [B,H]
        l_s = self._to_batch_first(mem, batch_size=B, like=s)  # [B,1,H] or [B,T,H]
        mean = self.student(l_s)                    # [B,1,na] or [B,T,na]
        mean = self._squeeze_time(mean, batch_size=B)          # [B,na]

        # dist
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" \
            else torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        a = self.distribution.sample()
        self._assert_action_shape(a, "act()")
        return a

    def act_inference(self, obs):
        s = self.get_student_obs(obs)
        s = self.student_obs_normalizer(s)
        B = s.shape[0]

        mem = self.memory_s(s)
        l_s = self._to_batch_first(mem, batch_size=B, like=s)   # [B,1,H] or [B,T,H]
        a = self.student(l_s)                                   # [B,1,na] or [B,T,na]
        a = self._squeeze_time(a, batch_size=B)                 # [B,na]
        self._assert_action_shape(a, "act_inference()")
        return a

    def evaluate(self, obs):
        t = self.get_teacher_obs(obs)
        t = self.teacher_obs_normalizer(t)
        B = t.shape[0]

        with torch.no_grad():
            if self.teacher_recurrent:
                self.memory_t.eval()
                mem_t = self.memory_t(t)                        # [T,B,H] or [B,T,H] or [B,H]
                t_bf  = self._to_batch_first(mem_t, batch_size=B, like=t)  # [B,1,H] or [B,T,H]
                a_t   = self.teacher(t_bf)                                  # [B,1,na] or [B,T,na]
                a_t   = self._squeeze_time(a_t, batch_size=B)               # [B,na]
            else:
                if t.dim() == 3:  # [B,T,Dt]
                    B, T, D = t.shape
                    a_flat = self.teacher(t.reshape(B*T, D))      # [B*T,na]
                    a_t    = a_flat.view(B, T, self.num_actions)
                    a_t    = self._squeeze_time(a_t, batch_size=B)  # [B,na]
                else:
                    a_t = self.teacher(t)                        # [B,na]
            self._assert_action_shape(a_t, "evaluate()")
            return a_t

    def evaluate_with_latent(self, obs, masks=None, hidden_states=None):
        """Return the latent and action vectors for teacher.

        Args:
            obs: {"policy": ..., "teacher": ...}
            masks: Trajectory masks for batch mode training [T, B] or [B, T]. 1 for valid, 0 for padding.
            hidden_states: Optional initial hidden states for RNN
        Returns:
            l_t_bar: teacher latent vector
            a_t: teacher action
        Raises:
            RuntimeError: if the teacher penultimate feature was not captured.
        """
        t = self.get_teacher_obs(obs)
        t = self.teacher_obs_normalizer(t)
        with torch.no_grad():
            if self.teacher_recurrent:
                self.memory_t.eval()
                # Check if we need to transpose for RNN (RNN expects [T, B, D])
                if masks is not None and t.dim() == 3:
                    # Batch mode: need to initialize hidden states
                    batch_size = t.shape[0]
                    
                    # Initialize teacher hidden states
                    if hidden_states is None or hidden_states[1] is None:
                        # Create zero initial hidden states for teacher
                        if isinstance(self.memory_t.rnn, torch.nn.LSTM):
                            h_t = torch.zeros(self.memory_t.rnn.num_layers, batch_size, 
                                             self.memory_t.rnn.hidden_size, device=t.device)
                            c_t = torch.zeros(self.memory_t.rnn.num_layers, batch_size, 
                                             self.memory_t.rnn.hidden_size, device=t.device)
                            teacher_hidden = (h_t, c_t)
                        else:  # GRU
                            teacher_hidden = torch.zeros(self.memory_t.rnn.num_layers, batch_size, 
                                                        self.memory_t.rnn.hidden_size, device=t.device)
                    else:
                        teacher_hidden = hidden_states[1]
                    
                    # t is [B, T, D], transpose to [T, B, D]
                    t_rnn = t.transpose(0, 1)
                    masks_rnn = masks.transpose(0, 1) if masks.dim() == 2 else masks
                    mem_t = self.memory_t(t_rnn, masks=masks_rnn, hidden_states=teacher_hidden)
                    # mem_t is [T, B, H], transpose back to [B, T, H]
                    t_bf = mem_t.transpose(0, 1)
                else:
                    mem_t = self.memory_t(t)
                    t_bf  = self._to_batch_first(mem_t, like=t)   # [B,T,H_tin]
                
                a_t   = self.teacher(t_bf)                    # [B,T,na]
                l_t   = self._teacher_penultimate["feat"]     # [B,T,H_t]
            else:
                if t.dim() == 3:
                    B, T, D = t.shape
                    a_flat = self.teacher(t.reshape(B*T, D))   # [B*T,na]
                    l_flat = self._teacher_penultimate["feat"] # [B*T,H_t]
                    a_t = a_flat.view(B, T, self.num_actions)
                    l_t = l_flat.view(B, T, -1)
                else:
                    a_t = self.teacher(t).unsqueeze(1)          # [B,1,na]
                    l_t = self._teacher_penultimate["feat"].unsqueeze(1)  # [B,1,H_t]
        return l_t.contiguous(), a_t.contiguous()

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
        
        print(f"[StudentTeacherRecurrent] Teacher observation reconstruction map successfully created.")
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
        if self.teacher_recurrent:
            return self.memory_s.hidden_states, self.memory_t.hidden_states
        else:
            return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)

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
            
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.teacher_recurrent:
                memory_t_state_dict = {}
                for key, value in state_dict.items():
                    if "memory_a." in key:
                        memory_t_state_dict[key.replace("memory_a.", "")] = value
                self.memory_t.load_state_dict(memory_t_state_dict, strict=strict)
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

    def __del__(self):
        if hasattr(self, "_teacher_hook") and self._teacher_hook is not None:
            try:
                self._teacher_hook.remove()
            except Exception:
                pass
