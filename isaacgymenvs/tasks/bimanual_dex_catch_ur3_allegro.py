# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time

import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi

import torch

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_from_euler_xyz, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.utils.general_utils import deg2rad


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class BimanualDexCatchUR3Allegro(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.ur3_position_noise = self.cfg["env"]["ur3PositionNoise"]
        self.ur3_rotation_noise = self.cfg["env"]["ur3RotationNoise"]
        self.ur3_dof_noise = self.cfg["env"]["ur3DofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs if osc: cubeA_pose (7) + eef_pose (7) + fingers (16) = 30
        # obs if joint: cubeA_pose (7) + joints (6) + fingers (16) = 29
        self.cfg["env"]["numObservations"] = 30 if self.control_type == "osc" else 51

        # actions if osc: delta EEF if OSC (6) + finger torques (16) = 22
        # actions if joint: joint torques (6) + finger torques (16) = 22
        self.cfg["env"]["numActions"] = 22 if self.control_type == "osc" else 22

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._pos_control = None        # position actions
        self._effort_control = None     # Torque actions

        self._ur3_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        # Left / Right arm placeholders
        self._l_eef_state = None  # end effector state (at grasping point)
        self._r_eef_state = None  # end effector state (at grasping point)

        self._l_j_eef = None  # Jacobian for left end effector
        self._r_j_eef = None  # Jacobian for right end effector

        self._l_mm = None  # Mass matrix for left
        self._r_mm = None  # Mass matrix for right

        self._l_q = None  # Joint positions           (n_envs, n_dof)
        self._l_qd = None  # Joint velocities          (n_envs, n_dof)

        self._r_q = None  # Joint positions           (n_envs, n_dof)
        self._r_qd = None  # Joint velocities          (n_envs, n_dof)

        self._l_arm_control = None  # Tensor buffer for controlling arm
        self._l_finger_control = None  # Tensor buffer for controlling gripper
        self._l_pos_control = None  # Position actions
        self._l_effort_control = None  # Torque actions

        self._r_arm_control = None  # Tensor buffer for controlling arm
        self._r_finger_control = None  # Tensor buffer for controlling gripper
        self._r_pos_control = None  # Position actions
        self._r_effort_control = None  # Torque actions

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # UR3 defaults
        self.default_left_ur3_pose = {"forward": [deg2rad(0.0), deg2rad(-90.0), deg2rad(110.0), deg2rad(-40.0), deg2rad(90.0), deg2rad(0.0)],
                                      "downward": [deg2rad(0.0), deg2rad(-120.0), deg2rad(-114.0), deg2rad(-36.0), deg2rad(80.0), deg2rad(0.0)]}
        self.default_right_ur3_pose = {"forward": [deg2rad(0.0), deg2rad(-90.0), deg2rad(-110.0), deg2rad(-160.0), deg2rad(-90.0), deg2rad(0.0)],
                                       "downward": [deg2rad(0.0), deg2rad(-120.0), deg2rad(-114.0), deg2rad(-36.0), deg2rad(80.0), deg2rad(0.0)]}

        self.default_allegro_pose = {
            "spread": [deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0),      # index
                       deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0),      # middle
                       deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0),      # ring
                       deg2rad(0.0), deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)],     # thumb
            }

        self.left_ur3_default_dof_pos = to_torch(
            self.default_left_ur3_pose["forward"] + self.default_allegro_pose["spread"], device=self.device
        )

        self.right_ur3_default_dof_pos = to_torch(
            self.default_right_ur3_pose["forward"] + self.default_allegro_pose["spread"], device=self.device
        )

        # OSC Gains
        nj = 6   # actual # of joints
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * nj, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits,
        # TODO!!!
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self.left_allegro_ur3_effort_limits[:6].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        left_allegro_ur3_asset_file = "urdf/ur_with_allegro_hand_description/urdf/ur3_allegro_left_hand.urdf"
        right_allegro_ur3_asset_file = "urdf/ur_with_allegro_hand_description/urdf/ur3_allegro_right_hand.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            left_allegro_ur3_asset_file = self.cfg["env"]["asset"].get("assetFileNameLeftAllegroUR3", right_allegro_ur3_asset_file)
            right_allegro_ur3_asset_file = self.cfg["env"]["asset"].get("assetFileNameRightAllegroUR3", right_allegro_ur3_asset_file)

        # load ur3 asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        left_allegro_ur3_asset = self.gym.load_asset(self.sim, asset_root, left_allegro_ur3_asset_file, asset_options)
        right_allegro_ur3_asset = self.gym.load_asset(self.sim, asset_root, right_allegro_ur3_asset_file, asset_options)
        self.allegro_ur3_assets = [left_allegro_ur3_asset, right_allegro_ur3_asset]

        ur3_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        ur3_dof_damping = to_torch([0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_breadth = 0.805  # y-direction
        table_length = 0.760  # x-direction
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[table_breadth, table_length, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.012
        table_stand_breadth = 0.13    # y-direction
        table_stand_length = 0.13   # x-direction
        table_stand_px = -table_breadth / 2 + table_stand_length * 0.5
        table_stand_pos = [table_stand_px, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[table_stand_breadth, table_stand_length, table_stand_height], table_opts)

        self.cubeA_size = 0.050

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        print("left ur3 body cnt: ", self.gym.get_asset_rigid_body_count(left_allegro_ur3_asset))
        print("right ur3 body cnt: ", self.gym.get_asset_rigid_body_count(right_allegro_ur3_asset))

        self.num_allegro_ur3_bodies = sum([self.gym.get_asset_rigid_body_count(asset) for asset in self.allegro_ur3_assets])
        self.num_allegro_ur3_dofs = sum([self.gym.get_asset_dof_count(asset) for asset in self.allegro_ur3_assets])

        print("num ur3 bodies: ", self.num_allegro_ur3_bodies)
        print("num ur3 dofs: ", self.num_allegro_ur3_dofs)

        # set franka dof properties
        self.left_allegro_ur3_dof_lower_limits = []
        self.left_allegro_ur3_dof_upper_limits = []
        self.left_allegro_ur3_effort_limits = []

        self.right_allegro_ur3_dof_lower_limits = []
        self.right_allegro_ur3_dof_upper_limits = []
        self.right_allegro_ur3_effort_limits = []

        self._dof_lower_limits = [self.left_allegro_ur3_dof_lower_limits, self.right_allegro_ur3_dof_lower_limits]
        self._dof_upper_limits = [self.left_allegro_ur3_dof_upper_limits, self.right_allegro_ur3_dof_upper_limits]
        self._dof_effort_limits = [self.left_allegro_ur3_effort_limits, self.right_allegro_ur3_effort_limits]

        self.bi_ur3_dof_props = []

        for idx, allegro_ur3_asset in enumerate([left_allegro_ur3_asset, right_allegro_ur3_asset]):
            ur3_dof_props = self.gym.get_asset_dof_properties(allegro_ur3_asset)

            lower_limits = []
            upper_limits = []
            effort_limits = []

            for i in range(self.num_allegro_ur3_dofs // len(self.allegro_ur3_assets)):
                # ur3_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 5 else gymapi.DOF_MODE_EFFORT
                if self.physics_engine == gymapi.SIM_PHYSX:
                    ur3_dof_props['stiffness'][i] = ur3_dof_stiffness[i]
                    ur3_dof_props['damping'][i] = ur3_dof_damping[i]
                else:
                    ur3_dof_props['stiffness'][i] = 7000.0
                    ur3_dof_props['damping'][i] = 50.0

                lower_limits.append(ur3_dof_props['lower'][i])
                upper_limits.append(ur3_dof_props['upper'][i])
                effort_limits.append(ur3_dof_props['effort'][i])

            self.bi_ur3_dof_props.append(ur3_dof_props)
            # Convert lists to tensors and assign to _dof_limits
            self._dof_lower_limits[idx] = torch.tensor(lower_limits, device=self.device)
            self._dof_upper_limits[idx] = torch.tensor(upper_limits, device=self.device)
            self._dof_effort_limits[idx] = torch.tensor(effort_limits, device=self.device)

        # Assign the converted tensors back to the original variables
        self.left_allegro_ur3_dof_lower_limits, self.right_allegro_ur3_dof_lower_limits = self._dof_lower_limits
        self.left_allegro_ur3_dof_upper_limits, self.right_allegro_ur3_dof_upper_limits = self._dof_upper_limits
        self.left_allegro_ur3_effort_limits, self.right_allegro_ur3_effort_limits = self._dof_effort_limits

        # self.ur3_dof_lower_limits = to_torch(self.ur3_dof_lower_limits, device=self.device)
        # self.ur3_dof_upper_limits = to_torch(self.ur3_dof_upper_limits, device=self.device)
        # self._ur3_effort_limits = to_torch(self._ur3_effort_limits, device=self.device)
        # self.ur3_dof_speed_scales = torch.ones_like(self.ur3_dof_lower_limits)
        # self.ur3_dof_speed_scales[[6, 7, 8, 9, 10, 11]] = 0.1
        # ur3_dof_props['effort'][6:12] = 200

        # Define start pose for franka
        left_ur3_start_pose = gymapi.Transform()
        right_ur3_start_pose = gymapi.Transform()
        left_ur3_start_pose.p = gymapi.Vec3(table_stand_px, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        right_ur3_start_pose.p = gymapi.Vec3(table_stand_px, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        # ur3_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        _q = quat_from_euler_xyz(roll=torch.tensor(deg2rad(0.0), device=self.device),
                                 pitch=torch.tensor(deg2rad(0.0), device=self.device),
                                 yaw=torch.tensor(deg2rad(180.0), device=self.device))
        # left_ur3_start_pose.r = gymapi.Quat(_q[0], _q[1], _q[2], _q[3])     # TODO
        right_ur3_start_pose.r = gymapi.Quat(_q[0], _q[1], _q[2], _q[3])

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for throwing
        self._throw_start_pos = np.array(table_pos) + np.array([table_length / 2, 0, table_thickness / 2])

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_ur3_bodies = sum([self.gym.get_asset_rigid_body_count(asset) for asset in self.allegro_ur3_assets])
        num_ur3_shapes = sum([self.gym.get_asset_rigid_shape_count(asset) for asset in self.allegro_ur3_assets])
        max_agg_bodies = num_ur3_bodies + 3     # 1 for table, table stand, cubeA
        max_agg_shapes = num_ur3_shapes + 3     # 1 for table, table stand, cubeA

        self.ur3s = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create allegro_ur3
            # Potentially randomize start pose
            if self.ur3_position_noise > 0:
                rand_xy = self.ur3_position_noise * (-1. + np.random.rand(2) * 2.0)
                left_ur3_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                    1.0 + table_thickness / 2 + table_stand_height)
            if self.ur3_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.ur3_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                left_ur3_start_pose.r = gymapi.Quat(*new_quat)
            left_ur3_actor = self.gym.create_actor(env_ptr, left_allegro_ur3_asset, left_ur3_start_pose, "left_ur3", i, 8, 0) # TODO, default: i,0,0
            right_ur3_actor = self.gym.create_actor(env_ptr, left_allegro_ur3_asset, left_ur3_start_pose, "right_ur3", i, 8, 0)
            self.gym.set_actor_dof_properties(env_ptr, left_ur3_actor, self.bi_ur3_dof_props[0])
            self.gym.set_actor_dof_properties(env_ptr, right_ur3_actor, self.bi_ur3_dof_props[1])

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.ur3s.append((left_ur3_actor, right_ur3_actor))     # save as tuple

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        left_ur3_handle = 0
        right_ur3_handle = 0
        self.handles = {
            # UR3
            "hand_left": self.gym.find_actor_rigid_body_handle(env_ptr, left_ur3_handle, "tool0"),
            "grip_site_left": self.gym.find_actor_rigid_body_handle(env_ptr, left_ur3_handle, "allegro_grip_site"),
            "hand_right": self.gym.find_actor_rigid_body_handle(env_ptr, right_ur3_handle, "tool0"),
            "grip_site_right": self.gym.find_actor_rigid_body_handle(env_ptr, right_ur3_handle + 1, "allegro_grip_site"),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        dof_per_arm = self.num_allegro_ur3_dofs // len(self.allegro_ur3_assets)
        self._l_q = self._dof_state[:, :dof_per_arm, 0]
        self._l_qd = self._dof_state[:, :dof_per_arm, 1]
        self._r_q = self._dof_state[:, dof_per_arm:, 0]
        self._r_qd = self._dof_state[:, dof_per_arm:, 1]
        self._l_eef_state = self._rigid_body_state[:, self.handles["grip_site_left"], :]   # TODO, grip_site
        self._r_eef_state = self._rigid_body_state[:, self.handles["grip_site_right"], :]  # TODO, grip_site
        # self._grip_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        _l_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "left_ur3")
        l_jacobian = gymtorch.wrap_tensor(_l_jacobian)
        _r_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "right_ur3")
        r_jacobian = gymtorch.wrap_tensor(_r_jacobian)

        '''
        'adaptor_joint' = {int} 11
        'base_joint' = {int} 0
        'base_link-base_fixed_joint' = {int} 1
        'base_link-base_link_inertia' = {int} 2
        'elbow_joint' = {int} 5
        'flange-tool0' = {int} 10
        'joint_0' = {int} 13
        'joint_1' = {int} 14
        'joint_10' = {int} 30
        'joint_11' = {int} 31
        'joint_11_tip' = {int} 32
        'joint_12' = {int} 18
        'joint_13' = {int} 19
        'joint_14' = {int} 20
        'joint_15' = {int} 21
        'joint_15_tip' = {int} 22
        'joint_2' = {int} 15
        'joint_3' = {int} 16
        'joint_3_tip' = {int} 17
        'joint_4' = {int} 23
        'joint_5' = {int} 24
        'joint_6' = {int} 25
        'joint_7' = {int} 26
        'joint_7_tip' = {int} 27
        'joint_8' = {int} 28
        'joint_9' = {int} 29
        'root_to_base' = {int} 12
        'shoulder_lift_joint' = {int} 4
        'shoulder_pan_joint' = {int} 3
        'wrist_1_joint' = {int} 6
        'wrist_2_joint' = {int} 7
        'wrist_3-flange' = {int} 9
        'wrist_3_joint' = {int} 8
        'wrist_3_link-ft_frame' = {int} 33
        '''

        # left arm
        temp = self.gym.get_actor_joint_dict(env_ptr, left_ur3_handle)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, left_ur3_handle)['flange-tool0']
        self._l_j_eef = l_jacobian[:, hand_joint_index, :, :6]
        _l_massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "left_ur3")
        _l_mm = gymtorch.wrap_tensor(_l_massmatrix)
        self._l_mm = _l_mm[:, :6, :6]

        # right arm
        temp = self.gym.get_actor_joint_dict(env_ptr, right_ur3_handle)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, right_ur3_handle)['flange-tool0']
        self._r_j_eef = r_jacobian[:, hand_joint_index, :, :6]
        _r_massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "right_ur3")
        _r_mm = gymtorch.wrap_tensor(_r_massmatrix)
        self._r_mm = _r_mm[:, :6, :6]

        self._cubeA_state = self._root_state[:, self._cubeA_id, :]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._l_eef_state[:, 0]) * self.cubeA_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        half = self.num_dofs // len(self.allegro_ur3_assets)
        self._l_pos_control = self._pos_control[:, :half]
        self._l_effort_control = torch.zeros_like(self._l_pos_control)
        self._r_pos_control = self._pos_control[:, half:]
        self._r_effort_control = torch.zeros_like(self._r_pos_control)

        # Initialize control
        self._l_arm_control = self._l_effort_control[:, :6]
        self._l_finger_control = self._l_effort_control[:, 6:]
        self._r_arm_control = self._r_effort_control[:, :6]
        self._r_finger_control = self._r_effort_control[:, 6:]

        # Initialize indices
        num_actors = 5
        self._global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

    def _update_states(self):

        self.states.update({
            # Left Allegro UR3
            "l_q": self._l_q[:, :6],
            "l_q_finger": self._l_q[:, 6:],
            "l_eef_pos": self._l_eef_state[:, :3],
            "l_eef_quat": self._l_eef_state[:, 3:7],
            "l_eef_vel": self._l_eef_state[:, 7:],
            # Right Allegro UR3
            "r_q": self._r_q[:, :6],
            "r_q_finger": self._r_q[:, 6:],
            "r_eef_pos": self._r_eef_state[:, :3],
            "r_eef_quat": self._r_eef_state[:, 3:7],
            "r_eef_vel": self._r_eef_state[:, 7:],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative_left_hand": self._cubeA_state[:, :3] - self._l_eef_state[:, :3],
            "cubeA_pos_relative_right_hand": self._cubeA_state[:, :3] - self._r_eef_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def get_all_env_ids(self):
        return torch.ones(self.num_envs, device=self.device, dtype=torch.long)

    def compute_reward(self, actions):  # TODO!
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self._l_qd, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        if self.control_type == "osc":
            obs = ["cubeA_quat", "cubeA_pos", "l_eef_pos", "l_eef_quat", "r_eef_pos", "r_eef_quat"]
        else:
            obs = ["cubeA_quat", "cubeA_pos", "l_q", "r_q"]
        obs += ["l_q_finger"] + ["r_q_finger"]

        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubeA
        # if not self._i:
        self._reset_init_cube_state(cube='A', env_ids=env_ids)
        # self._i = True

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), self.num_allegro_ur3_dofs), device=self.device)
        ur3_default_dof_pos = torch.concat((self.left_ur3_default_dof_pos, self.right_ur3_default_dof_pos), dim=-1)
        pos = tensor_clamp(
            ur3_default_dof_pos.unsqueeze(0) +
            self.ur3_dof_noise * 2.0 * (reset_noise - 0.5),
            torch.cat(self._dof_lower_limits), torch.cat(self._dof_upper_limits))

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, 6:22] = ur3_default_dof_pos[6:22]
        pos[:, 22+6:] = ur3_default_dof_pos[22+6:]

        # Reset the internal obs accordingly
        self._l_q[env_ids, :] = pos[..., :22]
        self._l_qd[env_ids, :] = torch.zeros_like(self._l_qd[env_ids])

        self._r_q[env_ids, :] = pos[..., 22:]
        self._r_qd[env_ids, :] = torch.zeros_like(self._r_qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids] = pos
        self._effort_control[env_ids] = torch.zeros_like(pos)

        self._l_pos_control[env_ids, :] = self._pos_control[env_ids, :22]    # pos[env_ids, :22]
        self._l_effort_control[env_ids, :] = torch.zeros_like(self._l_pos_control[env_ids])

        self._r_pos_control[env_ids, :] = self._pos_control[env_ids, 22:]   # pos[env_ids, 22:]
        self._r_effort_control[env_ids, :] = torch.zeros_like(self._r_pos_control[env_ids])

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Sampling is "centered" around middle of table
        # centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
        biased_cube_xy_state = torch.tensor(self._throw_start_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0
        # sampled_cube_state[:, :2] = biased_cube_xy_state.unsqueeze(0) + \
        #                             2.0 * self.start_position_noise * (torch.rand(num_resets, 2, device=self.device) - 0.5)
        sampled_cube_state[:, :2] = biased_cube_xy_state.unsqueeze(0) + torch.cat([
            0.5 * self.start_position_noise * (torch.rand(num_resets, 1, device=self.device) - 0.5),
            2.0 * self.start_position_noise * (torch.rand(num_resets, 1, device=self.device) - 0.5)
        ], dim=-1)

        sampled_cube_state[:, 2] = torch.tensor([1.5], device=self.device) + \
                                   2.0 * self.start_position_noise * (torch.rand(num_resets, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

        # linear/angular velocity randomization, m/s, radian/s
        this_cube_state_all[env_ids, 7:10] = 1.0 * torch.tensor([5.0, 1.0, 1.0], device=self.device) * (torch.rand(num_resets, 3, device=self.device) - 0.5)
        this_cube_state_all[env_ids, 7] = -torch.abs(this_cube_state_all[env_ids, 7])
        this_cube_state_all[env_ids, 10:] = 1.0 * torch.tensor([1.0, 1.0, 1.0], device=self.device) * (torch.rand(num_resets, 3, device=self.device) - 0.5)

    def _compute_osc_torques(self, dpose):
        d = 6   # actual joint dof
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :d], self._qd[:, :d]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.ur3_default_dof_pos[:d] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, d:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(d, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._ur3_effort_limits[:d].unsqueeze(0), self._ur3_effort_limits[:d].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        mask = torch.zeros_like(actions)
        # mask[:, 0] = 1.0
        self.actions = actions.clone().to(self.device)

        # Split arm and finger command
        u_arm, u_finger = self.actions[:, :6], self.actions[:, 6:]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._l_arm_control[:, :] = u_arm
        self._l_finger_control[:, :] = u_finger

        # Deploy actions
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torch.concat((self._l_effort_control,
                                                                                               self._r_effort_control),
                                                                                              dim=-1)))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]

            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]

            pos_list = [eef_pos]
            rot_list = [eef_rot]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip(tuple(pos_list), tuple(rot_list)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, _qd, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    cubeA_size = states["cubeA_size"]

    # distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative_left_hand"], dim=-1)
    # d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    # d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * d / 3)
    dist_reward += torch.where(d < 0.01, 1.0, 0.0)  # reward bonus

    # reward for lifting cubeA
    cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
    cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
    lift_reward = cubeA_lifted

    ur_actions_penalty = torch.sum(torch.abs(_qd[..., 0:6]), dim=-1) * 0.01
    allegro_actions_penalty = torch.sum(torch.abs(_qd[..., 7:]), dim=-1) * 0.01
    action_penalty = -1 * ur_actions_penalty - 1 * allegro_actions_penalty

    rewards = reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + action_penalty

    # Compute resets
    # drop_reset = (states["cubeA_pos"][:, 2] < -0.05) | (states["cubeB_pos"][:, 2] < -0.05)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (cubeA_height < cubeA_size / 2 + 1e-2),
                            torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
