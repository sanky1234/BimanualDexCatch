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

import cv2

from ..utils.utils import AttrDict

import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi

import torch
import torch.nn.functional as F

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_from_euler_xyz, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.base.multi_vec_task import MultiVecTask
from isaacgymenvs.tasks.utils.general_utils import deg2rad

from gym import spaces


def get_assets(attr_dict):
    assets = []
    for key, value in attr_dict.items():
        if isinstance(value, AttrDict) and 'asset' in value:
            assets.append(value.asset)
    return assets


def get_indices_from_dict(dictionary, keys):
    indices = []
    for key in keys:
        indices.append(dictionary.get(key, "Key not found"))
    return indices


def zero_state(ref_tensor: torch.tensor, obj_size_vec, device):
    # (N, 13) [x, y, z, xq, yq, zq, w, xd, yd, zd, rxd, ryd, rzd]
    _tensor = torch.zeros_like(ref_tensor, device=device)
    _tensor[:, :, 2] = torch.clamp(obj_size_vec.unsqueeze(-1).repeat(1, _tensor.shape[1]), max=0.5)
    # _tensor[:, :, 2] = 0.5
    _tensor[:, :, 6] = 1.0
    return _tensor


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

        self.controlled_experiment = self.cfg["env"].get("controlledExperiment", False)
        self.num_controlled_experiment_per_object = self.cfg["env"].get("numControlledExperimentPerObject", 10)
        self.sim_start = False

        # multi-agent RL (Heterogenuous Agent)
        self.is_multi_agent = self.cfg["env"]["multiAgent"].get("isMultiAgent", False)
        self.uniform_test = self.cfg["env"].get("uniformTest", False)
        if self.uniform_test:
            print("**** Uniform Test Mode ****")

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.noise_scale = self.cfg["env"]["noiseScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.ur3_position_noise = self.cfg["env"]["ur3PositionNoise"]
        self.ur3_rotation_noise = self.cfg["env"]["ur3RotationNoise"]
        self.ur3_dof_noise = self.cfg["env"]["ur3DofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_goal_scale": self.cfg["env"]["goalRewardScale"],
            "r_hand_scale": self.cfg["env"]["handRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "sep_dist_scale": self.cfg["env"]["sepRewardScale"],
            "r_contact_scale": self.cfg["env"]["contactRewardScale"],
            "contact_penalty_scale": self.cfg["env"]["contactPenaltyScale"],
            "act_penalty_scale": self.cfg["env"]["actionPenaltyScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs if osc: cube_pose (7) + eef_pose (7) + fingers (16) = 30
        # obs if joint: cube_pose (7) + joints (6) + fingers (16) = 29
        self.cfg["env"]["numObservations"] = 30 if self.control_type == "osc" else 253

        # Define the observations and actions of the thrower
        # initial state of the object to be thrown (pose, Xd, Rd)
        self.cfg["env"]["numThrowerActions"] = 6 if self.is_multi_agent else 0

        # actions if osc: delta EEF if OSC (6) + finger torques (16) = 22
        # actions if joint: joint torques (6) + finger torques (16) = 22
        self.cfg["env"]["numActions"] = 22 if self.control_type == "osc" else 44 + self.cfg["env"]["numThrowerActions"]

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.objects = AttrDict()               # will be dict filled with target object assets

        self.num_robot_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        # Objects
        self._init_object_state = None           # Initial state of cube for the current env
        self._target_obj_state = None
        self._object_idx_vec = None             # Contains each object id (e.g., _ball_id)
        self._object_size_vec = None            # Sizes of each object

        self._obj_ref_id = None

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._pos_control = None        # position actions
        self._effort_control = None     # Torque actions
        self._q = None                  # Joint position
        self._qd = None                 # Joint velocities

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

        self._l_contact_forces = None  # Contact forces in sim
        self._r_contact_forces = None  # Contact forces in sim
        self._l_arm_control = None  # Tensor buffer for controlling arm
        self._l_finger_control = None  # Tensor buffer for controlling gripper
        self._l_pos_control = None  # Position actions
        self._l_effort_control = None  # Torque actions

        self._r_arm_control = None  # Tensor buffer for controlling arm
        self._r_finger_control = None  # Tensor buffer for controlling gripper
        self._r_pos_control = None  # Position actions
        self._r_effort_control = None  # Torque actions

        self._object_shift_count = None     # object count per env for object category shift
        self._object_ids = None             # Tensor containing object ids

        self.debug_btn = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.is_multi_agent:
            # reward weight
            self.alpha_decay = self.cfg["env"]["multiAgent"].get("alpha_decay", False)
            self.alpha = self.cfg["env"]["multiAgent"].get("alpha", 0.9) \
                if not self.uniform_test else 1.0
            self.init_alpha = self.alpha
            self.final_alpha = self.cfg["env"]["multiAgent"].get("finalAlpha", 0.5)
            self.total_epochs = 10000
            print("Alpha: {}, init_alpha: {}, final_alpha: {}, total_epoch: {}"
                  .format(self.alpha, self.init_alpha, self.final_alpha, self.total_epochs))

            self.num_multi_agents = 2

            self.obs_space = spaces.Dict({"catch": spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf),
                                          "throw": spaces.Box(np.ones(self.num_obs+1) * -np.Inf, np.ones(self.num_obs+1) * np.Inf)})

            self.rew_bufs = torch.zeros(self.num_envs, self.num_multi_agents, device=self.device, dtype=torch.float)
            self.obs_bufs = {}
            self.obs_bufs.update({
                "obs0": torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float),
                "obs1": torch.zeros(self.num_envs, self.num_obs + 1, device=self.device, dtype=torch.float)
            })

            # remove buffers for single-agent
            del self.obs_buf
            del self.rew_buf

        self.num_a_actions = self.cfg["env"]["numThrowerActions"]

        # UR3 defaults
        self.default_left_ur3_pose = {"forward": [deg2rad(30.0), deg2rad(-90.0), deg2rad(110.0), deg2rad(-40.0), deg2rad(90.0), deg2rad(90.0)],
                                      "downward": [deg2rad(0.0), deg2rad(-120.0), deg2rad(-114.0), deg2rad(-36.0), deg2rad(80.0), deg2rad(0.0)]}
        self.default_right_ur3_pose = {"forward": [deg2rad(-30.0), deg2rad(-90.0), deg2rad(-110.0), deg2rad(-160.0), deg2rad(-90.0), deg2rad(-90.0)],
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
        self.l_cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self.left_allegro_ur3_effort_limits[:6].unsqueeze(0)
        self.r_cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
            self.control_type == "osc" else self.right_allegro_ur3_effort_limits[:6].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

        # Refresh tensors
        self._refresh()

        # viewer camera initial setting for result recording
        if self.viewer:
            from scipy.spatial.transform import Rotation as R
            # desired viewer camera pose
            cam_pos = gymapi.Vec3(22.9, 22.2, 2.01)
            cam_rot = np.array([0.75, -0.33, -0.23, 0.52])

            r = R.from_quat(cam_rot)
            forward_vector = np.array([0, 0, 1])
            cam_direction = r.apply(forward_vector)

            distance = 10.0
            cam_target = gymapi.Vec3(
                cam_pos.x + distance * cam_direction[0],
                cam_pos.y + distance * cam_direction[1],
                cam_pos.z + distance * cam_direction[2]
            )
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_assets(self):
        self.asset_files_dict = {
            "bottle": "mjcf/bottle_cap/bottle.urdf",
            "kettle": "mjcf/kettle/kettle.urdf",
            "cup": "mjcf/cup/cup.urdf",
            "bucket": "mjcf/bucket/bucket.urdf",
            "pen": "mjcf/pen/pen.urdf",
            "pot": "mjcf/pot/pot.urdf",
            "scissors": "mjcf/scissors/10495/scissors.urdf",
            "banana": "urdf/ycb/011_banana/011_banana.urdf",
            "meat_can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "mug": "urdf/ycb/025_mug/025_mug.urdf",
            "brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf",
        }

        """
        Allocate object dictionaries.
        Comment out the corresponding line if you want to exclude an object.
        """

        objects_to_create = [
            "gymball",
            "bowling",
            "cube",
            "board",

            "kettle",
            "bottle",
            "cup",
            "bucket",
            "pen",
            "pot",
            "scissors",

            "banana",
            "meat_can",
            "mug",
            "brick"
        ]

        for obj in objects_to_create:
            setattr(self.objects, obj, AttrDict())

        asset_creation_params = {
            "gymball": {"shape": "sphere",
                        "size": 0.6,
                        "color": (0.3, 0.6, 0.1),
                        "opts": {"density": 9, "disable_gravity": False}},
            "bowling": {"shape": "sphere",
                        "size": 0.215,
                        "color": (0.3, 0.1, 0.6),
                        "opts": {"density": 192, "disable_gravity": False}},
            "cube": {"shape": "box",
                     "size": 0.05,
                     "color": (0.6, 0.1, 0.0),
                     "opts": {}},
            "board": {"shape": "box",
                      "size": [0.9, 0.4, 0.05],  # width, length, height
                      "color": (0.1, 0.1, 0.6),
                      "opts": {}},

            "bottle": {"size": 0.2,
                       "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                                "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "kettle": {"size": 0.2,
                       "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                                "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "cup": {"size": 0.2,
                    "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                             "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "bucket": {"size": 0.2,
                       "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                                "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "pen": {"size": 0.2,
                    "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                             "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "pot": {"size": 0.4,
                    "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                             "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "scissors": {"size": 0.2,
                         "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                                  "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},

            "banana": {"size": 0.1,
                       "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                                "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "meat_can": {"size": 0.1,
                         "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                                  "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "mug": {"size": 0.1,
                    "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                             "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}},
            "brick": {"size": 0.1,
                      "opts": {"override_com": True, "override_inertia": True, "use_mesh_materials": True,
                               "mesh_normal_mode": gymapi.MeshNormalMode.COMPUTE_PER_VERTEX}}
        }

        for obj, params in asset_creation_params.items():
            if hasattr(self.objects, obj):
                self.objects[obj].size = params["size"]
                opts = gymapi.AssetOptions()
                for key, value in params["opts"].items():
                    setattr(opts, key, value)
                if obj in self.asset_files_dict:
                    self.objects[obj].opts = opts
                    self.objects[obj].asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_files_dict[obj], opts)
                else:
                    shape = params.get("shape", None)
                    if shape == "sphere":
                        self.objects[obj].asset = self.gym.create_sphere(self.sim, self.objects[obj].size * 0.5, opts)
                    elif shape == "box":
                        size = self.objects[obj].size
                        if isinstance(size, list) and len(size) == 3:
                            width, length, height = size
                            self.objects[obj].asset = self.gym.create_box(self.sim, width, length, height, opts)
                        else:
                            self.objects[obj].asset = self.gym.create_box(self.sim, *([size] * 3), opts)
                    elif shape == "capsule":
                        if isinstance(self.objects[obj].size, list) and len(self.objects[obj].size) == 2:
                            radius, length = self.objects[obj].size
                            self.objects[obj].asset = self.gym.create_capsule(self.sim, radius, length, opts)
                        else:
                            raise ValueError(f"Size for capsule must be a list of two elements: [radius, length].")
                    else:
                        raise ValueError(f"Unknown shape: {shape}")

                if "color" in params:
                    self.objects[obj].color = gymapi.Vec3(*params["color"])

        self.num_objs = len(get_assets(self.objects))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        left_allegro_ur3_asset_file = "urdf/ur_with_allegro_hand_description/urdf/ur3_allegro_left_hand.urdf"
        right_allegro_ur3_asset_file = "urdf/ur_with_allegro_hand_description/urdf/ur3_allegro_right_hand.urdf"

        if "asset" in self.cfg["env"]:
            self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", self.asset_root))
            left_allegro_ur3_asset_file = self.cfg["env"]["asset"].get("assetFileNameLeftAllegroUR3", left_allegro_ur3_asset_file)
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
        left_allegro_ur3_asset = self.gym.load_asset(self.sim, self.asset_root, left_allegro_ur3_asset_file, asset_options)
        right_allegro_ur3_asset = self.gym.load_asset(self.sim, self.asset_root, right_allegro_ur3_asset_file, asset_options)
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
        table_stand_py = 0.2 / 2.0     # (130mm x 130mm) 0 for single arm case
        l_table_stand_pos = [table_stand_px, table_stand_py, 1.0 + table_thickness / 2 + table_stand_height / 2]
        r_table_stand_pos = [table_stand_px, -table_stand_py, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[table_stand_breadth, table_stand_length, table_stand_height], table_opts)

        self._create_assets()

        print("left ur3 body cnt: ", self.gym.get_asset_rigid_body_count(left_allegro_ur3_asset))
        print("right ur3 body cnt: ", self.gym.get_asset_rigid_body_count(right_allegro_ur3_asset))

        self.num_allegro_ur3_bodies = sum([self.gym.get_asset_rigid_body_count(asset) for asset in self.allegro_ur3_assets])
        self.num_allegro_ur3_dofs = sum([self.gym.get_asset_dof_count(asset) for asset in self.allegro_ur3_assets])

        print("num ur3 bodies: ", self.num_allegro_ur3_bodies)
        print("num ur3 dofs: ", self.num_allegro_ur3_dofs)

        """
        DOF Properties
        Name        |   Data type       |   Description
        ==========================================================
        hasLimits       bool                Whether the DOF has limits or has unlimited motion.
        lower           float32             Lower limit.
        upper           float32             Upper limit.
        driveMode       gymapi.DofDriveMode DOF drive mode, see below.
        stiffness       float32             Drive stiffness.
        damping         float32             Drive damping.
        velocity        float32             Maximum velocity.
        effort          float32             Maximum effort (force or torque).
        friction        float32             DOF friction.
        armature        float32             DOF armature.
        """

        # set robot dof properties
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
            self._dof_lower_limits[idx] = to_torch(lower_limits, device=self.device)
            self._dof_upper_limits[idx] = to_torch(upper_limits, device=self.device)
            self._dof_effort_limits[idx] = to_torch(effort_limits, device=self.device)

        # Assign the converted tensors back to the original variables
        self.left_allegro_ur3_dof_lower_limits, self.right_allegro_ur3_dof_lower_limits = self._dof_lower_limits
        self.left_allegro_ur3_dof_upper_limits, self.right_allegro_ur3_dof_upper_limits = self._dof_upper_limits
        self.left_allegro_ur3_effort_limits, self.right_allegro_ur3_effort_limits = self._dof_effort_limits

        # set object dof properties
        for tag in self.objects:
            obj_dof_props = self.gym.get_asset_dof_properties(self.objects[tag].asset)
            for prop in obj_dof_props:
                # if not prop['hasLimits']:
                # prop['lower'] = 0.0
                # prop['upper'] = 0.0
                prop['driveMode'] = gymapi.DOF_MODE_POS
                prop['stiffness'] = 100
                prop['damping'] = 100
                prop['velocity'] = 5
                prop['effort'] = 1
                prop['friction'] = 100.0
                # prop['armature'] = 0.0
            self.objects[tag].dof_prop = obj_dof_props

        # Define start pose for franka
        left_ur3_start_pose = gymapi.Transform()
        right_ur3_start_pose = gymapi.Transform()
        left_ur3_start_pose.p = gymapi.Vec3(table_stand_px, table_stand_py, 1.0 + table_thickness / 2 + table_stand_height)
        right_ur3_start_pose.p = gymapi.Vec3(table_stand_px, -table_stand_py, 1.0 + table_thickness / 2 + table_stand_height)
        # ur3_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        _lq = quat_from_euler_xyz(roll=torch.tensor(deg2rad(0.0), device=self.device),
                                  pitch=torch.tensor(deg2rad(0.0), device=self.device),
                                  yaw=torch.tensor(deg2rad(0.0), device=self.device))
        _rq = quat_from_euler_xyz(roll=torch.tensor(deg2rad(0.0), device=self.device),
                                  pitch=torch.tensor(deg2rad(0.0), device=self.device),
                                  yaw=torch.tensor(deg2rad(180.0), device=self.device))
        left_ur3_start_pose.r = gymapi.Quat(_lq[0], _lq[1], _lq[2], _lq[3])
        right_ur3_start_pose.r = gymapi.Quat(_rq[0], _rq[1], _rq[2], _rq[3])

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for throwing
        self._throw_start_pos = np.array(table_pos) + np.array([table_length / 2, 0, table_thickness / 2])
        self._throw_start_pos[0] += 0.2
        self._throw_start_pos[2] += 0.3

        # Define start pose for table stand
        table_stand_left_start_pose = gymapi.Transform()
        table_stand_left_start_pose.p = gymapi.Vec3(*l_table_stand_pos)
        table_stand_left_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_stand_right_start_pose = gymapi.Transform()
        table_stand_right_start_pose.p = gymapi.Vec3(*r_table_stand_pos)
        table_stand_right_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        object_lounge = gymapi.Transform()
        object_lounge.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        object_lounge.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_ur3_bodies = sum([self.gym.get_asset_rigid_body_count(asset) for asset in self.allegro_ur3_assets + get_assets(self.objects)])
        num_ur3_shapes = sum([self.gym.get_asset_rigid_shape_count(asset) for asset in self.allegro_ur3_assets + get_assets(self.objects)])
        self.num_bodies = max_agg_bodies = num_ur3_bodies + 3   # 1 for table, 2 for table stands(x2), objects(cube, ball, etc)
        self.num_shapes = max_agg_shapes = num_ur3_shapes + 3   # 1 for table, 2 for table stands(x2), objects(cube, ball, etc)

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
            """
            * bitwise collision filter can be defined as following:
                left_ur3: 4 -->   100 (binary number)
                right_ur3: 8 --> 1000 (binary number)
                Both the left and right UR3 arms are not intersecting, so they can collide with each other
            """
            self._left_ur3_id = self.gym.create_actor(env_ptr, left_allegro_ur3_asset, left_ur3_start_pose, "left_ur3", i, 4, 0) # TODO, default: i,0,0
            self.gym.set_actor_dof_properties(env_ptr, self._left_ur3_id, self.bi_ur3_dof_props[0])
            # print("left arm index: ", self.gym.get_actor_index(env_ptr, left_ur3_actor, gymapi.DOMAIN_SIM))

            self._right_ur3_id = self.gym.create_actor(env_ptr, right_allegro_ur3_asset, right_ur3_start_pose, "right_ur3", i, 8, 0)
            self.gym.set_actor_dof_properties(env_ptr, self._right_ur3_id, self.bi_ur3_dof_props[1])
            # print("right arm index: ", self.gym.get_actor_index(env_ptr, right_ur3_actor, gymapi.DOMAIN_SIM))

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            l_table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_left_start_pose, "left_table_stand", i, 1, 0)
            r_table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_right_start_pose, "right_table_stand", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create object actors
            for tag in self.objects:
                _id = self.gym.create_actor(env_ptr, self.objects[tag].asset, object_lounge, tag, i, 2, 0)
                self.objects[tag].id = _id
                if hasattr(self.objects[tag], 'color'):
                    self.gym.set_rigid_body_color(env_ptr, _id, 0, gymapi.MESH_VISUAL, self.objects[tag].color)
                if hasattr(self.objects[tag], 'dof_prop'):
                    self.gym.set_actor_dof_properties(env_ptr, _id, self.objects[tag].dof_prop)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)

        # init object ids
        first_id = self.objects[list(self.objects.keys())[0]].id
        self._obj_ref_id = first_id

        id_size_dict = {b.id: b.size for b in self.objects.values()}
        self.obj_id_size_keys = to_torch(list(id_size_dict.keys()), device=self.device)

        id_size_list = []
        for value in id_size_dict.values():
            if isinstance(value, list):
                id_size_list.append((sum(value)/len(value)) * 0.5)  # ugly mean value
                # for item in value:
                #     id_size_list.append(item)
            else:
                id_size_list.append(value)
        self.obj_id_size_values = to_torch(id_size_list, device=self.device)
        # self.obj_id_size_values = to_torch(list(id_size_dict.values()), device=self.device)

        self.allegro_ur3_body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, self._left_ur3_id)
        """
        Rigid body flags:
            gymapi.RIGID_BODY_NONE= 0
            gymapi.RIGID_BODY_DISABLE_GRAVITY= 1
            gymapi.RIGID_BODY_DISABLE_SIMULATION(PhysX only)= 2
        """

        # if you want to show the items of the dict, comment out the following codes
        # sorted_dict = dict(sorted(self.allegro_ur3_body_dict.items(), key=lambda item: item[1]))
        # for key, value in sorted_dict.items():
        #     print(f"{key}: {value}")
        """
        * allegro_ur3_body_dict items
        --------------------------------
            world: 0    # world
        --------------------------------
            base_link: 1    # UR3 starts
            base: 2
            base_link_inertia: 3
            shoulder_link: 4
            upper_arm_link: 5
            forearm_link: 6
            wrist_1_link: 7
            wrist_2_link: 8
            wrist_3_link: 9
            flange: 10
            tool0: 11
        --------------------------------
            adaptor: 12     # adaptor
        --------------------------------
            palm_link: 13   # allegro hand starts
            link_0: 14
            link_1: 15
            link_2: 16
            link_3: 17
            link_3_tip: 18
            link_12: 19
            link_13: 20
            link_14: 21
            link_15: 22
            link_15_tip: 23
            link_4: 24
            link_5: 25
            link_6: 26
            link_7: 27
            link_7_tip: 28
            link_8: 29
            link_9: 30
            link_10: 31
            link_11: 32
            link_11_tip: 33
            allegro_grip_site: 34
            ft_frame: 35
        --------------------------------
        """

        # Setup init state buffer
        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_object_state[:, 7] = 1.0     # unit quaternion

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        left_ur3_handle = 0
        right_ur3_handle = 1
        self.handles = {
            # UR3
            "hand_left": self.gym.find_actor_rigid_body_handle(env_ptr, left_ur3_handle, "tool0"),
            "grip_site_left": self.gym.find_actor_rigid_body_handle(env_ptr, left_ur3_handle, "allegro_grip_site"),
            "hand_right": self.gym.find_actor_rigid_body_handle(env_ptr, right_ur3_handle, "tool0"),
            "grip_site_right": self.gym.find_actor_rigid_body_handle(env_ptr, right_ur3_handle, "allegro_grip_site"),
            "base_left": self.gym.find_actor_rigid_body_handle(env_ptr, left_ur3_handle, "base_link"),
            "base_right": self.gym.find_actor_rigid_body_handle(env_ptr, right_ur3_handle, "base_link"),
        }

        bodies_to_detect_contacts = ["base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
                                     "wrist_1_link", "wrist_2_link", "wrist_3_link"]
        # TODO, hand-side contacts should be encouraged..
        fingers_to_detect_contacts = ["link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"]     # "palm_link"
        self.ids_for_arm_contact = get_indices_from_dict(self.allegro_ur3_body_dict, bodies_to_detect_contacts)
        self.ids_for_hand_contact = get_indices_from_dict(self.allegro_ur3_body_dict, fingers_to_detect_contacts)

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_obj_dofs = sum([self.gym.get_asset_dof_count(self.objects[obj].asset) for obj in self.objects])
        self.num_robot_dofs = self.num_dofs - self.num_obj_dofs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._contact_forces = gymtorch.wrap_tensor(_contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        num_bodies_per_robot = self.num_allegro_ur3_bodies // len(self.allegro_ur3_assets)
        self._l_contact_forces = self._contact_forces[:, :num_bodies_per_robot, :]
        self._r_contact_forces = self._contact_forces[:, num_bodies_per_robot:, :]
        self._q = self._dof_state[:, :self.cfg["env"]["numActions"], 0]
        self._qd = self._dof_state[:, :self.cfg["env"]["numActions"], 1]
        self._obj_q = self._dof_state[:, self.cfg["env"]["numActions"]:, 0]     # object dofs
        self._obj_qd = self._dof_state[:, self.cfg["env"]["numActions"]:, 1]
        dof_per_arm = self.num_allegro_ur3_dofs // len(self.allegro_ur3_assets)
        self._l_q = self._q[:, :dof_per_arm]
        self._l_qd = self._qd[:, :dof_per_arm]
        self._r_q = self._q[:, dof_per_arm:]
        self._r_qd = self._qd[:, dof_per_arm:]
        self._l_eef_state = self._rigid_body_state[:, self.handles["grip_site_left"], :]
        self._r_eef_state = self._rigid_body_state[:, self.handles["grip_site_right"], :]
        self._l_base_state = self._rigid_body_state[:, self.handles["base_left"], :]
        self._r_base_state = self._rigid_body_state[:, self.handles["base_right"], :]
        _l_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "left_ur3")
        l_jacobian = gymtorch.wrap_tensor(_l_jacobian)
        _r_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "right_ur3")
        r_jacobian = gymtorch.wrap_tensor(_r_jacobian)

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

        first_id = self.objects[list(self.objects.keys())[0]].id
        last_id = self.objects[list(self.objects.keys())[-1]].id + 1
        self._target_obj_state = self._root_state[:, first_id:last_id, :]
        self._object_idx_vec = torch.ones(self.num_envs, device=self.device, dtype=torch.long) * -1
        self._object_size_vec = torch.zeros(self.num_envs, device=self.device)

        # Initialize states, always bbox size
        self.states.update({
            "object_size_vec": torch.ones_like(self._l_eef_state[:, 0]),    # Initial object size as 1.0
        })

        # Initialize robot actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        self._l_pos_control = self._pos_control[:, :dof_per_arm]
        self._l_effort_control = self._effort_control[:, :dof_per_arm]
        self._r_pos_control = self._pos_control[:, dof_per_arm:dof_per_arm*2]
        self._r_effort_control = self._effort_control[:, dof_per_arm:dof_per_arm*2]

        # Initialize object actions
        self._obj_pos_control = self._pos_control[:, -self.num_obj_dofs:] if self.num_obj_dofs != 0 else self._pos_control[:, :0]
        self._obj_effort_control = torch.zeros_like(self._obj_pos_control)

        # Initialize control
        self._l_arm_control = self._l_effort_control[:, :6]
        self._l_finger_control = self._l_effort_control[:, 6:]
        self._r_arm_control = self._r_effort_control[:, :6]
        self._r_finger_control = self._r_effort_control[:, 6:]

        # Initialize indices
        # left_ur3 + right_ur3 + table + table_stand x 2 + num_objects
        num_actors = len(self.allegro_ur3_assets) + 3 + self.num_objs
        self._global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)
        self._object_shift_count = torch.ones(self.num_envs, dtype=torch.uint8, device=self.device) * -1

        self.setup_controlled_experiment()

    def save_or_load_data(self, file_path, data, overwrite=False):
        if os.path.exists(file_path):
            if overwrite:
                np.save(file_path, data)
                print(f"[CE] Data has been overwritten and saved to {file_path}.")
            else:
                loaded_data = np.load(file_path)
                print(f"[CE] Data has been loaded from {file_path}.")
                return loaded_data
        else:
            np.save(file_path, data)
            print(f"[CE] Data has been saved to {file_path}.")
        return data

    def setup_controlled_experiment(self):
        """
            This setup is a code implementation used to arbitrarily control objects
            for the per-object experiments in the paper and derive the corresponding results.
            (See Figure 9 in our paper)
        """
        if self.controlled_experiment:
            self.ce_count = 0
            self.ce_env = self.num_envs - 1

            # controlled experiment indices
            self.ce_indices = np.repeat(np.arange(len(self.objects)), self.num_controlled_experiment_per_object)
            np.random.shuffle(self.ce_indices)

            ce_obj_state_list = []
            for i in range(len(self.ce_indices)):
                # controlled experiment uniform random pose & velocity
                self._reset_uniform_random_object_state(obj='A', env_ids=[self.ce_env])
                ce_obj_state = self._init_object_state[self.ce_env].clone()
                ce_obj_state_list.append(ce_obj_state)
            self.ce_states = torch.stack(ce_obj_state_list)

            # check existing evaluation files
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path_to_indices = os.path.join(current_dir, '..', 'evaluation', 'controlled_exp_object_indices.npy')
            path_to_obj_state = os.path.join(current_dir, '..', 'evaluation', 'controlled_exp_object_states.npy')

            self.ce_indices = self.save_or_load_data(file_path=path_to_indices, data=self.ce_indices, overwrite=False)
            self.ce_states = self.save_or_load_data(file_path=path_to_obj_state, data=self.ce_states.cpu().numpy(), overwrite=False)
            self.ce_states = torch.from_numpy(self.ce_states)
            print("[CE] Indices for Controlled Experiment: ", self.ce_indices)
            print("[CE] States for Controlled Experiment: ", self.ce_states)

    # Uncomment to retrieve the current viewer camera transformation
    # def render(self):
    #     cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
    #     cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
    #     cam_rot = np.array([cam_trans.r.x, cam_trans.r.y, cam_trans.r.z, cam_trans.r.w])
    #     print("cam_pos: ", cam_pos)
    #     print("cam_rot: ", cam_rot)
    #
    #     super().render()

    def _update_states(self):

        # print("left arm contact ", self._l_contact_forces[-1, self.ids_for_contact])
        # print("right arm contact ", self._r_contact_forces[-1, self.ids_for_contact])

        l_arm_contact_n_mag = F.normalize(self._l_contact_forces[:, self.ids_for_arm_contact], p=2, dim=-1)
        r_arm_contact_n_mag = F.normalize(self._r_contact_forces[:, self.ids_for_arm_contact], p=2, dim=-1)

        l_hand_contact_n_mag = F.normalize(self._l_contact_forces[:, self.ids_for_hand_contact], p=2, dim=-1)
        r_hand_contact_n_mag = F.normalize(self._r_contact_forces[:, self.ids_for_hand_contact], p=2, dim=-1)

        obj_goal_offset = torch.tensor([0.3, 0.0, 0.4], device=self.device)

        self.states.update({
            # Left Allegro UR3
            "l_q": self._l_q[:, :6],
            "l_q_finger": self._l_q[:, 6:],
            "l_eef_pos": self._l_eef_state[:, :3],
            "l_eef_quat": self._l_eef_state[:, 3:7],
            "l_eef_vel": self._l_eef_state[:, 7:],
            "left_arm_contact_n_mag": l_arm_contact_n_mag,
            "left_hand_contact_n_mag": l_hand_contact_n_mag,
            # Right Allegro UR3
            "r_q": self._r_q[:, :6],
            "r_q_finger": self._r_q[:, 6:],
            "r_eef_pos": self._r_eef_state[:, :3],
            "r_eef_quat": self._r_eef_state[:, 3:7],
            "r_eef_vel": self._r_eef_state[:, 7:],
            "right_arm_contact_n_mag": r_arm_contact_n_mag,
            "right_hand_contact_n_mag": r_hand_contact_n_mag,
            # Bimanual states
            "left_right_relative_hand_pos": self._l_eef_state[:, :3] - self._r_eef_state[:, :3],
            "object_goal_pos": (self._l_base_state[:, :3] + self._r_base_state[:, :3]) * 0.5 + obj_goal_offset,
            # Target Object
            "object_idx_vec": self._object_idx_vec - self._obj_ref_id,  # making id starts from 0
            "object_size_vec": self._object_size_vec,
            "object_pos": self._target_obj_state[:, :, :3],
            "object_quat": self._target_obj_state[:, :, 3:7],
            "object_pos_vel": self._target_obj_state[:, :, 7:10],
            "object_rot_vel": self._target_obj_state[:, :, 10:],
            "object_pos_relative_left_hand": self._target_obj_state[:, :, :3] - self._l_eef_state[:, :3].unsqueeze(1),
            "object_pos_relative_right_hand": self._target_obj_state[:, :, :3] - self._r_eef_state[:, :3].unsqueeze(1),
            # Reset Buff
            "reset_buf": self.reset_buf,
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def get_all_env_ids(self):
        return torch.ones(self.num_envs, device=self.device, dtype=torch.long)

    def decay_alpha(self, curr_epoch):
        if self.alpha_decay:
            self.alpha = self.init_alpha - (self.init_alpha - self.final_alpha) * (min(curr_epoch, self.total_epochs) / self.total_epochs)
            if curr_epoch % 100 == 0:
                print("Current alpha: {} / {}".format(self.alpha, curr_epoch))

    def compute_reward(self, actions):
        rew_catch_buf, self.reset_buf[:] = compute_catch_reward(
            self.reset_buf, self.progress_buf, self.actions, self._l_qd, self._r_qd, self.states, self.reward_settings, self.max_episode_length
        )

        if self.num_multi_agents > 1:
            rew_throw_buf, reset_buf = compute_throw_reward(
                self.reset_buf, self.actions, self.progress_buf, self.states, self.reward_settings, self.max_episode_length)

            self.reset_buf = self.reset_buf | reset_buf
            mean_rew = self.alpha * rew_catch_buf + (1 - self.alpha) * rew_throw_buf
            self.rew_bufs[:] = mean_rew.unsqueeze(-1).repeat(1, self.rew_bufs.shape[1])
        else:
            self.rew_buf = 1.0 * rew_catch_buf

    def compute_observations(self):
        self._refresh()
        if self.control_type == "osc":
            obs = ["cube_quat", "cube_pos", "l_eef_pos", "l_eef_quat", "r_eef_pos", "r_eef_quat"]
        else:
            # 6 + 6 + 16 + 16 + num_obj x (4 + 3 + 3 + 3)
            obs = ["l_q", "r_q", "l_q_finger", "r_q_finger", "object_pos", "object_quat",
                   "object_pos_relative_left_hand", "object_pos_relative_right_hand"]
        obs += ["l_eef_pos"] + ["l_eef_quat"] + ["r_eef_pos"] + ["r_eef_quat"]

        if self.num_multi_agents > 1:
            for agent_id in range(self.num_multi_agents):
                if agent_id == 1:
                    obs += ["reset_buf"]
                self.obs_bufs["obs" + str(agent_id)] = torch.cat([self.states[ob].reshape(self.num_envs, -1) for ob in obs], dim=-1)
                # self.obs_bufs[:, agent_id] = torch.cat([self.states[ob].reshape(self.num_envs, -1) for ob in obs], dim=-1)
            return self.obs_bufs
        else:
            self.obs_buf = torch.cat([self.states[ob].reshape(self.num_envs, -1) for ob in obs], dim=-1)
            return self.obs_buf

        # # TODO, should be removed later..
        # if torch.any(torch.isnan(self.obs_buf)):
        #     nan_indices = torch.where(torch.isnan(self.obs_buf))
        #     print("prev_obs_buf: ", self.prev_obs_buf[nan_indices])
        #     print("obs_buf: ", self.obs_buf[nan_indices])
        #     raise ValueError(f"obs_buf tensor contains NaN values at indices: {nan_indices}")
        # self.prev_obs_buf = self.obs_buf

        # maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        # return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset all objects to their origin in environments marked for reset
        self._target_obj_state[env_ids, :, :] = zero_state(self._target_obj_state[env_ids],
                                                           self._object_size_vec[env_ids], self.device).clone()

        # Get random indices for the next object
        first_id = self.objects[list(self.objects.keys())[0]].id
        last_id = self.objects[list(self.objects.keys())[-1]].id + 1
        rand_obj_ids = torch.randint(low=first_id, high=last_id, size=(len(env_ids),), device=self.device)

        idx = None
        if self.controlled_experiment and self.uniform_test and (self.ce_env in env_ids):     # use last environment for controlled exp
            print("last env is reset!!!")
            idx = torch.where(env_ids == self.ce_env)[0].item()
            ce_idx = self.ce_indices[self.ce_count]
            rand_obj_ids[idx] = ce_idx + self._obj_ref_id
            self._init_object_state[env_ids[idx]] = self.ce_states[self.ce_count]
            print("curr pose: ", self.ce_states[self.ce_count])
            self.ce_count += 1

        self._object_idx_vec[env_ids] = rand_obj_ids.clone()
        indices = torch.searchsorted(self.obj_id_size_keys, rand_obj_ids)
        self._object_size_vec[env_ids] = self.obj_id_size_values[indices].clone()

        if self.is_multi_agent:
            if self.actions is None or self.uniform_test:
                self._reset_uniform_random_object_state(obj='A', env_ids=env_ids[:idx])
            else:
                self._reset_adversarial_random_object_state(obj='A', env_ids=env_ids)
        else:
            self._reset_uniform_random_object_state(obj='A', env_ids=env_ids[:idx])

        _rand_obj_ids = rand_obj_ids - self._obj_ref_id
        self._target_obj_state[env_ids, _rand_obj_ids] = self._init_object_state[env_ids].clone()

        # Reset agent
        reset_noise = torch.rand((len(env_ids), self.num_allegro_ur3_dofs), device=self.device)
        ur3_default_dof_pos = torch.concat((self.left_ur3_default_dof_pos, self.right_ur3_default_dof_pos), dim=-1)
        pos = tensor_clamp(
            ur3_default_dof_pos.unsqueeze(0) +
            self.ur3_dof_noise * 2.0 * (reset_noise - 0.5),
            torch.cat(self._dof_lower_limits), torch.cat(self._dof_upper_limits))
        obj_pos = torch.zeros_like(self._obj_pos_control[env_ids])
        pos_all = torch.cat((pos, obj_pos), dim=-1)

        # # Overwrite gripper init pos (no noise since these are always position controlled)
        # pos[:, 6:22] = ur3_default_dof_pos[6:22]
        # pos[:, 22+6:] = ur3_default_dof_pos[22+6:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Reset the objects' dofs
        self._obj_q[env_ids, :] = to_torch([0.0 for _ in range(self.num_obj_dofs)], device=self.device)
        self._obj_qd[env_ids, :] = to_torch([0.0 for _ in range(self.num_obj_dofs)], device=self.device)
        # print("obj_q: ", self._obj_q[env_ids, :])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids] = pos_all
        self._effort_control[env_ids] = torch.zeros_like(pos_all)

        # Deploy updates
        # TODO, indexing, 0 --> left arm, 1 --> right arm
        multi_env_ids_int32 = self._global_indices[env_ids, 0:2].flatten()
        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._pos_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update object states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -self.num_objs:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_uniform_random_object_state(self, obj, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_obj_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if obj.lower() == 'a':
            this_object_state_all = self._init_object_state
            obj_size = self._object_size_vec
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {obj}")

        # Sampling is "centered" around middle of table
        # centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
        biased_obj_xy_state = torch.tensor(self._throw_start_pos[:2], device=self.device, dtype=torch.float)

        # Set z value, which is fixed height
        sampled_obj_state[:, 2] = self._table_surface_pos[2] + obj_size.squeeze(-1)[env_ids]

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_obj_state[:, 6] = 1.0
        sampled_obj_state[:, :2] = biased_obj_xy_state.unsqueeze(0) + torch.cat([
            0.5 * self.start_position_noise * self.noise_scale * (torch.rand(num_resets, 1, device=self.device) - 0.5),
            2.0 * self.start_position_noise * self.noise_scale * (torch.rand(num_resets, 1, device=self.device) - 0.5)
        ], dim=-1)

        sampled_obj_state[:, 2] = torch.tensor([1.5], device=self.device) + \
                                  2.0 * self.start_position_noise * self.noise_scale * (torch.rand(num_resets, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * self.noise_scale * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_obj_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_obj_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_object_state_all[env_ids, :] = sampled_obj_state

        # linear/angular velocity randomization, m/s, radian/s
        this_object_state_all[env_ids, 7:10] = 1.0 * self.noise_scale * torch.tensor([5.0, 2.0, 4.0], device=self.device) * (torch.rand(num_resets, 3, device=self.device) - 0.5)
        this_object_state_all[env_ids, 7] = -torch.abs(this_object_state_all[env_ids, 7]) - 1.0
        this_object_state_all[env_ids, 9] = torch.abs(this_object_state_all[env_ids, 9]) + 1.0
        this_object_state_all[env_ids, 10:] = 10.0 * self.noise_scale * torch.tensor([1.0, 1.0, 1.0], device=self.device) * (torch.rand(num_resets, 3, device=self.device) - 0.5)

    def _reset_adversarial_random_object_state(self, obj, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_obj_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if obj.lower() == 'a':
            this_object_state_all = self._init_object_state
            obj_size = self._object_size_vec
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {obj}")

        # Sampling is "centered" around middle of table
        # centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
        biased_obj_xy_state = torch.tensor(self._throw_start_pos[:2], device=self.device, dtype=torch.float)

        # Set z value, which is fixed height
        sampled_obj_state[:, 2] = self._table_surface_pos[2] + obj_size.squeeze(-1)[env_ids]

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_obj_state[:, 6] = 1.0
        sampled_obj_state[:, :2] = biased_obj_xy_state.unsqueeze(0) + torch.cat([
            0.5 * self.start_position_noise * (torch.rand(num_resets, 1, device=self.device) - 0.5),
            2.0 * self.start_position_noise * (torch.rand(num_resets, 1, device=self.device) - 0.5)
        ], dim=-1)

        sampled_obj_state[:, 2] = torch.tensor([1.5], device=self.device) + \
                                  2.0 * self.start_position_noise * (torch.rand(num_resets, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_obj_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_obj_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_object_state_all[env_ids, :] = sampled_obj_state

        # # linear/angular velocity randomization, m/s, radian/s
        this_object_state_all[env_ids, 7:10] = 1.0 * torch.tensor([5.0, 0.0, 4.0], device=self.device) * (torch.rand(num_resets, 3, device=self.device) - 0.5)
        this_object_state_all[env_ids, 7] = -torch.abs(this_object_state_all[env_ids, 7]) - 1.0
        this_object_state_all[env_ids, 9] = torch.abs(this_object_state_all[env_ids, 9]) + 1.0
        this_object_state_all[env_ids, 10:] = 1.0 * torch.tensor([1.0, 1.0, 1.0], device=self.device) * (torch.rand(num_resets, 3, device=self.device) - 0.5)

        # Thrower actions
        # obj_pose = self.actions[env_ids, 44:51]
        obj_lin_vel = self.actions[env_ids, 44:47]
        obj_rot_vel = self.actions[env_ids, 47:50]

        # this_object_state_all[env_ids, 0:7] = obj_pose
        # lin_scale = 27.78  # Max speed: 27.78 m/s, equivalent to 100 km/h (approximate speed of a fastball thrown by a pitcher)
        # rot_scale = 209.44  # Max 2000 RPM, representing the typical spin rate of a curveball thrown by a pitcher
        lin_scale = 10.0
        rot_scale = 100.0
        this_object_state_all[env_ids, 7:10] += obj_lin_vel * lin_scale * 0.1
        this_object_state_all[env_ids, 10:13] += obj_rot_vel * rot_scale * 0.1


    def _compute_osc_torques(self, dpose):
        """
            * NOT used in this project..
        """
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
        # cv2.imshow("step by press", np.zeros((100, 100, 1)))
        # key = cv2.waitKey(int(not self.debug_btn))
        # if key == ord('d'): self.debug_btn = not self.debug_btn

        """
        Action Composition
            Full Action Dim(57)
            * Catcher (44)
                - left arm [0:6]
                - left fingers [6:22]
                - right arm [22:28]
                - right fingers [28:44]
            * Thrower (13)
                - object pose [44:51]
                - object lin vel [51:54]
                - object rot vel [54:57]
        """

        mask = torch.zeros_like(actions)
        # mask[:, 0] = 1.0
        self.actions = actions.clone().to(self.device)

        # Catcher actions
        # Split arm and finger command
        l_u_arm, l_u_finger = self.actions[:, :6], self.actions[:, 6:22]
        r_u_arm, r_u_finger = self.actions[:, 22:22+6], self.actions[:, 22+6:22+6+16]

        # Control arm (scale value first)
        l_u_arm = l_u_arm * self.l_cmd_limit / self.action_scale
        r_u_arm = r_u_arm * self.r_cmd_limit / self.action_scale
        # if self.control_type == "osc":  # TODO, NOT used in this project..
        #     u_arm = self._compute_osc_torques(dpose=u_arm)
        self._l_arm_control[:, :] = l_u_arm
        self._l_finger_control[:, :] = l_u_finger
        self._r_arm_control[:, :] = r_u_arm
        self._r_finger_control[:, :] = r_u_finger

        # self._obj_q[:, :] = to_torch([0.0 for _ in range(self.num_obj_dofs)], device=self.device)
        # self._obj_qd[:, :] = to_torch([0.0 for _ in range(self.num_obj_dofs)], device=self.device)
        # self._obj_pos_control = to_torch([0.0 for _ in range(self.num_obj_dofs)], device=self.device)

        # Deploy actions
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        if self.controlled_experiment and not self.sim_start:
            user_input = input('go??')
            if user_input.lower() == 'y':
                self.sim_start = True
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
            l_eef_pos = self.states["l_eef_pos"]
            l_eef_rot = self.states["l_eef_quat"]
            r_eef_pos = self.states["r_eef_pos"]
            r_eef_rot = self.states["r_eef_quat"]
            obj_goal_pos = self.states["object_goal_pos"]
            obj_goal_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

            cube_pos = self.states["cube_pos"]
            cube_rot = self.states["cube_quat"]

            pos_list = [obj_goal_pos]
            rot_list = [obj_goal_rot]

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
def compute_catch_reward(
    reset_buf, progress_buf, actions, _l_qd, _r_qd, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    object_idx = states["object_idx_vec"].type(torch.long)
    object_size = states["object_size_vec"]

    # distance from hand to the target object
    ar_idx = torch.arange(len(object_idx), dtype=torch.long)
    # _ld = torch.where(object_idx.unsqueeze(-1) == 0, states["ball_pos_relative_left_hand"], states["cube_pos_relative_left_hand"])
    # _rd = torch.where(object_idx.unsqueeze(-1) == 0, states["ball_pos_relative_right_hand"], states["cube_pos_relative_right_hand"])
    _ld = states["object_pos_relative_left_hand"][ar_idx, object_idx, :]
    _rd = states["object_pos_relative_right_hand"][ar_idx, object_idx, :]
    ld = torch.norm(_ld, dim=-1)
    rd = torch.norm(_rd, dim=-1)
    # d_lf = torch.norm(states["cube_pos"] - states["eef_lf_pos"], dim=-1)
    # d_rf = torch.norm(states["cube_pos"] - states["eef_rf_pos"], dim=-1)
    l_dist_reward = torch.exp(-10.0 * ld)
    r_dist_reward = torch.exp(-10.0 * rd)
    # l_dist_reward = 1 - torch.tanh(10.0 * ld)
    # r_dist_reward = 1 - torch.tanh(10.0 * rd)
    # l_dist_reward += torch.where(ld < 0.01, 1.0, 0.0)  # reward bonus
    # r_dist_reward += torch.where(rd < 0.01, 1.0, 0.0)  # reward bonus
    hand_dist_reward = 0.5 * l_dist_reward + 0.5 * r_dist_reward
    # dist_reward = torch.max(l_dist_reward, r_dist_reward)

    # object goal distance to target point
    # _gd = states["object_goal_pos"] - torch.where(object_idx.unsqueeze(-1) == 0, states["ball_pos"], states["cube_pos"])
    _gd = states["object_goal_pos"] - states["object_pos"][ar_idx, object_idx, :]
    gd = torch.norm(_gd, dim=-1)
    goal_dist_reward = torch.exp(-10.0 * gd)
    # goal_dist_reward = 1 - torch.tanh(10.0 * gd)
    # goal_dist_reward += torch.where(gd < 0.01, 1.0, 0.0)    # reward bonus

    # distance between hands to avoid collision
    max_sep_dist = 0.1
    sep_d = torch.norm(states["left_right_relative_hand_pos"], dim=-1)
    sep_d = torch.clamp(sep_d, 0.0, 0.2)
    sep_dist_reward = 1 - torch.tanh(-10.0 * sep_d)

    # reward for lifting cube
    temp_pos = states["object_pos"][ar_idx, object_idx, 2]
    object_height = temp_pos - reward_settings["table_height"]
    object_lifted = (object_height - object_size) > object_size * 0.5 * 1.6    # cube: 0.04,
    lift_reward = object_lifted

    l_arm_contact_n_mag_mean = torch.mean(torch.norm(states["left_arm_contact_n_mag"], dim=-1), dim=-1)
    r_arm_contact_n_mag_mean = torch.mean(torch.norm(states["right_arm_contact_n_mag"], dim=-1), dim=-1)
    arm_contact_n_mag_mean = 0.5 * l_arm_contact_n_mag_mean + 0.5 * r_arm_contact_n_mag_mean

    l_hand_contact_n_mag_mean = torch.mean(torch.norm(states["left_hand_contact_n_mag"], dim=-1), dim=-1)
    r_hand_contact_n_mag_mean = torch.mean(torch.norm(states["right_hand_contact_n_mag"], dim=-1), dim=-1)
    hand_contact_n_mag_mean = 0.5 * l_hand_contact_n_mag_mean + 0.5 * r_hand_contact_n_mag_mean

    ur_actions_penalty = (torch.sum(torch.abs(_l_qd[..., 0:6]), dim=-1) + torch.sum(torch.abs(_r_qd[..., 0:6]), dim=-1))
    allegro_actions_penalty = (torch.sum(torch.abs(_l_qd[..., 7:]), dim=-1) + torch.sum(torch.abs(_r_qd[..., 7:]), dim=-1))
    action_penalty = 1.0 * ur_actions_penalty + 1.0 * allegro_actions_penalty

    rewards = (reward_settings["r_hand_scale"] * hand_dist_reward
               + reward_settings["r_goal_scale"] * goal_dist_reward
               + reward_settings["r_lift_scale"] * lift_reward
               + reward_settings["sep_dist_scale"] * sep_dist_reward
               + reward_settings["r_contact_scale"] * hand_contact_n_mag_mean
               - reward_settings["contact_penalty_scale"] * arm_contact_n_mag_mean
               - reward_settings["act_penalty_scale"] * action_penalty)

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (object_height < object_size * 0.5 + 1e-2),
                            torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


# @torch.jit.script
def compute_throw_reward(
    reset_buf, actions, progress_buf, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    progress_mask = torch.where(progress_buf==0, 1, -1)

    _act_obj_lin_vel = actions[:, 44:47]
    _act_obj_rot_vel = actions[:, 47:50]

    act_obj_lin_vel = torch.norm(_act_obj_lin_vel, dim=-1) * progress_mask
    act_obj_rot_vel = torch.norm(_act_obj_rot_vel, dim=-1) * progress_mask

    object_idx = states["object_idx_vec"].type(torch.long)
    object_size = states["object_size_vec"]

    ar_idx = torch.arange(len(object_idx), dtype=torch.long)

    # object height for reset_buf
    temp_pos = states["object_pos"][ar_idx, object_idx, 2]
    object_height = temp_pos - reward_settings["table_height"]

    # throw reward calc.
    obj_lin_vel = states["object_pos_vel"][ar_idx, object_idx, :]
    obj_rot_vel = states["object_rot_vel"][ar_idx, object_idx, :]
    negative_lin_vel_mask = torch.where(obj_lin_vel[:, 0] > 0, 0.0, 1.0)
    obj_pos_vel_norm = torch.norm(obj_lin_vel, dim=-1) * negative_lin_vel_mask
    obj_rot_vel_norm = torch.norm(obj_rot_vel, dim=-1)

    lin_scale = 27.78  # Max speed: 27.78 m/s, equivalent to 100 km/h (approximate speed of a fastball thrown by a pitcher)
    rot_scale = 209.44  # Max 2000 RPM, representing the typical spin rate of a curveball thrown by a pitcher
    obj_pos_vel_reward = 1.0 - torch.exp(-5.0 * obj_pos_vel_norm)
    obj_rot_vel_reward = 1.0 - torch.exp(-5.0 * obj_rot_vel_norm)

    obj_throw_reward = 0.5 * obj_pos_vel_reward + 0.5 * obj_rot_vel_reward

    act_lin_vel_reward = torch.tanh(1.0 * act_obj_lin_vel)
    act_rot_vel_reward = torch.tanh(1.0 * act_obj_rot_vel)
    act_reward = 0.5 * act_lin_vel_reward + 0.5 * act_rot_vel_reward

    # reward_settings["r_hand_scale"]
    rewards = (0.8 * obj_throw_reward + 0.2 * act_reward)

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (object_height > 3.0),
                            torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
