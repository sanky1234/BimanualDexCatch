import time 

from ..utils.utils import AttrDict 

import numpy as np
import os 

from isaacgym import gymtorch 
from isaacgym import gymapi 

import torch 
import torch.nn.functional as F 

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_from_euler_xyz, quat_apply 
from isaacgymenvs.tasks.base.vec_task import VecTaskSimple
from isaacgymenvs.tasks.base.multi_vec_task import MultiVecTask 
from isaacgymenvs.tasks.utils.general_utils import deg2rad 

from gym import spaces 

import pybullet as p
import pybullet_data

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


class BimanualDexCatchSpotCEM(VecTaskSimple):
     
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg 

        self.sim_start = False 
        
        # Time to test stability
        self.stability_time = self.cfg["env"]["stability_time"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]


        self.reward_scales = {
            "success": self.cfg["env"]["success_reward_scale"],
            "catch": self.cfg["env"]["catch_reward_scale"],
            "catch_dist": self.cfg["env"]["catch_dist_reward_scale"],
            "contact": self.cfg["env"]["contact_reward_scale"],
            "collision": self.cfg["env"]["collision_reward_scale"],
            "manipublity": self.cfg["env"]["manipulability_reward_scale"],
        }

        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.objects = AttrDict()

        self.num_robot_dofs = 2*7 + 2*6 # 2 arms, 2 grippers. 7 dof for arm, 6 dof for gripper

        self.input_control_names = self.cfg["input_control_names"]


        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self._refresh()

        ####################### pybullet init  for contact detection #######################
        self.physicsClient = p.connect(p.GUI)
        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For plane.urdf and other data
        p.setGravity(0, 0, -9.81)

        self.planeId = p.loadURDF("plane.urdf")
        self.pybullet_robot = p.loadURDF("/home/sankalp/ws_spot_catching/src/BimanualDexCatch/assets/urdf/spot_description/spot_7dof_psyonic_no_base.urdf", [0.0, 0.0, 0.7], [0,0,0,1], useFixedBase=True)
        self.pybullet_football = p.loadURDF("/home/sankalp/ws_spot_catching/src/BimanualDexCatch/assets/urdf/football.urdf", [0.2, 0.0, 0.3], [0,0,0,1], useFixedBase=False)
        self.pybullet_num_joints = p.getNumJoints(self.pybullet_robot)

        self.pybullet_joint_idx_mapping = -1 * np.ones(len(self.input_control_names), dtype=int)

        for i,name in enumerate(self.input_control_names):
            for j in range(self.pybullet_num_joints):
                joint_name = p.getJointInfo(self.pybullet_robot, j)[1].decode("utf-8")
                if name == joint_name:
                    self.pybullet_joint_idx_mapping[i] = j
                    break

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0.0 
        self.sim_params.gravity.y = 0.0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params) 
        self._create_ground_plane() 
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams() 
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # Nothing is touching the ground so setting the friction and ground to 0 

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        # Set upper and lower bounds of the envs 
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing) 

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")

        spot_asset_file = "urdf/spot_description/spot_7dof_psyonic_no_base.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False 
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False 
        asset_options.thickness = 0.001 
        asset_options.use_mesh_materials = True 
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        self.spot_asset = self.gym.load_asset(self.sim, asset_root, spot_asset_file, asset_options)
        self.spot_assets = [self.spot_asset]

        self._create_object_envs(asset_root)


        print("Orthrus body cnt: ", self.gym.get_asset_rigid_body_count(self.spot_asset))

        self.num_spot_bodies = self.gym.get_asset_rigid_body_count(self.spot_asset) 
        self.num_spot_dofs = self.gym.get_asset_dof_count(self.spot_asset)



        print("Total num spot bodies: ", self.num_spot_bodies)
        print("Total num spot dofs: ", self.num_spot_dofs)

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
        self.spot_dof_lower_limits = []
        self.spot_dof_upper_limits = []
        self.spot_effort_limits = []

        spot_dof_props = self.gym.get_asset_dof_properties(self.spot_asset)    

        lower_limits = []
        upper_limits = []
        effort_limits = []

        self._dof_lower_limits = [] 
        self._dof_upper_limits = []
        self._dof_effort_limits = [] 


        for i in range(self.num_spot_dofs):
            if self.physics_engine == gymapi.SIM_PHYSX:
                spot_dof_props['stiffness'][i] = 1000.0
                spot_dof_props['damping'][i] = 300.0
            else: 
                spot_dof_props['stiffness'][i] = 7000.0
                spot_dof_props['damping'][i] = 50.0

            lower_limits.append(spot_dof_props['lower'][i])
            upper_limits.append(spot_dof_props['upper'][i])
            effort_limits.append(spot_dof_props['effort'][i])

        self._dof_lower_limits = to_torch(lower_limits, device=self.device)
        self._dof_upper_limits = to_torch(upper_limits, device=self.device)
        self._dof_effort_limits = to_torch(effort_limits, device=self.device)


        # Assign the converted tensors back to the original variables
        lower_limits = self._dof_lower_limits
        upper_limits = self._dof_upper_limits
        effort_limits = self._dof_effort_limits

        # set object dof properties
        for tag in self.objects:
            obj_dof_props = self.gym.get_asset_dof_properties(self.objects[tag].asset)
            for prop in obj_dof_props:
                if not prop['hasLimits']:
                    prop['lower'] = 0.0
                    prop['upper'] = 0.0
                    prop['driveMode'] = gymapi.DOF_MODE_POS
                    prop['stiffness'] = 100
                    prop['damping'] = 100
                    prop['velocity'] = 5
                    prop['effort'] = 1
                    prop['friction'] = 100.0
                # prop['armature'] = 0.0
            self.objects[tag].dof_prop = obj_dof_props

        spot_start_pose = gymapi.Transform()

        spot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.7)
        spot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

         # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        object_lounge = gymapi.Transform()
        object_lounge.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        object_lounge.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)



        num_spot_bodies = sum([self.gym.get_asset_rigid_body_count(asset) for asset in self.spot_assets + get_assets(self.objects)])
        num_spot_shapes = sum([self.gym.get_asset_rigid_shape_count(asset) for asset in self.spot_assets + get_assets(self.objects)])
        self.num_bodies = max_agg_bodies = num_spot_bodies    # 1 for table, 2 for table stands(x2), objects(cube, ball, etc)
        self.num_shapes = max_agg_shapes = num_spot_shapes    # 1 for table, 2 for table stands(x2), objects(cube, ball, etc)

        self.envs = []

      # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

           
            """
            * bitwise collision filter can be defined as following:
                left_ur3: 4 -->   100 (binary number)
                right_ur3: 8 --> 1000 (binary number)
                Both the left and right UR3 arms are not intersecting, so they can collide with each other
            """
            # Spot 
            self._spot_id = self.gym.create_actor(env_ptr, self.spot_asset, spot_start_pose, "spot", i, 1, 0)
            print("Spot index: ", self.gym.get_actor_index(env_ptr, self._spot_id, gymapi.DOMAIN_SIM))

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

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

        # Create a joint mapping 
        env_ptr = self.envs[0]
        
        dof_names = self.gym.get_actor_dof_names(env_ptr, self._spot_id)

        self.isaacgym_control_names = dof_names 
        self.joint_idx_mapping = [self.isaacgym_control_names.index(name) for name in self.input_control_names]
        print("Joint idx mapping: ", self.joint_idx_mapping)
        print("IsaacGym control names: ", self.isaacgym_control_names)


        self.spot_body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, self._spot_id)
        """
        Rigid body flags:
            gymapi.RIGID_BODY_NONE= 0
            gymapi.RIGID_BODY_DISABLE_GRAVITY= 1
            gymapi.RIGID_BODY_DISABLE_SIMULATION(PhysX only)= 2
        """

        # if you want to show the items of the dict, comment out the following codes
        # sorted_dict = dict(sorted(self.spot_body_dict.items(), key=lambda item: item[1]))
        # for key, value in sorted_dict.items():
        #     print(f"{key}: {value}")

        """
            env: 0                      
            robot1/link1: 1             
            robot1/link2: 2             
            robot1/link3: 3             
            robot1/link4: 4             
            robot1/link5: 5             
            robot1/link6: 6             
            robot1/link7: 7             
            robot1/end_link: 8          
            robot1/base: 9              
            robot1/end_effector_link: 10
            robot1/thumb_base: 11   
            robot1/index_L1: 12     
            robot1/index_L2: 13     
            robot1/index_anchor: 14 
            robot1/middle_L1: 15    
            robot1/middle_L2: 16    
            robot1/middle_anchor: 17
            robot1/pinky_L1: 18         
            robot1/pinky_L2: 19         
            robot1/pinky_anchor: 20     
            robot1/ring_L1: 21          
            robot1/ring_L2: 22          
            robot1/ring_anchor: 23      
            robot1/thumb_L1: 24         
            robot1/thumb_L2: 25         
            robot1/thumb_anchor: 26     
            robot1/link6_middle: 27     
            robot1/link5_middle: 28     
            robot1/link4_middle: 29     
            robot1/link3_middle: 30     
            robot1/link2_middle: 31     
            robot2/link1: 32            
            robot2/link2: 33            
            robot2/link3: 34            
            robot2/link4: 35            
            robot2/link5: 36            
            robot2/link6: 37            
            robot2/link7: 38            
            robot2/end_link: 39         
            robot2/base: 40             
            robot2/end_effector_link: 41
            robot2/thumb_base: 42   
            robot2/index_L1: 43     
            robot2/index_L2: 44     
            robot2/index_anchor: 45 
            robot2/middle_L1: 46    
            robot2/middle_L2: 47    
            robot2/middle_anchor: 48
            robot2/pinky_L1: 49    
            robot2/pinky_L2: 50    
            robot2/pinky_anchor: 51
            robot2/ring_L1: 52
            robot2/ring_L2: 53
            robot2/ring_anchor: 54
            robot2/thumb_L1: 55
            robot2/thumb_L2: 56
            robot2/thumb_anchor: 57
            robot2/link6_middle: 58
            robot2/link5_middle: 59
            robot2/link4_middle: 60
            robot2/link3_middle: 61
            robot2/link2_middle: 62
        """
        
        # import pdb 
        # pdb.set_trace()
        # Setup init state buffer
        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_object_state[:, 7] = 1.0     # unit quaternion

        self._init_states()

    def _init_states(self):
        env_ptr = self.envs[0]
        spot_handle = 0
        self.handles = {
            # UR3
            "hand_left": self.gym.find_actor_rigid_body_handle(env_ptr, spot_handle, "robot1/end_effector_link"),
            "hand_right": self.gym.find_actor_rigid_body_handle(env_ptr, spot_handle, "robot2/end_effector_link"),
            "base_left": self.gym.find_actor_rigid_body_handle(env_ptr, spot_handle, "robot1/link1"),
            "base_right": self.gym.find_actor_rigid_body_handle(env_ptr, spot_handle, "robot2/link2"),
            "football": self.gym.find_actor_rigid_body_handle(env_ptr, self._obj_ref_id, "football"),
        }

        bodies_to_detect_contacts = ["env", "robot1/link1", "robot1/link2", "robot1/link3",
                                     "robot1/link4", "robot1/link5", "robot1/link6", "robot1/link7",
                                     "robot1/thumb_base", "robot2/link1", "robot2/link2", "robot2/link3",
                                     "robot2/link4", "robot2/link5", "robot2/link6", "robot2/link7",
                                     "robot2/thumb_base"
                                     ]
        
        self.ids_for_arm_contact = get_indices_from_dict(self.spot_body_dict, bodies_to_detect_contacts)
        self._spot_jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "spot")
        self.spot_jacobian = gymtorch.wrap_tensor(self._spot_jacobian_tensor)

    def set_initial_football_state(self, pose):
        self._init_object_state = pose.unsqueeze(0).expand(self.num_envs, -1)
        # self.football_pose


    def _update_states(self):
        pass

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def _create_object_envs(self, asset_root):
        football_asset_file = "urdf/football.urdf"
        setattr(self.objects, "football", AttrDict())

        football_asset_options = gymapi.AssetOptions() 

        # Sankalp Should I overridee or trust the URDF file?
        football_asset_options.override_com = False      
        football_asset_options.override_inertia = False 
        football_asset_options.use_mesh_materials = True 
        football_asset_options.mesh_normal_mode = gymapi.MeshNormalMode.COMPUTE_PER_VERTEX
        football_asset_options.thickness = 0.01 

        self.objects["football"].asset = self.gym.load_asset(self.sim, asset_root, football_asset_file, football_asset_options)
        self.objects["football"].size = 0.3 

        self.num_objs = len(get_assets(self.objects))

        
    def pre_physics_step(self, actions):
        if actions.shape[1] != len(self.joint_idx_mapping):
            actions = torch.zeros((self.num_envs, len(self.joint_idx_mapping)), dtype=torch.float32, device=self.device)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(root_tensor)

        object_indices = torch.arange(self.num_envs, device=self.device) * 2 + self.objects["football"].id
        root_states[object_indices,:] = self._init_object_state
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))

        self.actions = actions.clone().to(self.device)
        robot1_u_arm, robot1_u_finger = self.actions[:, :7], self.actions[:, 7:13]
        robot2_u_arm, robot2_u_finger = self.actions[:, 13:20], self.actions[:, 20:26]
        self._robot1_arm_control = robot1_u_arm
        self._robot1_finger_control = robot1_u_finger
        self._robot2_arm_control = robot2_u_arm
        self._robot2_finger_control = robot2_u_finger

        action = torch.zeros((self.num_envs, self.num_spot_dofs), dtype=torch.float32, device=self.device)
        velocity = torch.zeros_like(action)

        
        action[:,self.joint_idx_mapping] = self.actions 

        stacked_state = torch.stack([action, velocity], dim=2)
        stacked_state_flat = stacked_state.view(-1,2)

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(stacked_state_flat))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action.view(-1)))

    # implement pre-physics simulation code here
    #    - e.g. apply actions

    def post_physics_step(self):
        self._refresh()

        # contact_reward = self.compute_contact_rewards()

        stability_reward = self.compute_stability_reward()

        manipulability_reward = self.compute_manipulability_reward()

        self.total_reward = stability_reward + manipulability_reward

        # pass
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations


    def compute_contact_rewards(self):
        contact_reward = torch.zeros(self.num_envs, device=self.device)
        # contacts = self.gym.get_rigid_contacts(self.sim)
        # get current joint dof 
        joint_dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_dof_states = gymtorch.wrap_tensor(joint_dof_tensor).view(self.num_envs, 2, self.num_spot_dofs)

        # get current root states
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(root_tensor)

        # get only the object states
        object_indices = torch.arange(self.num_envs, device=self.device) * 2 + self.objects["football"].id
        object_states = root_states[object_indices, :]


        for i in range(self.num_envs):
            # set the pybullet scene
            p.resetBasePositionAndOrientation(self.pybullet_football, object_states[i, :3].cpu().numpy(), object_states[i, 3:7].cpu().numpy())
            p.resetBaseVelocity(self.pybullet_football, object_states[i, 7:10].cpu().numpy(), object_states[i, 10:13].cpu().numpy())
            for j,name in enumerate(self.input_control_names):
                pybullet_index = self.pybullet_joint_idx_mapping[j]
                gym_index = self.joint_idx_mapping[j]
                p.resetJointState(self.pybullet_robot, pybullet_index, joint_dof_states[i,0, gym_index].cpu().numpy())
            for z in range(100):
                p.stepSimulation()
                contacts = p.getContactPoints()
                if len(contacts) > 0:
                    print("Contact detected after ", z, " steps")
                    break
            for c in contacts:
                print(f"Contact between body {c[1]} link {c[3]} and body {c[2]} link {c[4]}")
                print(f"Contact position on A: {c[5]}, on B: {c[6]}")
                print(f"Contact normal on B: {c[7]}")
                print(f"Contact distance: {c[8]}, normal force: {c[9]}")
                print('---')

            # import pdb; pdb.set_trace()

        



    def compute_manipulability_reward(self):
        
        # robot1_eef_joint = self.gym.find_actor_dof_index(self.envs[0], self._spot_id, "robot1/end_effector_link")
        # robot2_eef_joint = self.gym.find_actor_dof_index(self.envs[0], self._spot_id, "robot2/end_effector_link")

        robot1_eef_jacobian = self.spot_jacobian[:, self.handles["hand_left"], :, :]
        robot2_eef_jacobian = self.spot_jacobian[:, self.handles["hand_right"], :, :]
        robot1_manip = torch.sqrt(torch.clamp(torch.linalg.det(torch.bmm(robot1_eef_jacobian, robot1_eef_jacobian.transpose(1, 2))),min=1e-6))
        robot2_manip = torch.sqrt(torch.clamp(torch.linalg.det(torch.bmm(robot2_eef_jacobian, robot2_eef_jacobian.transpose(1, 2))),min=1e-6))

        # Better to return the sum or multiply

        return robot1_manip + robot2_manip



    def compute_stability_reward(self):
        # get current root states
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(root_tensor)

        # get only the object states
        object_indices = torch.arange(self.num_envs, device=self.device) * 2 + self.objects["football"].id
        object_states = root_states[object_indices, :]


        # get the current pose and initial pose 
        current_pose = object_states[:, :3]
        initial_pose = self._init_object_state[:, :3]

        # euclidean distance shap should be [num_envs]
        dist = torch.norm(current_pose - initial_pose, dim=1)

        stability_reward = -dist
        return stability_reward

