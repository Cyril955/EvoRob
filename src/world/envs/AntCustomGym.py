from os import path
from typing import Dict, Union

import numpy as np
import math
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from src.utils.geometry import quat2rot

DEFAULT_CAMERA_CONFIG = {
    "distance": 5,
}


class AntCustomEnv(MujocoEnv, utils.EzPickle):
    r"""In this environment a Passive Dynamic Walker is tasked to locomote.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        upward_reward_weight: float = 1,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        main_body: Union[int, str] = 1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        pert_force=None,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            robot_path,
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            upward_reward_weight,
            forward_reward_weight,
            ctrl_cost_weight,
            cfrc_cost_weight,
            main_body,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            pert_force,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._upward_reward_weight = upward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            width=832,
            height=496,
            camera_name="track",
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += (
            self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        self.body_ids = None
        self.force = None
        self.previous_state = None
        self.stuck = 0
        if pert_force is not None:
            self.body_ids , self.force = pert_force


    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        z_position_before = self.data.body(self._main_body).xpos[2].copy()  # for climber

        if self.body_ids is not None:
            self.apply_force()
        self.do_simulation(action, self.frame_skip)

        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()
        z_position_after = self.data.body(self._main_body).xpos[2].copy()   # for climber

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        z_velocity = (z_position_after - z_position_before) / self.dt       # for climber
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity * self._forward_reward_weight
        upward_reward = z_velocity * self._upward_reward_weight             # for climber
        healthy_reward = 1
        ctrl_cost = np.linalg.norm(action)**2 * self._ctrl_cost_weight
        cfrc_cost = np.linalg.norm( self.data.cfrc_ext[1:])**2 * self._cfrc_cost_weight

        reward = healthy_reward + forward_reward -ctrl_cost -cfrc_cost
        observation = self._get_obs()

        info = {
            "upward_reward": upward_reward,
            "reward_forward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }

        # TERMINATION CONDITIONS
        terminated = False

        # Limit the acceleration to a reasonable range
        qacc = self.data.qacc
        if np.any(np.isnan(qacc)) or np.any(np.isinf(qacc)) or np.any(np.abs(qacc) > 1e6):
            DOF = np.argwhere((np.isnan(qacc)) + (np.isinf(qacc)) + (np.abs(qacc) > 1e6)).squeeze()[0]
            print(ValueError(f'MuJoCo Warning: Nan, Inf or huge value in QACC at DOF {DOF}'))
            terminated = True
        
        #print(self.data.body(self._main_body).xpos[2])
        # print xquat
        print(self.data.body(self._main_body).xquat)
        # Limit if it falls down the hill (z position) 
        if self.data.body(self._main_body).xpos[2] < 0:       
            terminated = True

        # # Limit if it falls on its back
        def is_180_deg_rotation_xy(xquat):
            w, x, y, z = xquat
            tol_rad = math.radians(90)

            # compute rotation angle
            w_clamped = max(-1.0, min(1.0, w))
            angle = 2 * math.acos(w_clamped)
            if abs(angle - math.pi) > tol_rad:
                return False

            # compute rotation axis
            sin_half = math.sin(angle / 2)
            if abs(sin_half) < 1e-6:
                return False
            ax, ay, az = x / sin_half, y / sin_half, z / sin_half

            # check X-axis (±1, 0, 0)
            if abs(abs(ax) - 1.0) < 1e-2 and abs(ay) < 1e-2 and abs(az) < 1e-2:
                return True
            # check Y-axis (0, ±1, 0)
            if abs(abs(ay) - 1.0) < 1e-2 and abs(ax) < 1e-2 and abs(az) < 1e-2:
                return True

            return False
        
        import math

        def fall_down(xquat, tol=5):
            w, x, y, z = xquat

            # Roll (X-axis rotation)
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll_rad = math.atan2(t0, t1)

            # Pitch (Y-axis rotation)
            t2 = 2.0 * (w * y - z * x)
            t2 = max(-1.0, min(1.0, t2))  # clamp to avoid domain errors
            pitch_rad = math.asin(t2)

            # Yaw (Z-axis rotation)
            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw_rad = math.atan2(t3, t4)

            # Convert to degrees
            roll_deg  = math.degrees(roll_rad)
            pitch_deg = math.degrees(pitch_rad)
            yaw_deg   = math.degrees(yaw_rad)

            #if roll_deg or pitch_deg or yaw_deg are < abs(120) return false else return true
            if abs(roll_deg) > 180-tol or abs(pitch_deg) > 90 - tol:
                # print("roll_deg: ", roll_deg)
                # print("pitch_deg: ", pitch_deg)
                # print("yaw_deg: ", yaw_deg)
                return True
            return False

        if (fall_down(self.data.body(self._main_body).xquat)):     
            print("Ant fell on its back")  
            terminated = True

        # Limit if there is a huge value in the observation
        if np.isinf(observation).any():
            terminated = True

        self.previous_state = observation

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))


    def apply_force(self):
        body_id = self.body_ids
        force = self.force
        pert = self.np_random.uniform(
            low=-0.1, high=0.1, size=3)
        rot = quat2rot([1, *pert])
        force = np.dot(rot, force.reshape(2, 3).T).T.flatten()
        self.data.xfrc_applied[body_id] = force


    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale
            * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
