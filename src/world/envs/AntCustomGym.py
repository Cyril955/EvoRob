from os import path
from typing import Dict, Union

import numpy as np
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
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 1,
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
        # Multi-reward function ----------------------------------------------------------------------------------------------------------------------------------------------
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        z_position_before = self.data.body(self._main_body).xpos[2].copy()
        if self.body_ids is not None:
            self.apply_force()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()
        z_position_after = self.data.body(self._main_body).xpos[2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        z_velocity = (z_position_after - z_position_before) / self.dt

        # forward_reward = abs(x_velocity)
        forward_reward = np.sqrt(x_velocity**2 + z_velocity**2) # tilted case
        ctrl_cost = np.linalg.norm(action)**2
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Not used ------------------------------------------------------------------------------------------------------------------------------------------------------------
        cfrc_cost = np.linalg.norm(self.data.cfrc_ext[1:])**2 * self._cfrc_cost_weight
        healthy_reward = 1
        reward = healthy_reward + forward_reward -ctrl_cost -cfrc_cost
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

         # Termination --------------------------------------------------------------------------------------------------------------------------------------------------------
        observation = self._get_obs()
        terminated = False

        # Check for NaN, Inf, or huge values
        qacc = self.data.qacc
        if np.any(np.isnan(qacc)) or np.any(np.isinf(qacc)) or np.any(np.abs(qacc) > 1e6):
            DOF = np.argwhere((np.isnan(qacc)) + (np.isinf(qacc)) + (np.abs(qacc) > 1e6)).squeeze()[0]
            print(ValueError(f'MuJoCo Warning: Nan, Inf or huge value in QACC at DOF {DOF}'))
            terminated = True

        # Check fo z-position
        # if self.data.qpos[2] < 0.2 or self.data.qpos[2] > 1.0:
        #     terminated = True

        if self.data.qpos[2] < 0.2:
            ctrl_cost += 10
        
        # Check if inf observation
        if np.isinf(observation).any():
            terminated = True

        # Check if flipped over
        def flipped_over(xquat, tol=5):
            w, x, y, z = xquat

            # Roll (X-axis rotation)
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll_rad = np.atan2(t0, t1)

            # Pitch (Y-axis rotation)
            t2 = 2.0 * (w * y - z * x)
            t2 = max(-1.0, min(1.0, t2))  # clamp to avoid domain errors
            pitch_rad = np.asin(t2)

            # Yaw (Z-axis rotation)
            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw_rad = np.atan2(t3, t4)

            # Convert to degrees
            roll_deg  = np.degrees(roll_rad)
            pitch_deg = np.degrees(pitch_rad)
            yaw_deg   = np.degrees(yaw_rad)

            #if roll_deg or pitch_deg or yaw_deg are < abs(120) return false else return true
            if abs(roll_deg) > 180-tol or abs(pitch_deg) > 90 - tol:
                # print("roll_deg: ", roll_deg)
                # print("pitch_deg: ", pitch_deg)
                # print("yaw_deg: ", yaw_deg)
                return True
            return False

        if (flipped_over(self.data.body(self._main_body).xquat)):     
            ctrl_cost += 10 

        self.previous_state = observation

        if self.render_mode == "human":
            self.render()
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

        info = {
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