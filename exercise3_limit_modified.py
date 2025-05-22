from src.EA.CMAES import CMAES, CMAES_opts
from src.EA.NSGA import NSGAII, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv

import xml.etree.ElementTree as xml
import gymnasium as gym
import numpy as np
import os

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise3 body-brain: This is your first full body+brain evolution by adapting a custom Ant-v5 gym environment. 
    We adjust both leg lengths and controller weights for a locomotion task.   
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'

START_JOINT_LIMITS =  [[-30, 30], [30, 70], # orientation andd limits of the joints -> can be part of genotype
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [30, 70], ]

class AntWorld(World):
    def __init__(self, ):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        self.n_repeats = 3
        self.n_steps = 1000
        self.controller = MLP.NNController(state_space, action_space)
        self.n_weights = self.controller.n_params

        self.n_params = self.n_weights + 8 + 16
        self.world_file = os.path.join(ROOT_DIR, "AntEnv.xml")

        self.max_joint_limits = [[-30, 30], [30, 70], # orientation andd limits of the joints -> can be part of genotype
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [30, 70], ]
        
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           [0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           ]

# each joint in genotype raw_lo and raw_hi are in [-1, 1] range
# we defined a variable for the maximum joint limits : max_joint_limits
# We map each gene from its native range [-1, +1] into the physical range: 
# ex : l = (raw_lo + 1) / 2 * (hi_phys - lo_phys) + lo_phys 
# if raw_lo = -1 -> l = lo_phys, raw_lo = 1 -> l = hi_phys, raw_lo = 0 -> l = (lo_phys + hi_phys) / 2
# verify that the mapped values guarantee lo < hi, if not, sort them
    
    def geno2pheno(self, genotype):
        # --- extract and split genotype ---
        n_body   = 8
        n_joints = 8          # there are 8 joints, not 16
        n_jlim   = n_joints * 2  # 16 values total

        assert len(genotype) == n_body + n_jlim + self.n_weights, \
            f"genotype must be {n_body + n_jlim + self.n_weights} long"

        body_raw        = genotype[:n_body]
        joint_raw       = genotype[n_body:n_body + n_jlim]
        control_weights = genotype[-self.n_weights:]  # always take from the end

        # --- compute body parameters ---
        body_params = (body_raw + 1.5) / 5 * 0.5 + 0.1
        #print(f"Body parameters: {body_params}")

        # --- build joint‐limits table with linear mapping + guaranteed lo<hi ---
        joint_limits = np.zeros((n_joints, 2))
        for i in range(n_joints):
            lo_phys, hi_phys = self.max_joint_limits[i]

            # raw values in [-1,1]
            raw_lo = joint_raw[2*i]
            raw_hi = joint_raw[2*i + 1]

            # map each into the [lo_phys, hi_phys] interval
            mapped_lo = lo_phys + (raw_lo + 1) * 0.5 * (hi_phys - lo_phys)
            mapped_hi = lo_phys + (raw_hi + 1) * 0.5 * (hi_phys - lo_phys)

            # sort so that mapped_lo ≤ mapped_hi
            lo, hi = sorted([mapped_lo, mapped_hi])
            joint_limits[i, 0] = lo
            joint_limits[i, 1] = hi

            # verify they still respect the global bounds
            assert lo  >= lo_phys and lo  <= hi_phys, \
                f"Joint {i} lower {lo:.3f} outside [{lo_phys}, {hi_phys}]"
            assert hi  >= lo_phys and hi  <= hi_phys, \
                f"Joint {i} upper {hi:.3f} outside [{lo_phys}, {hi_phys}]"

        print(f"Joint limits:\n{joint_limits}")


        # --- sanity checks ---
        assert body_params.shape    == (n_body,)
        assert joint_limits.shape   == (n_joints, 2)
        assert control_weights.size == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle, = body_params

        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array(
            [np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array(
            [np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array(
            [-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array(
            [-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array(
            [-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array(
            [-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array(
            [np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array(
            [np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE] between the points just defined above
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        return points, connectivity_mat, joint_limits

    def evaluate_individual(self, genotype):
        points, connectivity_mat, joint_limits = self.geno2pheno(genotype)

        robot = AntRobot(points, connectivity_mat, joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml()

        # % Defining the Robot environment in MuJoCo
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding='unicode')

        with open(self.world_file, "w") as f:
            f.write(world_xml)

        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=0.1,
                    max_episode_steps=self.n_steps,
                )
                for i_env in range(self.n_repeats)
            ]
        )

        rewards_full = np.zeros((self.n_steps, self.n_repeats))
        multi_obj_rewards_full = np.zeros((self.n_steps, self.n_repeats, 2))  # TODO

        observations, info = envs.reset()
        done_mask = np.zeros(self.n_repeats, dtype=bool)
        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations.T).T)
            observations, rewards, dones, truncated, infos = envs.step(actions)

            # Store rewards for active environments only
            rewards_full[step, done_mask == False] = rewards[done_mask == False]

            # print(infos)

            multi_obj_reward = np.array([infos['reward_forward'], -infos['cfrc_cost']]).T # check dynamic walker -> reward is custom
            multi_obj_rewards_full[step, done_mask == False] = multi_obj_reward[done_mask == False]


            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


def run_EA_single(ea_single, world):
    for gen in range(ea_single.n_gen):
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_single.tell(pop, fitnesses_gen)


def run_EA_multi(ea_multi, world):
    for gen in range(ea_multi.n_gen):
        print(f"Generation {gen+1}/{ea_multi.n_gen}")
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_multi.tell(pop, fitnesses_gen)


def generate_best_individual_video(world, video_name: str = 'EvoRob3_video.mp4'):
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def visualise_individual(genotype):
    world = AntWorld()
    points, connectivity_mat, joint_limits = world.geno2pheno(genotype)
    robot = AntRobot(points, connectivity_mat, joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    for step in range(1000):
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    env.close()
    print(np.sum(rewards_list))


def main():
    # %% Understanding the world
    genotype = np.random.uniform(-1, 1, 969)  # 8 body parameters, 16 joint limits, 945 NN weights
    # set joint limits to a initial values
    visualise_individual(genotype)

    # # %% Optimise single-objective
    # world = AntWorld()
    # n_parameters = world.n_params

    # population_size = 250
    # CMAES_opts["min"] = -1
    # CMAES_opts["max"] = 1
    # CMAES_opts["num_parents"] = 100
    # CMAES_opts["num_generations"] = 100
    # CMAES_opts["mutation_sigma"] = 0.33

    # results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')
    # ea_single = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    # run_EA_single(ea_single, world)

    # %% Optimise multi-objective
    # TODO implement NSGAII
    world = AntWorld()
    n_parameters = world.n_params

    population_size = 10
    NSGA_opts["min"] = -1
    NSGA_opts["max"] = 1
    NSGA_opts["num_parents"] = population_size
    NSGA_opts["num_generations"] = 1
    NSGA_opts["mutation_prob"] = 0.3
    NSGA_opts["crossover_prob"] = 0.5

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'multi')
    ea_multi_obj = NSGAII(population_size, n_parameters, NSGA_opts, results_dir)

    run_EA_multi(ea_multi_obj, world)

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.
    best_individual = np.load(os.path.join(results_dir, f"{NSGA_opts['num_generations']-1}", "x_best.npy"))

    points, connectivity_mat, joint_limits = world.geno2pheno(best_individual)
    robot = AntRobot(points, connectivity_mat, joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    generate_best_individual_video(world)

    #print genotype : body parameters and joint limits only
    print("Best individual genotype:", best_individual[:24])  # first 8 body parameters + 16 joint limits


if __name__ == '__main__':
    main()