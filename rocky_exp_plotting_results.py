import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False 
})


def create_figures(results_dir, num_generations, population_size):
    # Pareto front plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    all_rewards = []
    all_best_rewards = []

    for j in range(num_generations):
        reward_i = np.load(os.path.join(results_dir, f"{j}", "f.npy"))
        best_reward_i = np.load(os.path.join(results_dir, f"{j}", "f_best.npy"))

        all_rewards.append(reward_i)
        all_best_rewards.append(best_reward_i)

    # Stack all for normalization
    all_data = np.vstack(all_rewards + all_best_rewards)

    x_vals = all_data[:, 0]
    y_vals = all_data[:, 1]

    # Normalization parameters
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # Normalization functions
    normalize_x = lambda x: (x - x_min) / (x_max - x_min)
    normalize_y = lambda y: -1 + (y - y_min) * (1 / (y_max - y_min))

    # Plotting
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    cmap = mpl.colormaps.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=num_generations - 1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for j in range(num_generations):
        reward_i = all_rewards[j]
        best_reward_i = all_best_rewards[j]

        x = normalize_x(reward_i[:, 0])
        y = normalize_y(reward_i[:, 1])
        ax1.plot(x, y, 'o', color=cmap(norm(j)), markersize=5)

        x_best = normalize_x(best_reward_i[0])
        y_best = normalize_y(best_reward_i[1])
        ax1.plot(x_best, y_best, 'o', color='red', markersize=8)
        ax1.plot(x_best, y_best, 'o', color=cmap(norm(j)), markersize=5)

    ax1.set_xlabel('Normalized velocity reward term [m/s]', fontsize=11)
    ax1.set_ylabel('Normalized control cost reward term [Nm]', fontsize=11)
    ax1.grid()

    best_handle = Line2D([], [], marker='o', color='red', linestyle='None', markerfacecolor='none', markersize=8, label='Best individual for each generation')
    ax1.legend(handles=[best_handle], loc='upper right', fontsize=10)

    cbar = plt.colorbar(sm, ax=ax1, ticks=np.linspace(0, num_generations-1, min(num_generations, 10)))
    cbar.set_label('Generations', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig('rocky_exp_pareto_plot_normalized.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Reward plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    best_reward = np.zeros((num_generations, 2))
    gen_index = np.arange(1, num_generations + 1)

    for i in range(num_generations):
        best_reward[i][:] = np.load(os.path.join(results_dir, f"{i}", "f_best.npy"))

    # Normalize both axes
    x_vals = best_reward[:, 0]
    y_vals = best_reward[:, 1]

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # Define normalization functions
    normalize_x = lambda x: (x - x_min) / (x_max - x_min)
    normalize_y = lambda y: (y - y_min) / (y_max - y_min)

    # Apply normalization
    x_normalized = normalize_x(x_vals)
    y_normalized = normalize_y(y_vals)

    # Plotting
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.plot(gen_index, x_normalized, '-', color='darkorange', label='Normalized velocity reward term [m/s]')
    ax2.plot(gen_index, y_normalized, '-', color='royalblue', label='Normalized control cost reward term [Nm]')

    ax2.set_xlabel('Generations', fontsize=11)
    ax2.set_ylabel('Normalized reward function terms', fontsize=11)
    ax2.grid()
    ax2.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig('rocky_exp_rewards_plot_normalized.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Leg lengths plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    best_individual_genotype = np.zeros((num_generations, 953))

    front_leg_length = np.zeros((num_generations, 4))  # front: [upper0, lower1, upper2, lower3]
    back_leg_length = np.zeros((num_generations, 4))   # back:  [upper4, lower5, upper6, lower7]

    front_back_ratio = np.zeros(num_generations)
    upper_lower_front_ratio = np.zeros(num_generations)
    upper_lower_back_ratio = np.zeros(num_generations)
    gen_index = np.arange(1, num_generations + 1)

    def convert_to_meters(param):
        # from [-1, 1] to [0.15, 0.35] m
        return (param + 1.5) / 5 * 0.5 + 0.1

    # Processing
    for i in range(num_generations):
        genotype = np.load(os.path.join(results_dir, f"{i}", "x_best.npy"))
        best_individual_genotype[i][:] = genotype

        # Extract and convert
        front_upper = convert_to_meters(np.array([genotype[0], genotype[2]]))  # front left/right upper
        front_lower = convert_to_meters(np.array([genotype[1], genotype[3]]))  # front left/right lower
        back_upper  = convert_to_meters(np.array([genotype[4], genotype[6]]))  # back left/right upper
        back_lower  = convert_to_meters(np.array([genotype[5], genotype[7]]))  # back left/right lower

        avg_front_leg = np.mean(np.concatenate([front_upper, front_lower]))
        avg_back_leg  = np.mean(np.concatenate([back_upper, back_lower]))

        front_back_ratio[i] = avg_front_leg / avg_back_leg if avg_back_leg != 0 else np.nan
        upper_lower_front_ratio[i] = np.mean(front_upper) / np.mean(front_lower) if np.mean(front_lower) != 0 else np.nan
        upper_lower_back_ratio[i] = np.mean(back_upper) / np.mean(back_lower) if np.mean(back_lower) != 0 else np.nan

    fig3, ax3 = plt.subplots(figsize=(6, 4))

    ax3.plot(gen_index, front_back_ratio, '-', color='orange', label='Front / Back leg length ratio')
    ax3.plot(gen_index, upper_lower_front_ratio, '-', color='green', label='Front upper / lower leg ratio')
    ax3.plot(gen_index, upper_lower_back_ratio, '-', color='purple', label='Back upper / lower leg ratio')

    ax3.set_xlabel('Generations', fontsize=11)
    ax3.set_ylabel('Leg length ratios', fontsize=11)
    ax3.grid()
    ax3.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('rocky_exp_leg_ratios_plot.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    results_dir = 'results/Ant_custom/multi'
    num_generations = 1
    population_size = 20

    create_figures(results_dir, num_generations, population_size)