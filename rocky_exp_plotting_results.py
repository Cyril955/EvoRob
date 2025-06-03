import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

    # Normalization functions
    normalize_x = lambda x: x / x_vals.max()
    normalize_y = lambda y: y / abs(y_vals.min())

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
        # ax1.plot(x_best, y_best, 'o', color='red', markersize=8)
        # ax1.plot(x_best, y_best, 'o', color=cmap(norm(j)), markersize=5) # NOTHING TO DO WITH NSGA!!
    ax1.set_ylim(-1, 0)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Normalized velocity reward term [m/s]', fontsize=11)
    ax1.set_ylabel('Normalized control cost reward term [Nm]', fontsize=11)
    ax1.grid()

    # best_handle = Line2D([], [], marker='o', color='red', linestyle='None', markerfacecolor='none', markersize=8, label='Best individual for each generation')
    # ax1.legend(handles=[best_handle], loc='upper right', fontsize=10)

    # Define ticks
    step = 10
    ticks = list(range(0, num_generations, step))
    if ticks[-1] != num_generations:
        ticks.append(num_generations)

    # Define ScalarMappable with correct normalization range
    norm = mcolors.Normalize(vmin=0, vmax=num_generations)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # required for colorbar

    # Create colorbar
    cbar = plt.colorbar(sm, ax=ax1, ticks=ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])  # Optional
    cbar.set_label('Generations', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig('rocky_exp_pareto_plot_normalized.png', dpi=600, bbox_inches='tight')
    plt.show()


    # Last generation plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    last_gen_rewards = all_rewards[-1]
    normalize_x = lambda x: x / 1
    normalize_y = lambda y: y / 1
    x_last = normalize_x(last_gen_rewards[:, 0])
    y_last = normalize_y(last_gen_rewards[:, 1])

    print(x_last.shape)

    ax2.plot(x_last, y_last, 'o', color='blue', markersize=6)
    ax2.set_title("Last Generation Only")
    ax2.set_xlabel('Normalized velocity reward term [m/s]', fontsize=11)
    ax2.set_ylabel('Normalized control cost reward term [Nm]', fontsize=11)
    ax2.grid()

    plt.tight_layout()
    plt.savefig('rocky_exp_last_generation_plot.png', dpi=600, bbox_inches='tight')
    plt.show()




    # # Leg lengths plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # front_back_ratio_mean = np.zeros(num_generations)
    # front_back_ratio_std = np.zeros(num_generations)
    # gen_index = np.arange(0, num_generations)

    # def convert_to_meters(param):
    #     # from [-1, 1] to [0.15, 0.35] m
    #     return (param + 1.5) / 5 * 0.5 + 0.1
    
    # for i in range(num_generations):
    #     all_genotype = np.load(os.path.join(results_dir, f"{i}", "x.npy"))

    #     all_front_left_leg = convert_to_meters(all_genotype[:, 0])
    #     all_front_left_ankle = convert_to_meters(all_genotype[:, 1])
    #     all_front_right_leg = convert_to_meters(all_genotype[:, 2])
    #     all_front_right_ankle = convert_to_meters(all_genotype[:, 3])
    #     all_back_left_leg = convert_to_meters(all_genotype[:, 4])
    #     all_back_left_ankle = convert_to_meters(all_genotype[:, 5])
    #     all_back_right_leg = convert_to_meters(all_genotype[:, 6])
    #     all_back_right_ankle = convert_to_meters(all_genotype[:, 7])

    #     all_left_leg_ratio = (all_front_left_leg + all_front_left_ankle) / (all_back_left_leg + all_back_left_ankle)
    #     all_right_leg_ratio = (all_front_right_leg + all_front_right_ankle) / (all_back_right_leg + all_back_right_ankle)
    #     all_ratio = np.concatenate((all_left_leg_ratio, all_right_leg_ratio))
        
    #     front_back_ratio_mean[i] = np.mean(all_ratio)
    #     front_back_ratio_std[i] = np.std(all_ratio)
    
    # fig3, ax3 = plt.subplots(figsize=(6, 4))

    # # Plot the mean
    # mean_line = ax3.plot(gen_index, front_back_ratio_mean, '-', color='purple', label='Mean front/back leg length ratio')

    # # Plot the shaded std dev
    # std_fill = ax3.fill_between(
    #     gen_index,
    #     front_back_ratio_mean - front_back_ratio_std,
    #     front_back_ratio_mean + front_back_ratio_std,
    #     color='purple',
    #     alpha=0.3,
    #     label='$\pm 1 \ \sigma$'
    # )

    # # To ensure both show up in the legend:
    # handles, labels = ax3.get_legend_handles_labels()
    # ax3.legend(handles, labels, loc='upper right', fontsize=10)

    # # Labels and limits
    # ax3.set_xlabel('Generations', fontsize=11)
    # ax3.set_ylabel('Leg length ratios', fontsize=11)
    # ax3.set_ylim(0, 2)
    # ax3.set_xlim(0, num_generations)
    # ax3.grid()

    # plt.tight_layout()
    # plt.savefig('rocky_exp_leg_ratios_plot.png', dpi=600, bbox_inches='tight')
    # plt.show()



def leg_figure(path_tilted, path_flat, num_generations, population_size):
    def convert_to_meters(param):
        # from [-1, 1] to [0.15, 0.35] m
        return (param + 1.5) / 5 * 0.5 + 0.1

    def compute_leg_ratios(path):
        ratio_mean = np.zeros(num_generations)
        ratio_std = np.zeros(num_generations)
        for i in range(num_generations):
            all_genotype = np.load(os.path.join(path, f"{i}", "x.npy"))

            all_front_left_leg = convert_to_meters(all_genotype[:, 0])
            all_front_left_ankle = convert_to_meters(all_genotype[:, 1])
            all_front_right_leg = convert_to_meters(all_genotype[:, 2])
            all_front_right_ankle = convert_to_meters(all_genotype[:, 3])
            all_back_left_leg = convert_to_meters(all_genotype[:, 4])
            all_back_left_ankle = convert_to_meters(all_genotype[:, 5])
            all_back_right_leg = convert_to_meters(all_genotype[:, 6])
            all_back_right_ankle = convert_to_meters(all_genotype[:, 7])

            all_left_leg_ratio = (all_front_left_leg + all_front_left_ankle) / (all_back_left_leg + all_back_left_ankle)
            all_right_leg_ratio = (all_front_right_leg + all_front_right_ankle) / (all_back_right_leg + all_back_right_ankle)
            all_ratio = np.concatenate((all_left_leg_ratio, all_right_leg_ratio))

            ratio_mean[i] = np.mean(all_ratio)
            ratio_std[i] = np.std(all_ratio)
        return ratio_mean, ratio_std

    gen_index = np.arange(0, num_generations)

    tilted_mean, tilted_std = compute_leg_ratios(path_tilted)
    flat_mean, flat_std = compute_leg_ratios(path_flat)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Tilted terrain plot
    ax.plot(gen_index, tilted_mean, '-', color='purple', label='Tilted: Mean front/back leg length ratio')
    ax.fill_between(gen_index, tilted_mean - tilted_std, tilted_mean + tilted_std,
                    color='purple', alpha=0.3, label='Tilted: $\pm 1 \ \sigma$')

    # Flat terrain plot
    ax.plot(gen_index, flat_mean, '-', color='green', label='Flat: Mean front/back leg length ratio')
    ax.fill_between(gen_index, flat_mean - flat_std, flat_mean + flat_std,
                    color='green', alpha=0.3, label='Flat: $\pm 1 \ \sigma$')

    ax.set_xlabel('Generations', fontsize=11)
    ax.set_ylabel('Leg length ratios', fontsize=11)
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0, num_generations)
    ax.grid()
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('leg_ratios_tilted_vs_flat.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results_dir = 'results/Ant_custom/multi'
    num_generations = 100
    population_size = 30

    path_tilted = 'results/Ant_custom/tilted_map_100gen_30ind'
    path_flat = 'results/Ant_custom/Flat_map_no_rocks_100gen_30ind'

    create_figures(results_dir, num_generations, population_size)
    # leg_figure(path_tilted, path_flat, num_generations, population_size)