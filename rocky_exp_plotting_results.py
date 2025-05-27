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
    # # Pareto front plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # fig1, ax1 = plt.subplots(figsize=(6, 5))

    # cmap = mpl.colormaps.get_cmap('viridis')
    # norm = Normalize(vmin=0, vmax=num_generations - 1)
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # for j in range(num_generations):
    #     reward_i = np.load(os.path.join(results_dir, f"{j}", "f.npy"))
    #     ax1.plot(reward_i[:, 0], reward_i[:, 1], 'o', color=cmap(norm(j)), markersize=5)

    #     best_reward_i = np.load(os.path.join(results_dir, f"{j}", "f_best.npy"))
    #     ax1.plot(best_reward_i[0], best_reward_i[1], 'o',  color='red', markersize=8)
    #     ax1.plot(best_reward_i[0], best_reward_i[1], 'o', color=cmap(norm(j)), markersize=5)

    # ax1.set_xlabel('Weighted upward velocity', fontsize=11)
    # ax1.set_ylabel('Weighted control cost', fontsize=11)
    # ax1.grid()

    # best_handle = Line2D([], [], marker='o', color='red', linestyle='None',markerfacecolor='none', markersize=8, label='Best individual for each generation')
    # ax1.legend(handles=[best_handle], loc='upper left', fontsize=10)

    # cbar = plt.colorbar(sm, ax=ax1, ticks=np.linspace(0, num_generations - 1, min(num_generations, 10)))
    # cbar.set_label('Generations', fontsize=11)
    # cbar.ax.tick_params(labelsize=9)

    # plt.tight_layout()
    # plt.savefig('rocky_exp_pareto_plot.png', dpi=600, bbox_inches='tight')
    # plt.show()


    # # Reward plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # best_reward = np.zeros((num_generations, 2))
    # gen_index = np.arange(1, num_generations+1)

    # for i in range(num_generations):
    #     best_reward[i][:] = np.load(os.path.join(results_dir, f"{i}", "f_best.npy"))

    # fig2, ax2 = plt.subplots(figsize=(6, 4))

    # ax2.plot(gen_index, best_reward[:, 0], '-', color='darkorange', label='Best individual weighted upward velocity')
    # ax2.plot(gen_index, best_reward[:, 1], '-', color='royalblue', label='Best individual weighted control cost')
    # ax2.set_xlabel('Generations', fontsize=11)
    # ax2.set_ylabel('Reward function terms', fontsize=11)
    # ax2.grid()
    # ax2.legend(loc='lower left', fontsize=11)

    # plt.tight_layout()
    # plt.savefig('rocky_exp_rewards_plot.png', dpi=600, bbox_inches='tight')
    # plt.show()


    # Legs length plot --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    best_individual_genotype = np.zeros((num_generations, 953))
    upper_leg_length = np.zeros((num_generations, 4))
    lower_leg_length = np.zeros((num_generations, 4))
    avrage_upper_leg_length = np.zeros(num_generations)
    avrage_lower_leg_length = np.zeros(num_generations)
    gen_index = np.arange(1, num_generations+1)

    def convert_to_meters(param):
        # from[-1, 1] to [0.15, 0.35] m
        return (param + 1.5) / 5 * 0.5 + 0.1

    for i in range(num_generations):
        best_individual_genotype[i][:] = np.load(os.path.join(results_dir, f"{i}", "x_best.npy"))

        upper_leg_length[i][0] = convert_to_meters(best_individual_genotype[i][0]) # front_left_leg
        lower_leg_length[i][0] = convert_to_meters(best_individual_genotype[i][1]) # front_left_ankle
        upper_leg_length[i][1] = convert_to_meters(best_individual_genotype[i][2]) # front_right_leg
        lower_leg_length[i][1] = convert_to_meters(best_individual_genotype[i][3]) # front_right_ankle
        upper_leg_length[i][2] = convert_to_meters(best_individual_genotype[i][4]) # back_left_leg
        lower_leg_length[i][2] = convert_to_meters(best_individual_genotype[i][5]) # back_left_ankle
        upper_leg_length[i][3] = convert_to_meters(best_individual_genotype[i][6]) # back_right_leg
        lower_leg_length[i][3] = convert_to_meters(best_individual_genotype[i][7]) # back_right_ankle

        avrage_upper_leg_length[i] = np.mean(upper_leg_length[i])
        avrage_lower_leg_length[i] = np.mean(lower_leg_length[i])

    fig3, ax3 = plt.subplots(figsize=(6, 4))

    ax3.plot(gen_index, avrage_upper_leg_length, '-', color='blue', label='Average upper legs length')
    ax3.plot(gen_index, avrage_lower_leg_length, '-', color='lightblue', label='Average lower legs length')
    ax3.set_xlabel('Generations', fontsize=11)
    ax3.set_ylabel('Legs length [m]', fontsize=11)
    ax3.grid()
    ax3.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('rocky_exp_legs_plot.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    results_dir = 'results/Ant_custom/multi'
    num_generations = 100
    population_size = 20

    create_figures(results_dir, num_generations, population_size)