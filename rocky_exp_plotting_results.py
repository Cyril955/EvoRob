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
    # Pareto front plot
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    cmap = mpl.colormaps.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=num_generations - 1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for j in range(num_generations):
        reward_i = np.load(os.path.join(results_dir, f"{j}", "f.npy"))
        ax1.plot(reward_i[:, 0], reward_i[:, 1], 'o', color=cmap(norm(j)), markersize=5)

        best_reward_i = np.load(os.path.join(results_dir, f"{j}", "f_best.npy"))
        ax1.plot(best_reward_i[0], best_reward_i[1], 'o',  color='red', markersize=8)
        ax1.plot(best_reward_i[0], best_reward_i[1], 'o', color=cmap(norm(j)), markersize=5)

    ax1.set_xlabel('Weighted upward velocity', fontsize=11)
    ax1.set_ylabel('Weighted control cost', fontsize=11)
    ax1.grid()

    best_handle = Line2D([], [], marker='o', color='red', linestyle='None',markerfacecolor='none', markersize=8, label='Best individual for each generation')
    ax1.legend(handles=[best_handle], loc='upper left', fontsize=10)

    cbar = plt.colorbar(sm, ax=ax1, ticks=np.linspace(0, num_generations - 1, min(num_generations, 10)))
    cbar.set_label('Generations', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig('rocky_exp_pareto_plot.png', dpi=600, bbox_inches='tight')
    plt.show()


    # Reward plot
    best_reward = np.zeros((num_generations, 2))
    gen_index = np.arange(1, num_generations+1)
    print(results_dir)

    for i in range(num_generations):
        best_reward[i][:] = np.load(os.path.join(results_dir, f"{i}", "f_best.npy"))

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.plot(gen_index, best_reward[:, 0], '-', color='darkorange', label='Best individual weighted upward velocity')
    ax2.plot(gen_index, best_reward[:, 1], '-', color='royalblue', label='Best individual weighted control cost')
    ax2.set_xlabel('Generations', fontsize=11)
    ax2.set_ylabel('Reward function terms', fontsize=11)
    ax2.grid()
    ax2.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    plt.savefig('rocky_exp_rewards_plot.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    results_dir = 'results/Ant_custom/multi'
    num_generations = 24
    population_size = 20

    create_figures(results_dir, num_generations, population_size)