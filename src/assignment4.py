import numpy as np
import A4.frozenlake as frozenlake
from mdptoolbox import mdp
import A4.tictactoe as ttt
from util import mktmpdir
import matplotlib.pyplot as plt
from timeit import default_timer
import pickle


def save_stats(outdir, name, stat):
    with open(f'{outdir}/{name}.pkl', 'wb') as f:
        pickle.dump(stat, f)


def frozen_lake(gamma=0.99):
    print('==== Running FrozenLake ====\n')
    outdir = mktmpdir('a4_fl')
    timings = {}
    for fl_size in [8]:
        timings[fl_size] = {}
        print(f'Frozen Lake {fl_size} x {fl_size}')
        fl = frozenlake.custom_frozen_lake(size=fl_size)
        print('** Starting value iteration')
        vi_time = default_timer()
        policy, V, n_iter, r, delta_history = frozenlake.value_iteration(fl,
                                                                         theta=1e-8,
                                                                         discount_factor=gamma)
        vi_time = default_timer() - vi_time
        print('Converged in {} iterations ({} s)'.format(n_iter, vi_time))
        plot_policy(fl, policy, title=f'{fl_size} x {fl_size} VI',
                    outfile=f'{outdir}/fl{fl_size}_vi_map.png')
        save_stats(outdir, f'fl{fl_size}_vi_policy', policy)

        print('\n**Starting policy iteration')
        pi_time = default_timer()
        pi_policy, pi_V, pi_n_iter, pi_r, pi_d = frozenlake.policy_iteration(fl, discount_factor=gamma)
        pi_time = default_timer() - pi_time
        print('PI converged in {} iterations ({} seconds)'.format(pi_n_iter, pi_time))
        save_stats(outdir, f'fl{fl_size}_pi_policy', pi_policy)

        print('PI and VI converged to same policy? {}'
              .format(np.all(policy == pi_policy)))
        plot_policy(fl, pi_policy, title=f'{fl_size} x {fl_size} PI',
                    outfile=f'{outdir}/fl{fl_size}_pi_map.png')

        # plot delta history for VI and PI
        plt.plot(delta_history)
        plt.plot(pi_d)
        plt.savefig(f'{outdir}/fl{fl_size}_delta.png')
        plt.close('all')

        print('\n**Starting SARSA learner')
        alpha = 0.7
        alpha_decay = 0.9
        gamma = 0.999
        epsilon = [0.4, 0.9]
        epsilon_decay = 0.9
        episodes = 50000 if fl_size < 50 else 100000
        max_steps = 2500
        n_tests = 2
        sarsa_time = [0, 0]
        for i, e in enumerate(epsilon):
            sarsa_time[i] = default_timer()
            timestep_reward, Q = frozenlake.sarsa(fl,
                                                  alpha, alpha_decay,
                                                  gamma,
                                                  e, epsilon_decay,
                                                  episodes,
                                                  max_steps,
                                                  n_tests,
                                                  test=True,
                                                  render=False)
            sarsa_time[i] = default_timer() - sarsa_time[i]
            print('SARSA completed in {} seconds'.format(sarsa_time[i]))
            avg_reward = mean_per_t(timestep_reward, 1000)
            plt.plot(avg_reward)
            plt.xlabel('1000s of Iterations')
            plt.ylabel('Average Reward')
            plt.savefig(f'{outdir}/fl{fl_size}_e{e}_sarsa_avgreward.png')
            save_stats(outdir, f'fl{fl_size}_e{e}_sarsa_reward', timestep_reward)
            plot_policy(fl, Q, title=make_policy_title(fl_size, alpha, e, gamma, prefix='SARSA'),
                        outfile=f'{outdir}/fl{fl_size}_e{e}_sarsa_map.png')

        timings[fl_size] = {
            'vi': vi_time,
            'pi': pi_time,
            'sarsa_e40': sarsa_time[0],
            'sarsa_e90': sarsa_time[1]
        }
        print(timings)


def tictactoe(gamma=0.95):
    outdir = mktmpdir('a4_ttt')
    timings = {}
    print('====== Running Tic Tac Toe =======')
    gamma = 0.95
    P, R = ttt.getTransitionAndRewardArrays()

    print('\nValue Iteration')
    ttt_vi = mdp.ValueIteration(P, R, gamma)
    ttt_vi.setVerbose()
    vi_time = default_timer()
    ttt_vi.run()
    vi_time = default_timer() - vi_time
    print(f'MDP Toolbox VI finished in {ttt_vi.iter} iterations')
    print(f'Accumulated reward: {len(ttt_vi.rewards)}')
    print(f'Rewards: {ttt_vi.rewards}')
    save_stats(outdir, f'vi', ttt_vi)

    print('\nPolicy Iteration')
    ttt_pi = mdp.PolicyIteration(P, R, gamma)
    ttt_pi.setVerbose()
    pi_time = default_timer()
    ttt_pi.run()
    pi_time = default_timer() - pi_time
    print(f'MDP Toolbox PI finished in {ttt_pi.iter} iterations')
    print(f'Accumulated reward: {len(ttt_pi.rewards)}')
    print(f'Rewards: {ttt_pi.rewards}')
    save_stats(outdir, 'pi', ttt_pi)

    print('PI/VI same policy?: {}'.format(np.all(ttt_vi.policy == ttt_pi.policy)))
    save_stats(outdir, 'pi_policy', ttt_pi.policy)
    save_stats(outdir, 'vi_policy', ttt_vi.policy)

    # Q vs random
    epsilons = [0.4, 0.9]
    rewards = []
    agents = []
    qtimes = []
    for i, epsilon in enumerate(epsilons):
        qtimes.append(default_timer())
        r, agent = ttt.train_agents('random', 500000, epsilon, 0.9, 0.4, 0.9, 0.99, False)
        qtimes[i] = default_timer() - qtimes[i]
        rewards.append(r)
        agents.append(agent)
        qpolicy = agent.policy()

        save_stats(outdir, f'ttt_agents{epsilon}', agent)
        save_stats(outdir, f'ttt_rewards{epsilon}', r)
        save_stats(outdir, f'q_policy_{epsilon}', qpolicy)
        # print(f'{epsilon} policy same as vi?: {np.all(ttt_vi.policy == qpolicy)}')

    timings = {
        # 'vi': vi_time,
        # 'pi': pi_time,
        'q_eps4': qtimes[0],
        'q_eps7': qtimes[1]
    }
    print(timings)


def play_ttt():
    agent_path = 'output/a4_ttt/ttt_agents0.9.pkl'
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
        # set to non-explore mode
        agent.epsilon = 0.
        ttt.play_game(agent)


def mean_per_t(rewards, t):
    """
    Compute mean reward over t timesteps
    :param rewards: numpy array containing rewards
    :param t: number of time steps
    :return: numpy array containing avg rewards
    """
    return np.mean(rewards.reshape(-1, t), axis=1)


def plot_policy(env, Q, figsize=(12, 8), title='Learned Policy', outfile=None):
    """
    Plot quiver plot of policy for gridworld
    :param env:
    :param Q:
    :param figsize:
    :param title:
    :param outfile:
    :return:

    Based on:
        https://medium.com/analytics-vidhya/intro-to-reinforcement-learning-q-learning-101-8df55fed9be0
    """
    x = np.linspace(0, env.ncol - 1, env.ncol) + 0.5
    y = np.linspace(env.nrow - 1, 0, env.nrow) + 0.5
    X, Y = np.meshgrid(x, y)
    zeros = np.zeros((env.nrow, env.ncol))

    # Generate list of the map
    row, col = env.s // env.ncol, env.s % env.ncol
    desc = env.desc.tolist()
    desc = [[c.decode('utf-8') for c in line] for line in desc]

    # Set up colors for holes and goal
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    plt.text(0, env.nrow - 1, 'S')

    # Get max values
    q_table = np.array([Q[:, c].reshape((env.nrow, env.ncol)) for c in range(Q.shape[1])])
    q_max = q_table.max(axis=0)
    for i in range(env.nrow):
        for j in range(env.ncol):
            if desc[i][j] == 'H':
                continue
            if desc[i][j] == 'G':
                plt.text(env.ncol-1, 0, 'G')
                continue
            q_star = np.zeros((env.nrow, env.ncol))
            q_max_s = q_max[i, j]

            max_vals = np.where(q_max_s == q_table[:, i, j])[0]
            # print(np.where(q_max_s == q_table[:, i, j]))
            for action in max_vals:
                q_star[i, j] = 0.4
                # Plot results
                if action == 3:
                    # Move up
                    plt.quiver(X, Y, zeros, q_star, scale=1, units='xy')
                elif action == 0:
                    # Move left
                    plt.quiver(X, Y, -q_star, zeros, scale=1, units='xy')
                elif action == 1:
                    # Move down
                    plt.quiver(X, Y, zeros, -q_star, scale=1, units='xy')
                elif action == 2:
                    # Move right
                    plt.quiver(X, Y, q_star, zeros, scale=1, units='xy')

    plt.xlim([0, env.ncol])
    plt.ylim([0, env.nrow])
    ax.set_yticklabels([])
    ax.yaxis.set_ticks(range(env.nrow))
    ax.set_xticklabels([])
    ax.xaxis.set_ticks(range(env.ncol))
    plt.title(title)
    plt.grid(True)
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    plt.close('all')


def make_policy_title(size, alpha, epsilon, gamma, prefix='Policy'):
    return f'{prefix}: {size}x{size}, α={alpha}, ε={epsilon}, γ={gamma}'


if __name__ == "__main__":
    frozen_lake()
    tictactoe()
    play_ttt()