import numpy as np
import A4.frozenlake as frozenlake
from mdptoolbox import mdp
import A4.tictactoe as ttt


print('==== Running FrozenLake ====')
gamma = 0.99
# fl = frozenlake.custom_frozen_lake(size=10)
# print('Starting value iteration')
# policy, V, n_iter, r = frozenlake.value_iteration(fl, theta=1e-8, discount_factor=gamma)
# print('Converged in {} iterations'.format(n_iter))
# print(r)
#
# print('\nStarting policy iteration')
# pi_policy, pi_V, pi_n_iter, pi_r = frozenlake.policy_iteration(fl, discount_factor=gamma)
# print('PI converged in {} iterations'.format(pi_n_iter))
# print(pi_r)
#
# print('PI and VI converged to same policy? {}'
#       .format(np.all(policy == pi_policy)))

print('====== Running Tic Tac Toe =======')

print('\nValue Iteration')
P, R = ttt.getTransitionAndRewardArrays()
ttt_vi = mdp.ValueIteration(P, R, gamma)
ttt_vi.setVerbose()
ttt_vi.run()
print(f'MDP Toolbox VI finished in {ttt_vi.iter} iterations')
print(f'Accumulated reward: {len(ttt_vi.rewards)}')
print(f'Rewards: {ttt_vi.rewards}')

print('\nPolicy Iteration')
ttt_pi = mdp.PolicyIteration(P, R, gamma)
ttt_pi.setVerbose()
ttt_pi.run()
print(f'MDP Toolbox PI finished in {ttt_pi.iter} iterations')
print(f'Accumulated reward: {len(ttt_pi.rewards)}')
print(f'Rewards: {ttt_pi.rewards}')

print('\nQ-Learning')
ttt_q = mdp.QLearning(P, R, gamma)
