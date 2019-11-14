import gym
from gym import wrappers
from mdptoolbox import mdp
import numpy as np
from A4 import policy_iteration, value_iteration


print('==== Running FrozenLake ====')
gamma = 1.0
fl = gym.make('FrozenLake8x8-v0')
print('Starting value iteration')
policy, V, n_iter = value_iteration(fl, theta=1e-8, discount_factor=gamma)
print('Converged in {} iterations'.format(n_iter))
print(V.shape)

print('\nStarting policy iteration')
pi_policy, pi_V, pi_n_iter = policy_iteration(fl, discount_factor=gamma)
print('PI converged in {} iterations'.format(pi_n_iter))
print(pi_V)

print('PI and VI converged to same policy? {}'
      .format(np.all(policy == pi_policy)))