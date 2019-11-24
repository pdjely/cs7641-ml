import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm


def custom_frozen_lake(size=8, p=0.8, nondeterministic=False):
    """
    Create a custom-sized frozen-lake environment

    :param size: size x size lake
    :param p: probability of creating frozen tile
    :return: environment

    based on:
        https://reinforcementlearning4.fun/2019/06/24/create-frozen-lake-random-maps/
    """
    random_map = generate_random_map(size=size, p=p)
    fl = gym.make('FrozenLake-v0', desc=random_map, is_slippery=nondeterministic)
    return fl


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.

    Stolen from:
        https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    """
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    n_iter = 0
    max_iter = 5000
    accum_reward = []
    delta_history = []
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
            # Check if we can stop
        n_iter += 1
        accum_reward.append(np.max(V))
        delta_history.append(delta)
        if delta < theta or n_iter > max_iter:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V, n_iter, accum_reward, delta_history


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.

    Stolen from:
        https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    delta_history = []
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            delta_history.append(delta)
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V), np.mean(delta_history)


def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    Stolen from:
        https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    n_iter = 0
    max_iter = 500
    accum_reward = []
    delta_history = []
    while True:
        # Evaluate the current policy
        V, delta = policy_eval_fn(policy, env, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        n_diff = 0
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
                n_diff += 1
            policy[s] = np.eye(env.nA)[best_a]
        delta_history.append(n_diff)
        n_iter += 1
        accum_reward.append(np.max(V))

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable or n_iter > max_iter:
            return policy, V, n_iter, accum_reward, delta_history


"""
Qlearning is an off policy learning python implementation.
This is a python implementation of the qlearning algorithm in the Sutton and
Barto's book on RL. Modified to use Q learning instead of SARSA
Using the simplest gym environment for brevity: https://gym.openai.com/envs/FrozenLake-v0/


"""

def init_q(s, a, type="ones"):
    """
    @param s the number of states
    @param a the number of actions
    @param type random, ones or zeros for the initialization
    """
    if type == "ones":
        return np.ones((s, a))
    elif type == "random":
        return np.random.random((s, a))
    elif type == "zeros":
        return np.zeros((s, a))


def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param n_actions number of actions
    @param s current state
    @param train if true then no random actions selected
    """
    if np.random.rand() > epsilon:
        action = np.argmax(Q[s, :])
    else:
        # print('explore')
        action = np.random.randint(0, n_actions)
    return action


def sarsa(env,
          alpha, alpha_decay,
          gamma,
          epsilon, epsilon_decay,
          episodes,
          max_steps,
          n_tests,
          render = False,
          test=False):
    """
    SARSA TD learner. Also has Q learning but that totally failed.

    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes

    Based on:
        https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e
    """
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_q(n_states, n_actions, type="random")
    timestep_reward = []
    for episode in tqdm(range(episodes)):
        # print(f"Episode: {episode}")
        s = env.reset()
        t = 0
        total_reward = 0
        done = False
        a = epsilon_greedy(Q, epsilon, n_actions, s)
        while not done:
            # choose a from S using epsilon greedy
            if render:
                env.render()
            # print(f'action: {a}')
            t += 1
            # take action a, obeserve new state s_
            s_, reward, done, info = env.step(a)
            total_reward += reward

            # SARSA, on policy learning
            a_ = epsilon_greedy(Q, epsilon, n_actions, s_)
            if done:
                Q[s, a] += alpha * (reward - Q[s, a])
                # print(Q)
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_]) - Q[s, a])

            s, a = s_, a_
            if done:
                # print(f'reward: {reward}')
                if render:
                    print(f"This episode took {t} timesteps and reward: {total_reward}")
                timestep_reward.append(total_reward)

        # Decay alpha and epsilon
        alpha = max(0.05, alpha * alpha_decay)
        epsilon = max(0.05, epsilon * epsilon_decay)

    return np.array(timestep_reward), Q


def qlearning(env,
              alpha, alpha_decay,
              gamma,
              epsilon, epsilon_decay,
              episodes,
              max_steps,
              n_tests,
              render = False,
              test=False):
    """
    Q learner. Doesn't work because of reward structuring

    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes

    Based on:
        https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e
    """
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_q(n_states, n_actions, type="ones")
    timestep_reward = []
    for episode in tqdm(range(episodes)):
        # print(f"Episode: {episode}")
        s = env.reset()
        t = 0
        total_reward = 0
        done = False
        while not done:
            # choose a from S using epsilon greedy
            if render:
                env.render()
            a = epsilon_greedy(Q, epsilon, n_actions, s)
            # print(f'action: {a}')
            t += 1
            # take action a, obeserve new state s_
            s_, reward, done, info = env.step(a)
            total_reward += reward

            # Update Q table
            Q[s, a] += alpha * (reward + gamma * np.max(Q[s_, a]) - Q[s, a])
            s = s_
            if done:
                # print(f'reward: {reward}')
                if render:
                    print(f"This episode took {t} timesteps and reward: {total_reward}")
                timestep_reward.append(total_reward)

        # Decay alpha and epsilon
        alpha = max(0.05, alpha * alpha_decay)
        epsilon = max(0.05, epsilon * epsilon_decay)

    return np.array(timestep_reward), Q


