# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import dok_matrix
import random
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark, O_REWARD, X_REWARD
from collections import defaultdict
from tqdm import tqdm
import itertools


"""
Define Tic Tac Toe as an MDP. Taken from mdptoolbox example

Source:
    https://github.com/sawcordwell/pymdptoolbox/blob/master/src/examples/tictactoe.py
"""

ACTIONS = 9
STATES = 3**ACTIONS
PLAYER = 1
OPPONENT = 2
WINS = ([1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0])

# The valid number of cells belonging to either the player or the opponent:
# (player, opponent)
OWNED_CELLS = ((0, 0),
               (1, 1),
               (2, 2),
               (3, 3),
               (4, 4),
               (0, 1),
               (1, 2),
               (2, 3),
               (3, 4))


def convertIndexToTuple(state):
    """
    Convert index in range 0 to STATE to game board state (0-2 per square)
    :param state:
    :return:
    """
    return(tuple(int(x) for x in np.base_repr(state, 3, 9)[-9::]))


def convertTupleToIndex(state):
    """"""
    return(int("".join(str(x) for x in state), 3))


def getLegalActions(state):
    """"""
    return(tuple(x for x in range(ACTIONS) if state[x] == 0))


def getTransitionAndRewardArrays():
    """"""
    P = [dok_matrix((STATES, STATES)) for a in range(ACTIONS)]
    #R = spdok((STATES, ACTIONS))
    R = np.zeros((STATES, ACTIONS))
    # Naive approach, iterate through all possible combinations
    for a in range(ACTIONS):
        for s in range(STATES):
            state = convertIndexToTuple(s)
            if not isValid(state):
                # There are no defined moves from an invalid state, so
                # transition probabilities cannot be calculated. However,
                # P must be a square stochastic matrix, so assign a
                # probability of one to the invalid state transitioning
                # back to itself.
                P[a][s, s] = 1
                # Reward is 0
            else:
                s1, p, r = getTransitionProbabilities(state, a)
                P[a][s, s1] = p
                R[s, a] = r
        P[a] = P[a].tocsr()
    #R = R.tolil()
    return(P, R)


def getTransitionProbabilities(state, action):
    """
    Parameters
    ----------
    state : tuple
        The state
    action : int
        The action

    Returns
    -------
    s1, p, r : tuple of two lists and an int
        s1 are the next states, p are the probabilities, and r is the reward

    """
    #assert isValid(state)
    assert 0 <= action < ACTIONS
    if not isLegal(state, action):
        # If the action is illegal, then transition back to the same state but
        # incur a high negative reward
        s1 = [convertTupleToIndex(state)]
        return(s1, [1], -10)
    # Update the state with the action
    state = list(state)
    state[action] = PLAYER
    if isWon(state, PLAYER):
        # If the player's action is a winning move then transition to the
        # winning state and receive a reward of 1.
        s1 = [convertTupleToIndex(state)]
        return(s1, [1], 1)
    elif isDraw(state):
        s1 = [convertTupleToIndex(state)]
        return(s1, [1], 0)
    # Now we search through the opponents moves, and calculate transition
    # probabilities based on maximising the opponents chance of winning..
    s1 = []
    p = []
    legal_a = getLegalActions(state)
    for a in legal_a:
        state[a] = OPPONENT
        # If the opponent is going to win, we assume that the winning move will
        # be chosen:
        if isWon(state, OPPONENT):
            s1 = [convertTupleToIndex(state)]
            return(s1, [1], -1)
        elif isDraw(state):
            s1 = [convertTupleToIndex(state)]
            return(s1, [1], 0)
        # Otherwise we assume the opponent will select a move with uniform
        # probability across potential moves:
        s1.append(convertTupleToIndex(state))
        p.append(1.0 / len(legal_a))
        state[a] = 0
    # During non-terminal play states the reward is 0.
    return(s1, p, 0)


def getReward(state, action):
    """"""
    if not isLegal(state, action):
        return -100
    state = list(state)
    state[action] = PLAYER
    if isWon(state, PLAYER):
        return 1
    elif isWon(state, OPPONENT):
        return -1
    else:
        return 0


def isDraw(state):
    """"""
    try:
        state.index(0)
        return False
    except ValueError:
        return True


def isLegal(state, action):
    """"""
    if state[action] == 0:
        return True
    else:
        return False


def isWon(state, who):
    """Test if a tic-tac-toe game has been won.

    Assumes that the board is in a legal state.
    Will test if the value 1 is in any winning combination.

    """
    for w in WINS:
        S = sum(1 if (w[k] == 1 and state[k] == who) else 0
                for k in range(ACTIONS))
        if S == 3:
            # We have a win
            return True
    # There were no wins so return False
    return False


def isValid(state):
    """"""
    # S1 is the sum of the player's cells
    S1 = sum(1 if x == PLAYER else 0 for x in state)
    # S2 is the sum of the opponent's cells
    S2 = sum(1 if x == OPPONENT else 0 for x in state)
    if (S1, S2) in OWNED_CELLS:
        return True
    else:
        return False


##############################################################################

##############################################################################


"""
Open AI Gym Tic Tac Toe agents
Based on examples from:
    https://github.com/haje01/gym-tictactoe/blob/master/examples/base_agent.py
"""


class RandomAgent(object):
    """
    Random tic tac toe agent. Will select random actions
    """
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action
        return random.choice(ava_actions)

    def update(self, state, nstate, action, reward, done):
        pass


st_values = {}
st_visits = defaultdict(lambda: 0)


def reset_state_values():
    global st_values, st_visits
    st_values = {}
    st_visits = defaultdict(lambda: 0)


def set_state_value(state, value):
    st_visits[state] += 1
    st_values[state] = value


def best_val_indices(values, fn):
    best = fn(values)
    return [i for i, v in enumerate(values) if v == best]


class TDAgent(object):
    """
    TD(0) learning agent for tic tac toe
    """
    def __init__(self, mark, epsilon, epsilon_decay,
                 alpha, alpha_decay, gamma):
        self.mark = mark
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma

        # stats
        self.rewards = []

    def act(self, state, ava_actions):
        return self.epsilon_greedy_policy(state, ava_actions)

    def epsilon_greedy_policy(self, state, ava_actions):
        """
        Returns action by Epsilon greedy policy.
            Args:
                state (tuple): Board status + mark
                ava_actions (list): Available actions
            Returns:
                int: Selected action.
        """
        e = random.random()
        if e < self.epsilon:
            # random exploration
            action = random.choice(ava_actions)
        else:
            # exploitation
            action = self.greedy_action(state, ava_actions)
        return action

    def greedy_action(self, state, ava_actions):
        """Return best action by current state value.
        Evaluate each action, select best one. Tie-breaking is random.
        Args:
            state (tuple): Board status + mark
            ava_actions (list): Available actions
        Returns:
            int: Selected action
        """
        assert len(ava_actions) > 0

        ava_values = []
        for action in ava_actions:
            nstate = after_action_state(state, action)
            nval = self.ask_value(nstate)
            ava_values.append(nval)
            vcnt = st_visits[nstate]
            # print("  nstate {} val {:0.2f} visits {}".
            #       format(nstate, nval, vcnt))

        # select most right action for 'O' or 'X'
        if self.mark == 'O':
            indices = best_val_indices(ava_values, max)
        else:
            indices = best_val_indices(ava_values, min)

        # tie breaking by random choice
        aidx = random.choice(indices)
        # print("greedy_action mark {} ava_values {} indices {} aidx {}".
        #       format(self.mark, ava_values, indices, aidx))

        action = ava_actions[aidx]

        return action

    def ask_value(self, state):
        """Returns value of given state.
        If state is not exists, set it as default value.
        Args:
            state (tuple): State.
        Returns:
            float: Value of a state.
        """
        if state not in st_values:
            gstatus = check_game_status(state[0])
            val = 0
            # win
            if gstatus > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[state]

    def update(self, state, nstate, action, reward, done):
        """Backup value by difference and step size.
        Execute an action then backup Q by best value of next state.
        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        # print("backup state {} nstate {} reward {}".
        #       format(state, nstate, reward))

        val = self.ask_value(state)
        nval = self.ask_value(nstate)
        diff = nval - val
        val2 = val + self.alpha * diff

        # print("  value from {:0.2f} to {:0.2f}".format(val, val2))
        set_state_value(state, val2)


class QAgent(object):
    """
    Tic Tac Toe Q Learner

    Based on:
        https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e
    """
    def __init__(self, n_states, n_actions,
                 mark, epsilon, epsilon_decay,
                 alpha, alpha_decay, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.mark = mark
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma

        # stats
        self.episode_rewards = []

        # allocate Q table
        self.Q = np.ones((STATES, n_actions))

    def act(self, state, ava_actions):
        if np.random.random() < self.epsilon:
            # explore
            action = random.choice(ava_actions)
        else:
            s = convertTupleToIndex(state[0])
            tQ = np.copy(self.Q[s, :])
            invalid_actions = list(set(range(self.n_actions)) - set(ava_actions))
            tQ[invalid_actions] = -np.inf
            # remove invalid actions
            action = np.argmax(tQ)
        return action

    def update(self, state, nstate, action, reward, done=False):
        s = convertTupleToIndex(state)
        ns = convertTupleToIndex(nstate)
        if done:
            self.Q[s, action] += self.alpha * (reward - self.Q[s, action])
            self.episode_rewards.append(reward)
        else:
            expected_value = np.mean(self.Q[ns, :])
            self.Q[s, action] += self.alpha * (reward + (self.gamma * expected_value) - self.Q[s, action])

        # decay alpha and epsilon
        self.alpha = max(self.alpha * self.alpha_decay, 0.01)
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

    def policy(self):
        policy = np.zeros((self.Q.shape[0], 2))
        policy[:, 0] = range(self.Q.shape[0])
        policy[:, 1] = np.argmax(self.Q, axis=1)
        return policy


def train_agents(opponent,
                 max_episode,
                 epsilon,
                 epsilon_decay,
                 alpha,
                 alpha_decay,
                 gamma,
                 render=False):
    reset_state_values()

    env = TicTacToeEnv()
    if opponent == 'random':
        agents = [QAgent(env.observation_space.n,
                         env.action_space.n,
                         'O', epsilon, epsilon_decay,
                         alpha, alpha_decay, gamma),
                  RandomAgent('X')]
    else:  # Two Q agents
        agents = [
            QAgent(env.observation_space.n,
                   env.action_space.n,
                   'O',
                   epsilon, epsilon_decay,
                   alpha, alpha_decay, gamma),
            QAgent(env.observation_space.n,
                   env.action_space.n,
                   'X',
                   epsilon, epsilon_decay,
                   alpha, alpha_decay, gamma)]

    start_mark = 'O'
    agent_rewards = {'O': [], 'X': []}
    episode = 0
    for i in tqdm(range(max_episode)):
        episode += 1
        env.show_episode(False, episode)

        # reset agent for new episode
        for agent in agents:
            agent.episode_rate = episode / float(max_episode)

        env.set_start_mark(start_mark)
        state = env.reset()
        s, mark = state
        done = False
        while not done:
            if render:
                env.render()
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            env.show_turn(False, mark)
            action = agent.act(state, ava_actions)

            # update (no rendering)
            nstate, reward, done, info = env.step(action)
            agent.update(s, nstate[0], action, reward, done)

            if done:
                if render:
                    env.render()
                env.show_result(render, mark, reward)
                # set terminal state value
                set_state_value(state, reward)
                agent_rewards['O'].append(reward)
                agent_rewards['X'].append(-reward)

            s, mark = state = nstate

        # rotate start
        start_mark = next_mark(start_mark)

    return agent_rewards, agent_by_mark(agents, 'O')


def play_game(qagent):
    env = TicTacToeEnv()
    opponent = RandomAgent('X')
    start_mark = 'O'
    env.set_start_mark(start_mark)
    state = env.reset()
    s, mark = state
    done = False
    agents = [qagent, opponent]
    while not done:
        env.render()
        agent = agent_by_mark(agents, mark)
        ava_actions = env.available_actions()
        env.show_turn(False, mark)
        action = agent.act(state, ava_actions)

        nstate, reward, done, info = env.step(action)
        print(f'state: {s}, action: {action}')
        if done:
            env.render()
            env.show_result(True, mark, reward)

        s, mark = state = nstate
