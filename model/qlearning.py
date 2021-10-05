import numpy as np
import math
import copy

def qlearn(env, num_iters, alpha, gamma, epsilon, max_steps, use_softmax_policy,
           init_beta=None, k_exp_sched=None):
    """ Runs tabular Q learning algorithm for stochastic environment.

    Args:
        env: instance of environment object
        num_iters (int): Number of episodes to run Q-learning algorithm
        alpha (float): The learning rate between [0,1]
        gamma (float): Discount factor, between [0,1)
        epsilon (float): Probability in [0,1] that the agent selects a random
                         move instead of selecting greedily from Q value
        max_steps (int): Maximum number of steps in the environment per episode
        use_softmax_policy (bool): Whether to use softmax policy (True) or
                                   Epsilon-Greedy (False)
        init_beta (float): If using stochastic policy, sets the initial beta as
                           the parameter for the softmax
        k_exp_sched (float): If using stochastic policy, sets hyperparameter
                             for exponential schedule on beta

    Returns:
        q_hat: A Q-value table shaped [num_states, num_actions] for environment
               with with num_states number of states (e.g. num rows * num
               columns for grid) and num_actions number of possible actions
               (e.g. 4 actions up/down/left/right)
        steps_vs_iters: An array of size num_iters. Each element denotes the
                        number of steps in the environment that the agent took
                        to get to the goal (capped to max_steps)
    """
    action_space_size = env.num_actions
    state_space_size = env.num_states
    q_hat = np.zeros(shape=(state_space_size, action_space_size))
    steps_vs_iters = np.zeros(num_iters)

    for i in range(num_iters):
        curr_state = env.reset()
        num_steps = 0
        done = False

        while (num_steps < max_steps) and (not done):
            num_steps += 1

            if use_softmax_policy:
                if all(np.unique(q_hat[curr_state, :]) == 0):
                    action = np.random.choice(range(action_space_size))
                else:
                    assert(init_beta is not None)
                    assert(k_exp_sched is not None)
                    beta = beta_exp_schedule(init_beta, num_iters, k_exp_sched)
                    action = softmax_policy(q_hat, beta, curr_state)
            else:
                action = epsilon_greedy(q_hat, epsilon, curr_state,
                                        action_space_size)

            next_state, reward, done = env.step(action)

            if next_state != curr_state:
                new_value = reward + gamma * np.max(q_hat[next_state, :])
                q_hat[curr_state, action] += alpha * (new_value -
                                                      q_hat[curr_state, action])
                curr_state = next_state

        steps_vs_iters[i] = num_steps

    return q_hat, steps_vs_iters


def epsilon_greedy(q_hat, epsilon, state, action_space_size):
    """ Chooses a random action with p_rand_move probability,
    otherwise choose the action with highest Q value for
    current observation

    Args:
        q_hat: A Q-value table shaped [num_rows, num_col, num_actions] for
            grid environment with num_rows rows and num_col columns and num_actions
            number of possible actions
        epsilon (float): Probability in [0,1] that the agent selects a random
            move instead of selecting greedily from Q value
        state: A 2-element array with integer element denoting the row and column
            that the agent is in
        action_space_size (int): number of possible actions

    Returns:
        action (int): A number in the range [0, action_space_size-1]
            denoting the action the agent will take
    """
    uniform_sample = np.random.uniform(0, 1, 1)

    if (uniform_sample <= epsilon) or all(np.unique(q_hat[state, :]) == 0):
        action = np.random.choice(range(action_space_size))
    else:
        action = np.argmax(q_hat[state, :])

    return action


def softmax_policy(q_hat, beta, state):
    """ Choose action using policy derived from Q, using
    softmax of the Q values divided by the temperature.

    Args:
        q_hat: A Q-value table shaped [num_rows, num_col, num_actions] for
            grid environment with num_rows rows and num_col columns
        beta (float): Parameter for controlling the stochasticity of the action
        obs: A 2-element array with integer element denoting the row and column
            that the agent is in

    Returns:
        action (int): A number in the range [0, action_space_size-1]
            denoting the action the agent will take
    """
    probs = stable_softmax(beta * q_hat, axis=1)
    action = np.random.choice([0, 1, 2, 3], p=probs[state])

    return action


def beta_exp_schedule(init_beta, iteration, k=0.1):
   beta = init_beta * np.exp(k * iteration)

   return beta


def stable_softmax(x, axis=2):
    """ Numerically stable softmax:
    softmax(x) = e^x /(sum(e^x))
               = e^x / (e^max(x) * sum(e^x/e^max(x)))

    Args:
        x: An N-dimensional array of floats
        axis: The axis for normalizing over.

    Returns:
        output: softmax(x) along the specified dimension
    """
    max_x = np.max(x, axis, keepdims=True)
    z = np.exp(x - max_x)
    output = z / np.sum(z, axis, keepdims=True)

    return output

