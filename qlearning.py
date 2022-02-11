import numpy as np
import argparse
from DQL.agent import MineSweeperAgent
from minesweeper import *

RENDER = False
RENDER_FRAME = 50
SAVE_TRAINING_FREQUENCY = 25
UPDATE_TARGET_MODEL_FREQUENCY = 10

def rand_action(board_size):
    # TODO: remove ....
    x_ind = np.random.choice(range(board_size))
    y_ind = np.random.choice(range(board_size))
    return np.array([x_ind, y_ind])


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description='Training a QL agent to play MineSweeper.')
    parser.add_argument("-l", '--level', help="Game level", default=1)
    args = parser.parse_args()

    # Env
    env = MineSweeper(level=int(args.level))
    init_state = env.game_board

    # Agent
    agent = MineSweeperAgent("minesweeper-deepql_v1", board_size=env.board_size)

    # States & actions size
    states_size = init_state.shape
    states_size = states_size[0] * states_size[1]
    actions_size = states_size

    # Parameters
    lr = 0.02
    gamma = 0.95
    # ----
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.005
    # ----
    n_episode = 10000
    n_iteration = 10000

    reward_episode = np.array([])

    for e in range(n_episode):
        print("Episode {i}/{n} e = {ep}".format(i=e, n=n_episode, ep=epsilon))

        # Reset env
        env.reset()
        state = env.game_board

        # Episode parameters
        done = False
        total_reward = 0

        for i in range(n_iteration):
            # Render
            if RENDER and i % RENDER_FRAME == 0:
                env.render(waitkey=10)

            # Action
            action = agent.act(state)
            observation, reward, done = env.step(action[0], action[1])

            next_state = env.game_board
            agent.memorize(state, action, reward, next_state, done)

            if len(agent.memory) > agent.batch_size:
                agent.replay()

            total_reward += reward

            if done:
                break

            # Next state
            state = next_state

            if i % 100 == 0:
                print("a")

        # Append reward
        reward_episode = np.append(reward_episode, total_reward)

        if e % UPDATE_TARGET_MODEL_FREQUENCY:
            agent.update_target_network()

        if e % SAVE_TRAINING_FREQUENCY:
            agent.save("./saved/trial_{}.h5".format(e))








