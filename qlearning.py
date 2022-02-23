import argparse

import numpy as np

from DQL.agent import MineSweeperAgent
from minesweeper import *
from tqdm import tqdm
from DQL.utils import *

RENDER = False
RENDER_FRAME = 50
SAVE_TRAINING_FREQUENCY = 3000
UPDATE_TARGET_MODEL_FREQUENCY = 10
PRINT_LOG_FREQUENCY = 100
TRAINING_FREQUENCY = 5


def main():
    # Args
    parser = argparse.ArgumentParser(description='Training a QL agent to play MineSweeper.')
    parser.add_argument("-l", '--level', help="Game level", default=1)
    args = parser.parse_args()

    # Env
    env = MineSweeper(level=int(args.level))
    init_state = env.game_board

    # Agent
    agent = MineSweeperAgent("minesweeper-deepql_v1", board_size=env.board_size)
    #agent.main_network.load_weights("./saved/trial_weight_l10_1000.h5")

    # States & actions size
    states_size = init_state.shape
    states_size = states_size[0] * states_size[1]
    actions_size = states_size

    # Parameters
    n_episode = 100000
    n_iteration = 1000

    reward_episode = np.array([])
    n_opens_episode = np.array([])
    n_wins_episode = np.array([])
    past_n_wins = 0
    for e in tqdm(range(1, n_episode+1)):

        # Reset env
        env.reset()

        # Episode parameters
        done = False
        total_reward = 0
        #print("Episode")

        for i in range(n_iteration):
            state = env.game_board/8.0
            action = agent.act(state)

            observation, reward, done = env.step(action[0], action[1])
            next_state = env.game_board/8.0

            agent.memorize(state, action, reward, next_state, done)

            if len(agent.memory) > 1000:
                agent.replay()

            total_reward += reward
            if done:
                n_opens_episode = np.append(n_opens_episode, i + 1)
                break

        # Append reward
        reward_episode = np.append(reward_episode, total_reward)


        if not e % PRINT_LOG_FREQUENCY:
            print("Episode {i}/{n} e = {ep} Total Reward = {r:.2} average open = {o:.2} wins = {w:.2}".format(i=e, n=n_episode,
                                                                                                     ep=agent.epsilon,
                                                                                                     r=np.average(
                                                                                                         reward_episode[
                                                                                                         -PRINT_LOG_FREQUENCY:]),
                                                                                                     o=np.average(
                                                                                                         n_opens_episode[
                                                                                                         -PRINT_LOG_FREQUENCY:]),
                                                                                                     w=(env.n_wins-past_n_wins)/PRINT_LOG_FREQUENCY
                                                                                                     ))

            past_n_wins = env.n_wins

        if not e % UPDATE_TARGET_MODEL_FREQUENCY:
            agent.update_target_network()

        if not e % SAVE_TRAINING_FREQUENCY:
            agent.save("./saved/trial_weight_l{l}_{e}.h5".format(e=e, l=agent.board_size))


if __name__ == '__main__':
    main()
