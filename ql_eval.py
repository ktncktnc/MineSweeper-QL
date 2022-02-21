import argparse
import cv2
from keras.models import load_model

from DQL.agent import MineSweeperAgent
from minesweeper import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a QL agent to play MineSweeper.')
    parser.add_argument("-l", '--level', help="Game level", default=1)
    parser.add_argument("-m", '--model', help="DQL link")
    parser.add_argument("-e", '--episode', help="Number of episodes", default=10)
    args = parser.parse_args()

    env = MineSweeper(level=int(args.level))

    agent = MineSweeperAgent("minesweeper-deepql_v1", board_size=env.board_size)
    agent.epsilon = 0
    agent.main_network.load_weights(args.model)
    agent.update_target_network()

    for e in range(int(args.episode)):
        env.reset()
        done = False

        i = 1
        while not done:
            state = env.game_board
            action = agent.act(state)

            observation, reward, done = env.step(action[0], action[1])

            cv2.imwrite('./result/{e}_{i}.png'.format(e=e, i=i), observation)
            i += 1
