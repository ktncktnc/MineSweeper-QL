import argparse
from tqdm import tqdm
from keras.models import load_model
from DQL.agent_them import MineSweeperAgent
from minesweeper_env import *


def parse_args():
    parser = argparse.ArgumentParser(description='Play Minesweeper online using a DQN')
    parser.add_argument('--model', type=str, default='saved/trial_weight_l9_2000',
                        help='name of model')
    parser.add_argument('--episodes', type=int, default=4,
                        help='Number of episodes to play')

    return parser.parse_args()


params = parse_args()

def main():
    env = MinesweeperEnv(9, 9, 10)
    agent = MineSweeperAgent("minesweeper-deepql_v1", board_size=9)
    agent.main_network.load_weights(f'{params.model}.h5')
    agent.epsilon = 0

    output_file = open("result.txt", "a+")

    for episode in tqdm(range(1, params.episodes + 1)):
        env.reset()
        output_file.write("Episode {e}\n".format(e=episode))
        done = False
        while not done:
            current_state = env.state_im
            output_file.writelines(str((current_state[:, :, 0]*8).astype('int32')))
            output_file.write("\n")
            action = agent.act(current_state)

            new_state, reward, done = env.step(action)

        output_file.writelines(str((new_state[:, :, 0] * 8).astype('int32')))

        output_file.write("\n")


if __name__ == "__main__":
    main()
