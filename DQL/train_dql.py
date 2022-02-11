from DQL.agent import MineSweeperAgent
import argparse

RENDER = True
SAVE_TRAINING_FREQUENCY = 25
UPDATE_TARGET_MODEL_FREQUENCY = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="aaa")
    parser.add_argument('-l', '--level')

    args = parser.parse_args()

