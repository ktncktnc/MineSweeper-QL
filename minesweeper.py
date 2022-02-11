import cv2
import numpy as np


class MineSweeper:
    def __init__(self, level=3, unknow_cell=-1, mine_cell=-10, mark_cell=-20, is_view=True, image_size=720):
        self.level = level
        self.is_view = is_view
        self.mine_cell = mine_cell
        self.mark_cell = mark_cell
        self.unknow_cell = unknow_cell
        self.image_size = image_size
        self.window_name = 'minesweeper'
        self.mine_num_color = [(187, 186, 178), (226, 173, 93), (157, 179, 69), (63, 208, 244), (51, 118, 220), (38, 50, 169), (172, 68, 142)]
        self.init()

    def init(self):
        self.is_done = 0
        self.set_level(self.level)
        self.create_board(self.mine_cell)
        self.enable_click()

    def reset(self):
        self.init()

    def set_level(self, level):
        if not isinstance(level, int):
            level = 3

        if level <= 1:
            self.board_size = 10
            self.mine_num = int(self.board_size * self.board_size * 0.1)
        elif level == 2:
            self.board_size = 15
            self.mine_num = int(self.board_size * self.board_size * 0.15)
        else:
            self.board_size = 20
            self.mine_num = int(self.board_size * self.board_size * 0.2)

    def create_board(self, mine_cell=-10):
        board = np.zeros((self.board_size, self.board_size), int)
        x_indices = np.random.choice(range(self.board_size), size=self.mine_num)
        y_indices = np.random.choice(range(self.board_size), size=self.mine_num)
        board[x_indices, y_indices] = mine_cell

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] != mine_cell:
                    cell_board = self.get_nearby_cell(board, i, j)
                    cell_board = cell_board[cell_board < 0]
                    cell_mine_num = len(cell_board)
                    board[i, j] = cell_mine_num
        self.result_board = board

        play_board = np.full((self.board_size, self.board_size), self.unknow_cell)
        self.game_board = play_board

    def get_nearby_cell(self, board, i, j):
        def get_start_end_indices(indice):
            if indice == 0:
                i_start = 0
                i_end = indice + 2
            elif indice == self.board_size - 1:
                i_start = indice - 1
                i_end = indice + 1
            else:
                i_start = indice - 1
                i_end = indice + 2
            return i_start, i_end

        x_start, x_end = get_start_end_indices(i)
        y_start, y_end = get_start_end_indices(j)
        return board[x_start: x_end, y_start: y_end]

    def click_image(self, event, y, x, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cell_x = int(x / (self.image_size / self.board_size))
            cell_y = int(y / (self.image_size / self.board_size))
            self.step(cell_x, cell_y)
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.game_board[self.game_board == self.unknow_cell]) != self.board_size * self.board_size:
                cell_x = int(x / (self.image_size / self.board_size))
                cell_y = int(y / (self.image_size / self.board_size))
                self.mark(cell_x, cell_y)

    def enable_click(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_image)

    def step(self, x, y, mode='play'):
        self.check_step(x, y)
        result = self.do_step(x, y, mode=mode)
        done = self.check_result(result)
        reward = self.check_reward(result)
        observation = self.render(True)
        return observation, reward, done

    def check_step(self, x, y):
        if len(self.game_board[self.game_board == self.unknow_cell]) == self.board_size * self.board_size:
            while self.result_board[x, y] != 0:
                self.create_board(self.mine_cell)

    def check_reward(self, result):
        if result == 'over':
            return -100
        elif result == 'complete':
            return 100
        elif result == 'fail':
            return -10
        else:
            return 1

    def check_result(self, result):
        image = self.render(True)
        if result == 'fail' or result == 'success':
            if len(self.game_board[self.game_board == self.unknow_cell]) + len(self.game_board[self.game_board == self.mark_cell]) == self.mine_num:
                color = (226, 173, 93)
                text = 'Complete'
                self.is_done = 1
            else:
                self.is_done = 0
        else:
            if result == 'over':
                color = (53, 67, 203)
                text = 'Game over'
                self.is_done = 1

        if self.is_done:
            cv2.setMouseCallback(self.window_name, lambda *args: None)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 2, 4)[0]
            x_text = int((self.image_size - text_size[0]) / 2)
            y_text = int((self.image_size - text_size[1]) / 2)
            image = cv2.putText(image, text, (x_text, y_text), font, 2, color, 4)
            cv2.imshow(self.window_name, image)
            k = cv2.waitKey(10)

        return self.is_done

    def render(self, return_image=False, waitkey=10):
        if not self.is_view or self.is_done == 1:
            return False
        image = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        color = (160, 206, 125)
        thickness = 2
        for i in range(1, self.board_size):
            start_point = (int(self.image_size * i / self.board_size), 0)
            end_point = (int(self.image_size * i / self.board_size), self.image_size)
            image = cv2.line(image, start_point, end_point, color, thickness)
        for i in range(1, self.board_size):
            start_point = (0, int(self.image_size * i / self.board_size))
            end_point = (self.image_size, int(self.image_size * i / self.board_size))
            image = cv2.line(image, start_point, end_point, color, thickness)

        x_cell = self.image_size / self.board_size
        y_cell = self.image_size / self.board_size
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(self.board_size):
            for j in range(self.board_size):
                cell = self.game_board[i, j]
                if cell == self.unknow_cell:
                    continue

                if cell == self.mine_cell:
                    text = 'X'
                    color = (53, 67, 203)
                elif cell == self.mark_cell:
                    text = 'F'
                    color = (53, 67, 203)
                else:
                    text = str(cell)
                    color = self.mine_num_color[cell]

                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                x_text = int(x_cell * j + (x_cell - text_size[1]) / 2)
                y_text = int(y_cell * i + (y_cell + text_size[0]) / 2)
                image = cv2.putText(image, text, (x_text, y_text), font, 1, color, 2)
        if return_image:
            return image
        cv2.imshow(self.window_name, image)
        k = cv2.waitKey(waitkey)
        return True

    def mark(self, x, y):
        if x < 0 or x > self.board_size - 1:
            return 'fail'
        if y < 0 or y > self.board_size - 1:
            return 'fail'

        game_cell = self.game_board[x, y]

        if game_cell != self.unknow_cell:
            if game_cell == self.mark_cell:
                self.game_board[x, y] = self.unknow_cell
                return 'success'
            else:
                return 'fail'

        self.game_board[x, y] = self.mark_cell
        return 'success'

    def do_step(self, x, y, mode='play'):
        if x < 0 or x > self.board_size - 1:
            return 'fail'
        if y < 0 or y > self.board_size - 1:
            return 'fail'

        game_cell = self.game_board[x, y]
        if game_cell != self.unknow_cell:
            if mode == 'play':
                return 'fail'
            else:
                return 'success'
        result_cell = self.result_board[x, y]
        if result_cell == self.mine_cell and mode == 'play':
            self.game_board[x, y] = self.result_board[x, y]
            return 'over'
        if result_cell > 0:
            self.game_board[x, y] = self.result_board[x, y]
            return 'success'
        if result_cell == 0:
            self.game_board[x, y] = self.result_board[x, y]
            self.expand_zero_cell(x, y)
            return 'success'

    def expand_zero_cell(self, x, y):
        self.do_step(x - 1, y - 1, 'expand')
        self.do_step(x - 1, y, 'expand')
        self.do_step(x - 1, y + 1, 'expand')
        self.do_step(x, y - 1, 'expand')
        self.do_step(x, y + 1, 'expand')
        self.do_step(x + 1, y - 1, 'expand')
        self.do_step(x + 1, y, 'expand')
        self.do_step(x + 1, y + 1, 'expand')


if __name__ == '__main__':
    game = MineSweeper(3)
    x = game.board_size
    y = game.board_size

    done = 0
    while not done:

        # x_indice = np.random.choice(range(game.board_size))
        # y_indice = np.random.choice(range(game.board_size))
        # observation, reward, done = game.step(x_indice, y_indice)
        game.render()
        # print(x_indice, y_indice, reward, done)

        # cv2.imshow(game.window_name, observation)
        # cv2.waitKey(500)
