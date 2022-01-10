import numpy as np
import pygame
from mino import *
from random import *
from pygame.locals import *

pygame.init()

MOVEMENT = [
    'NOOP',
    'Turn Left',
    'Turn Right',
    'Move Left',
    'Move Right',
]


class ui_variables:
    # Fonts
    font_path = "./assets/fonts/OpenSans-Light.ttf"
    font_path_b = "./assets/fonts/OpenSans-Bold.ttf"
    font_path_i = "./assets/fonts/Inconsolata/Inconsolata.otf"

    h1 = pygame.font.Font(font_path, 50)
    h2 = pygame.font.Font(font_path, 30)
    h4 = pygame.font.Font(font_path, 20)
    h5 = pygame.font.Font(font_path, 13)
    h6 = pygame.font.Font(font_path, 10)

    h1_b = pygame.font.Font(font_path_b, 50)
    h2_b = pygame.font.Font(font_path_b, 30)

    h2_i = pygame.font.Font(font_path_i, 30)
    h5_i = pygame.font.Font(font_path_i, 13)

    # Background colors
    black = (10, 10, 10)  # rgb(10, 10, 10)
    white = (255, 255, 255)  # rgb(255, 255, 255)
    grey_1 = (26, 26, 26)  # rgb(26, 26, 26)
    grey_2 = (35, 35, 35)  # rgb(35, 35, 35)
    grey_3 = (55, 55, 55)  # rgb(55, 55, 55)

    # Tetrimino colors
    cyan = (69, 206, 204)  # rgb(69, 206, 204) # I
    blue = (64, 111, 249)  # rgb(64, 111, 249) # J
    orange = (253, 189, 53)  # rgb(253, 189, 53) # L
    yellow = (246, 227, 90)  # rgb(246, 227, 90) # O
    green = (98, 190, 68)  # rgb(98, 190, 68) # S
    pink = (242, 64, 235)  # rgb(242, 64, 235) # T
    red = (225, 13, 27)  # rgb(225, 13, 27) # Z

    t_color = [grey_2, cyan, blue, orange, yellow, green, pink, red, grey_3]


#  想办法修改成一个能用的俄罗斯方块！
#  1、增加接口。输出和gym-tetris类似的数据
#  2、更好的盘面。空洞0，已放置1，正在下落2
#  3、能够自主调节速度。训练的时候快速，测试的时候慢速
#  4、自动下落，不支持加速下落。这样学习可能更专心。
#  5、能够关闭界面显示，直接在终端运行计算结果（这一步待定）。
#  6、还需要想办法把正在下落的方块以不同的形式表示出来！！！
# FIXME 中间有很多关于matrix越界的问题，我觉得是碰撞检测、边界判断没做好，需要修改中间的部分函数！
class Tetris(object):
    def __init__(self, speed=15, levelup=False):
        # Define
        self.block_size = 17  # Height, width of single block
        self.width = 10  # Board width
        self.height = 20  # Board height
        self.speed = speed  # 保存设置的速度
        self.framerate = self.speed  # Bigger -> Slower
        # self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((300, 374))
        pygame.time.set_timer(pygame.USEREVENT, self.framerate)
        pygame.display.set_caption("TETRIS")

        # Initial values
        self.blink = False
        self.start = False
        self.pause = False
        self.done = False
        self.game_over = False

        self.score = 0
        self.level = 1
        self.levelup = levelup  # 是否等级提升
        self.goal = self.level * 5
        self.bottom_count = 0
        self.hard_drop = False

        # TODO 这里应该是表示当前方块的左上角位置！
        #  所以应该加上一个限制：dx>=0,dx<=6;dy>=0,dy<=17
        self.dx, self.dy = 3, 0  # Minos location status
        self.rotation = 0  # Minos rotation status

        self.mino = randint(1, 7)  # Current mino
        self.next_mino = randint(1, 7)  # Next mino

        self.hold = False  # Hold status
        self.hold_mino = -1  # Holded mino

        self.name_location = 0
        self.name = [65, 65, 65]

        self.matrix = [[0 for _ in range(self.height + 1)] for _ in range(self.width)]  # Board matrix

    # Draw block
    def __draw_block__(self, x, y, color):
        pygame.draw.rect(
            self.screen,
            color,
            Rect(x, y, self.block_size, self.block_size)
        )
        pygame.draw.rect(
            self.screen,
            ui_variables.grey_1,
            Rect(x, y, self.block_size, self.block_size),
            1
        )

    # Draw game screen
    def __draw_board__(self):
        # self.next_mino, self.hold_mino, self.score, self.level, self.goal
        self.screen.fill(ui_variables.grey_1)

        # Draw sidebar
        pygame.draw.rect(
            self.screen,
            ui_variables.white,
            Rect(204, 0, 96, 374)
        )

        # Draw next_mino mino
        grid_n = tetris_mimo.mino_map[self.next_mino - 1][0]

        for i in range(4):
            for j in range(4):
                dx = 220 + self.block_size * j
                dy = 140 + self.block_size * i
                if grid_n[i][j] != 0:
                    pygame.draw.rect(
                        self.screen,
                        ui_variables.t_color[grid_n[i][j]],
                        Rect(dx, dy, self.block_size, self.block_size)
                    )

        # Draw hold_mimo mino
        grid_h = tetris_mimo.mino_map[self.hold_mino - 1][0]

        if self.hold_mino != -1:
            for i in range(4):
                for j in range(4):
                    dx = 220 + self.block_size * j
                    dy = 50 + self.block_size * i
                    if grid_h[i][j] != 0:
                        pygame.draw.rect(
                            self.screen,
                            ui_variables.t_color[grid_h[i][j]],
                            Rect(dx, dy, self.block_size, self.block_size)
                        )

        # Set max score
        if self.score > 999999:
            self.score = 999999

        # Draw texts
        # text_hold = ui_variables.h5.render("HOLD", 1, ui_variables.black)
        text_next = ui_variables.h5.render("NEXT", 1, ui_variables.black)
        text_score = ui_variables.h5.render("SCORE", 1, ui_variables.black)
        score_value = ui_variables.h4.render(str(self.score), 1, ui_variables.black)
        text_level = ui_variables.h5.render("LEVEL", 1, ui_variables.black)
        level_value = ui_variables.h4.render(str(self.level), 1, ui_variables.black)
        text_goal = ui_variables.h5.render("GOAL", 1, ui_variables.black)
        goal_value = ui_variables.h4.render(str(self.goal), 1, ui_variables.black)

        # Place texts
        # self.screen.blit(text_hold, (215, 14))
        self.screen.blit(text_next, (215, 104))
        self.screen.blit(text_score, (215, 194))
        self.screen.blit(score_value, (220, 210))
        self.screen.blit(text_level, (215, 254))
        self.screen.blit(level_value, (220, 270))
        self.screen.blit(text_goal, (215, 314))
        self.screen.blit(goal_value, (220, 330))

        # Draw board
        for x in range(self.width):
            for y in range(self.height):
                dx = 17 + self.block_size * x
                dy = 17 + self.block_size * y
                self.__draw_block__(dx, dy, ui_variables.t_color[self.matrix[x][y + 1]])

    # Draw a tetris_mimo
    def __draw_mino__(self):
        # self.dx, self.dy, self.mino, self.rotation
        self.rotation = self.rotation % 4
        grid = tetris_mimo.mino_map[self.mino - 1][self.rotation]

        tx, ty = self.dx, self.dy
        while not self.__is_bottom__(tx, ty, self.mino, self.rotation):
            ty += 1

        # Draw ghost
        # for i in range(4):
        #     for j in range(4):
        #         if grid[i][j] != 0:
        #             self.matrix[tx + j][ty + i] = 8

        # Draw mino
        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    # FIXME 这里也发生过越界！！！问题太多了！！！
                    # if self.dx + j > 9 or self.dy + i > 20:
                    #     continue
                    self.matrix[self.dx + j][self.dy + i] = grid[i][j]

    # Erase a tetris_mimo
    def __erase_mino__(self):
        # self.dx, self.dy, self.mino, self.rotation
        grid = tetris_mimo.mino_map[self.mino - 1][self.rotation]

        # Erase ghost
        # for j in range(21):
        #     for i in range(10):
        #         if self.matrix[i][j] == 8:
        #             self.matrix[i][j] = 0

        # Erase mino
        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    self.matrix[self.dx + j][self.dy + i] = 0

    # Returns true if mino is at bottom
    def __is_bottom__(self, x, y, mino, r):
        grid = tetris_mimo.mino_map[mino - 1][r]

        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    if (y + i + 1) > 20:
                        return True
                    # FIXME 这里也发生过越界！！！
                    #  我发现越界是因为x+j=10导致的，目前不清楚为什么会这样
                    #  如果用return True避免越界好像会导致另外一个地方越界。。。先用continue填坑看看
                    # elif (x + j) > 9:
                    #     continue
                    # return True
                    # elif self.matrix[x + j][y + i + 1] != 0 and self.matrix[x + j][y + i + 1] != 8: 修改后不存在等于8的单元格了！
                    elif self.matrix[x + j][y + i + 1] != 0:
                        return True

        return False

    # Returns true if mino is at the left edge
    def __is_leftedge__(self, x, y, mino, r):
        grid = tetris_mimo.mino_map[mino - 1][r]

        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    if (x + j - 1) < 0:
                        return True
                    elif self.matrix[x + j - 1][y + i] != 0:
                        return True

        return False

    # Returns true if mino is at the right edge
    def __is_rightedge__(self, x, y, mino, r):
        grid = tetris_mimo.mino_map[mino - 1][r]

        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    if (x + j + 1) > 9:
                        return True
                    elif self.matrix[x + j + 1][y + i] != 0:
                        return True

        return False

    # Returns true if turning right is possible
    def __is_turnable_r__(self, x, y, mino, r):
        if r != 3:
            grid = tetris_mimo.mino_map[mino - 1][r + 1]
        else:
            grid = tetris_mimo.mino_map[mino - 1][0]
        # grid = tetris_mimo.mino_map[mino - 1][(r + 1) % 4]

        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    if (x + j) < 0 or (x + j) > 9 or (y + i) < 0 or (y + i) > 20:
                        return False
                    elif self.matrix[x + j][y + i] != 0:
                        return False

        return True

    # Returns true if turning left is possible
    def __is_turnable_l__(self, x, y, mino, r):
        if r != 0:
            grid = tetris_mimo.mino_map[mino - 1][r - 1]
        else:
            grid = tetris_mimo.mino_map[mino - 1][3]
        # grid = tetris_mimo.mino_map[mino - 1][(r + 3) % 4]

        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    if (x + j) < 0 or (x + j) > 9 or (y + i) < 0 or (y + i) > 20:
                        return False
                    elif self.matrix[x + j][y + i] != 0:
                        return False

        return True

    # Returns true if new block is drawable
    def __is_stackable__(self):
        grid = tetris_mimo.mino_map[self.next_mino - 1][0]

        for i in range(4):
            for j in range(4):
                # print(grid[i][j], matrix[3 + j][i])
                if grid[i][j] != 0 and self.matrix[3 + j][i] != 0:
                    return False

        return True

    # 这里用0表示空，1表示已经放置，2表示正在下落！希望这样DQN可以更好的训练
    def get_state(self):
        grid = tetris_mimo.mino_map[self.mino - 1][self.rotation]

        state = [[1 if self.matrix[j][i] != 0 else 0
                  for i in range(self.height + 1)] for j in range(self.width)]  # Board matrix

        for i in range(4):
            for j in range(4):
                if grid[i][j] != 0:
                    # FIXME 这里也发生过越界，应该是这个程序本身没设计好！
                    state[self.dx + j][self.dy + i] = 2

        return np.array(state).reshape(10, 21).T.flatten()

    def reset(self):
        self.game_over = False
        self.hold = False
        self.dx, self.dy = 3, 0
        self.rotation = 0
        self.mino = randint(1, 7)
        self.next_mino = randint(1, 7)
        self.hold_mino = -1
        self.framerate = self.speed
        self.score = 0
        self.level = 1
        self.goal = self.level * 5
        self.bottom_count = 0
        self.hard_drop = False
        self.name_location = 0
        self.name = [65, 65, 65]
        self.matrix = [[0 for _ in range(self.height + 1)] for _ in range(self.width)]

        pygame.time.set_timer(pygame.USEREVENT, 1)
        return self.get_state()

    def machine_render(self, action):
        reward = 0
        lines_cleared = 0
        # Game screen
        if self.start:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.done = True
                elif event.type == USEREVENT:
                    # Set speed
                    if not self.game_over:
                        pygame.time.set_timer(pygame.USEREVENT, self.framerate)

                    # Draw a mino
                    self.__draw_mino__()
                    self.__draw_board__()

                    # Erase a mino
                    if not self.game_over:
                        self.__erase_mino__()

                    # Move mino down
                    if not self.__is_bottom__(self.dx, self.dy, self.mino, self.rotation):
                        self.dy += 1

                    # Create new mino
                    else:
                        if self.hard_drop or self.bottom_count == 6:
                            self.hard_drop = False
                            self.bottom_count = 0
                            self.score += 1 * self.level
                            self.__draw_mino__()
                            self.__draw_board__()
                            if self.__is_stackable__():
                                self.mino = self.next_mino
                                self.next_mino = randint(1, 7)
                                self.dx, self.dy = 3, 0
                                self.rotation = 0
                                self.hold = False
                            else:
                                self.start = False
                                self.game_over = True
                                pygame.time.set_timer(pygame.USEREVENT, 1)
                        else:
                            self.bottom_count += 1

                    # Erase line
                    erase_count = 0
                    for j in range(21):
                        is_full = True
                        for i in range(10):
                            if self.matrix[i][j] == 0:
                                is_full = False
                                break  # 没必要就继续检测了！
                        if is_full:
                            erase_count += 1
                            k = j
                            while k > 0:
                                for i in range(10):
                                    self.matrix[i][k] = self.matrix[i][k - 1]
                                k -= 1

                    if erase_count == 1:
                        self.score += 5 * self.level
                    elif erase_count == 2:
                        self.score += 15 * self.level
                    elif erase_count == 3:
                        self.score += 35 * self.level
                    elif erase_count == 4:
                        self.score += 100 * self.level

                    # set reward in this step!
                    reward = 1 + (erase_count ** 2) * self.width
                    lines_cleared = erase_count

                    # Increase level
                    self.goal -= erase_count
                    if self.goal < 1 and self.level < 15 and self.levelup:
                        self.level += 1
                        self.goal += self.level * 5
                        # self.framerate = int(self.framerate * 0.8)

            self.__erase_mino__()
            # NOOP
            if MOVEMENT[action] == 'NOOP':
                self.__draw_mino__()
                self.__draw_board__()
            # Turn right
            elif MOVEMENT[action] == 'Turn Right':
                if self.__is_turnable_r__(self.dx, self.dy, self.mino, self.rotation):
                    self.rotation += 1
                # TODO 这部分看起来并没有意义，并且还造成了越界，先删除了！
                # # Kick
                # elif self.__is_turnable_r__(self.dx, self.dy - 1, self.mino, self.rotation):
                #     self.dy -= 1
                #     self.rotation += 1
                # elif self.__is_turnable_r__(self.dx + 1, self.dy, self.mino, self.rotation):
                #     self.dx += 1
                #     self.rotation += 1
                # elif self.__is_turnable_r__(self.dx - 1, self.dy, self.mino, self.rotation):
                #     self.dx -= 1
                #     self.rotation += 1
                # elif self.__is_turnable_r__(self.dx, self.dy - 2, self.mino, self.rotation):
                #     self.dy -= 2
                #     self.rotation += 1
                # elif self.__is_turnable_r__(self.dx + 2, self.dy, self.mino, self.rotation):
                #     self.dx += 2
                #     self.rotation += 1
                # elif self.__is_turnable_r__(self.dx - 2, self.dy, self.mino, self.rotation):
                #     self.dx -= 2
                #     self.rotation += 1
                if self.rotation == 4:
                    self.rotation = 0
                self.__draw_mino__()
                self.__draw_board__()
            # Turn left
            elif MOVEMENT[action] == 'Turn Left':
                if self.__is_turnable_l__(self.dx, self.dy, self.mino, self.rotation):
                    self.rotation -= 1
                # TODO 这部分看起来并没有意义，并且还造成了越界，先删除了！
                # # Kick
                # elif self.__is_turnable_l__(self.dx, self.dy - 1, self.mino, self.rotation):
                #     self.dy -= 1
                #     self.rotation -= 1
                # elif self.__is_turnable_l__(self.dx + 1, self.dy, self.mino, self.rotation):
                #     self.dx += 1
                #     self.rotation -= 1
                # elif self.__is_turnable_l__(self.dx - 1, self.dy, self.mino, self.rotation):
                #     self.dx -= 1
                #     self.rotation -= 1
                # elif self.__is_turnable_l__(self.dx, self.dy - 2, self.mino, self.rotation):
                #     self.dy -= 2
                #     self.rotation += 1
                # elif self.__is_turnable_l__(self.dx + 2, self.dy, self.mino, self.rotation):
                #     self.dx += 2
                #     self.rotation += 1
                # elif self.__is_turnable_l__(self.dx - 2, self.dy, self.mino, self.rotation):
                #     self.dx -= 2
                if self.rotation == -1:
                    self.rotation = 3
                self.__draw_mino__()
                self.__draw_board__()
            # Move left
            elif MOVEMENT[action] == 'Move Left':
                if not self.__is_leftedge__(self.dx, self.dy, self.mino, self.rotation):
                    self.dx -= 1
                self.__draw_mino__()
                self.__draw_board__()
            # Move right
            elif MOVEMENT[action] == 'Move Right':
                if not self.__is_rightedge__(self.dx, self.dy, self.mino, self.rotation):
                    self.dx += 1
                self.__draw_mino__()
                self.__draw_board__()
            pygame.display.update()
        else:
            self.reset()
            self.start = True

        return self.get_state(), reward, self.game_over, lines_cleared

    def human_rander(self):  # 人玩的
        while not self.done:
            # Game screen
            if self.start:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.done = True
                    elif event.type == USEREVENT:
                        # Set speed
                        if not self.game_over:
                            keys_pressed = pygame.key.get_pressed()
                            if keys_pressed[K_DOWN]:
                                pygame.time.set_timer(pygame.USEREVENT, self.framerate * 1)
                            else:
                                pygame.time.set_timer(pygame.USEREVENT, self.framerate * 10)

                        # Draw a mino
                        self.__draw_mino__()
                        self.__draw_board__()

                        # Erase a mino
                        if not self.game_over:
                            self.__erase_mino__()

                        # Move mino down
                        if not self.__is_bottom__(self.dx, self.dy, self.mino, self.rotation):
                            self.dy += 1

                        # Create new mino
                        else:
                            if self.hard_drop or self.bottom_count == 6:
                                self.hard_drop = False
                                self.bottom_count = 0
                                self.score += 1 * self.level
                                self.__draw_mino__()
                                self.__draw_board__()
                                if self.__is_stackable__():
                                    self.mino = self.next_mino
                                    self.next_mino = randint(1, 7)
                                    self.dx, self.dy = 3, 0
                                    self.rotation = 0
                                    self.hold = False
                                else:
                                    self.start = False
                                    self.game_over = True
                                    self.done = True
                                    pygame.time.set_timer(pygame.USEREVENT, 1)
                            else:
                                self.bottom_count += 1

                        # Erase line
                        erase_count = 0
                        for j in range(21):
                            is_full = True
                            for i in range(10):
                                if self.matrix[i][j] == 0:
                                    is_full = False
                            if is_full:
                                erase_count += 1
                                k = j
                                while k > 0:
                                    for i in range(10):
                                        self.matrix[i][k] = self.matrix[i][k - 1]
                                    k -= 1

                        if erase_count == 1:
                            self.score += 5 * self.level
                        elif erase_count == 2:
                            self.score += 15 * self.level
                        elif erase_count == 3:
                            self.score += 35 * self.level
                        elif erase_count == 4:
                            self.score += 100 * self.level

                        # Increase level
                        self.goal -= erase_count
                        if self.goal < 1 and self.level < 15 and self.levelup:
                            self.level += 1
                            self.goal += self.level * 5
                            self.framerate = int(self.framerate * 0.8)

                    elif event.type == KEYDOWN:
                        self.__erase_mino__()
                        if event.key == K_ESCAPE:
                            self.pause = True
                        # Hard drop
                        elif event.key == K_SPACE:
                            while not self.__is_bottom__(self.dx, self.dy, self.mino, self.rotation):
                                self.dy += 1
                            self.hard_drop = True
                            pygame.time.set_timer(pygame.USEREVENT, 1)
                            self.__draw_mino__()
                            self.__draw_board__()
                        # Hold
                        elif event.key == K_LSHIFT or event.key == K_c:
                            if not self.hold:
                                if self.hold_mino == -1:
                                    self.hold_mino = self.mino
                                    self.mino = self.next_mino
                                    self.next_mino = randint(1, 7)
                                else:
                                    self.hold_mino, self.mino = self.mino, self.hold_mino
                                self.dx, self.dy = 3, 0
                                self.rotation = 0
                                self.hold = True
                            self.__draw_mino__()
                            self.__draw_board__()
                        # Turn right
                        elif event.key == K_UP or event.key == K_x:
                            if self.__is_turnable_r__(self.dx, self.dy, self.mino, self.rotation):
                                self.rotation += 1
                            # TODO 这部分看起来并没有意义，并且还造成了越界，先删除了！
                            # # Kick
                            # elif self.__is_turnable_r__(self.dx, self.dy - 1, self.mino, self.rotation):
                            #     self.dy -= 1
                            #     self.rotation += 1
                            # elif self.__is_turnable_r__(self.dx + 1, self.dy, self.mino, self.rotation):
                            #     self.dx += 1
                            #     self.rotation += 1
                            # elif self.__is_turnable_r__(self.dx - 1, self.dy, self.mino, self.rotation):
                            #     self.dx -= 1
                            #     self.rotation += 1
                            # elif self.__is_turnable_r__(self.dx, self.dy - 2, self.mino, self.rotation):
                            #     self.dy -= 2
                            #     self.rotation += 1
                            # elif self.__is_turnable_r__(self.dx + 2, self.dy, self.mino, self.rotation):
                            #     self.dx += 2
                            #     self.rotation += 1
                            # elif self.__is_turnable_r__(self.dx - 2, self.dy, self.mino, self.rotation):
                            #     self.dx -= 2
                            #     self.rotation += 1
                            if self.rotation == 4:
                                self.rotation = 0
                            self.__draw_mino__()
                            self.__draw_board__()
                        # Turn left
                        elif event.key == K_z or event.key == K_LCTRL:
                            if self.__is_turnable_l__(self.dx, self.dy, self.mino, self.rotation):
                                self.rotation -= 1
                            # TODO 这部分看起来并没有意义，并且还造成了越界，先删除了！
                            # # Kick
                            # elif self.__is_turnable_l__(self.dx, self.dy - 1, self.mino, self.rotation):
                            #     self.dy -= 1
                            #     self.rotation -= 1
                            # elif self.__is_turnable_l__(self.dx + 1, self.dy, self.mino, self.rotation):
                            #     self.dx += 1
                            #     self.rotation -= 1
                            # elif self.__is_turnable_l__(self.dx - 1, self.dy, self.mino, self.rotation):
                            #     self.dx -= 1
                            #     self.rotation -= 1
                            # elif self.__is_turnable_l__(self.dx, self.dy - 2, self.mino, self.rotation):
                            #     self.dy -= 2
                            #     self.rotation += 1
                            # elif self.__is_turnable_l__(self.dx + 2, self.dy, self.mino, self.rotation):
                            #     self.dx += 2
                            #     self.rotation += 1
                            # elif self.__is_turnable_l__(self.dx - 2, self.dy, self.mino, self.rotation):
                            #     self.dx -= 2
                            if self.rotation == -1:
                                self.rotation = 3
                            self.__draw_mino__()
                            self.__draw_board__()
                        # Move left
                        elif event.key == K_LEFT:
                            if not self.__is_leftedge__(self.dx, self.dy, self.mino, self.rotation):
                                self.dx -= 1
                            self.__draw_mino__()
                            self.__draw_board__()
                        # Move right
                        elif event.key == K_RIGHT:
                            if not self.__is_rightedge__(self.dx, self.dy, self.mino, self.rotation):
                                self.dx += 1
                            self.__draw_mino__()
                            self.__draw_board__()
                pygame.display.update()
            else:
                self.reset()
                self.start = True
        pygame.quit()


def test_machine():
    env = Tetris(2)
    # env.human_rander()
    for i in range(5):  # 演示5轮
        print('<<<<<<<<<Episode: %s' % i)
        s = env.reset()  # 重置环境
        # episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
        while True:  # 开始一个episode (每一个循环代表一步)
            # action = 0  # 输入该步对应的状态s，选择动作
            action = np.random.randint(0, 5)  # 输入该步对应的状态s，选择动作
            s_, r, done, _ = env.machine_render(action)  # 执行动作，获得反馈

            # print(env.get_state())

            if done:  # 如果done为True
                break


def test_human():
    env = Tetris(100)
    env.human_rander()


if __name__ == "__main__":
    # test_human()
    test_machine()
