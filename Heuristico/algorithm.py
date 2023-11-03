from game.snake_algoritmo import SnakeGame
import random

BLOCK_SIZE = 20


def train():
    game = SnakeGame()
    last_head = game.get_position()
    last_head = [last_head[0] - 1, last_head[1]]

    while True:
        # Obtenha as coordenadas x e y da cabeça da cobra
        head_x, head_y = game.get_position()

        # Lista de ações disponíveis
        available_moves = []

        if head_y < last_head[1]:
            if head_y % 2 == 0:
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])
            else:
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0])

        elif head_y > last_head[1]:
            if head_y % 2 == 0:
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0])
            else:
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])
            
        elif head_x < last_head[0]:
            if head_x % 2 == 0:
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])
            else:
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0])

        elif head_x > last_head[0]:
            if head_x % 2 == 0:
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0]) 
            else:
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])

        if not game.is_collision_after_move([1, 0, 0], last_head)[0]:
            available_moves.append([1, 0, 0])

        less_distance = 10000000
        final_move = None
        for i in available_moves:
            dist = game.is_collision_after_move(i, last_head)[1]
            
            if dist == less_distance:
                if random.random() > 0.5:
                    less_distance = dist
                    final_move = i

            if dist < less_distance:
                less_distance = dist
                final_move = i

        last_head = [head_x, head_y]

        reward, done, score = game.play_step(final_move)
        if done:
            game.reset()


if __name__ == '__main__':
    train()