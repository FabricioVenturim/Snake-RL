from game.snake_algoritmo import SnakeGame
import random


BLOCK_SIZE = 20

def train():
    """Function that trains the snake to play the game.
    """   
    game = SnakeGame()
    # Get the initial position of the snake's head and adjust it for the starting direction
    last_head = game.get_position()
    last_head = [last_head[0] - 1, last_head[1]]

    while True:
        # Get the current position of the snake's head
        head_x, head_y = game.get_position()

        # Initialize an empty list to keep track of available moves
        available_moves = []

        # Logic to determine the available moves based on the snake's current head position
        # It takes into account where the head was in the last frame to avoid going backwards
        if head_y < last_head[1]:
             # When the head's Y position has decreased (moved up on the grid)
            if head_y % 2 == 0:
                # Check if turning left causes a collision
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])
            else:
                # Check if turning right causes a collision
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0])

        elif head_y > last_head[1]:
            # When the head's Y position has increased (moved down on the grid)
            if head_y % 2 == 0:
                # Check if turning right causes a collision
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0])
            else:
                # Check if turning left causes a collision
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])
            
        elif head_x < last_head[0]:
            # When the head's X position has decreased (moved left on the grid)
            if head_x % 2 == 0:
                # Check if turning left causes a collision
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])
            else:
                # Check if turning right causes a collision
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0])

        elif head_x > last_head[0]:
            # When the head's X position has increased (moved right on the grid)
            if head_x % 2 == 0:
                # Check if turning right causes a collision
                if not game.is_collision_after_move([0, 1, 0], last_head)[0]:
                    available_moves.append([0, 1, 0]) 
            else:
                # Check if turning left causes a collision
                if not game.is_collision_after_move([0, 0, 1], last_head)[0]:
                    available_moves.append([0, 0, 1])

        # Check if moving forward causes a collision
        if not game.is_collision_after_move([1, 0, 0], last_head)[0]:
            available_moves.append([1, 0, 0])

        # Initialize a very large number to find the minimum distance to food
        less_distance = 10000000
        # This will hold the best move after evaluating options
        final_move = None

        # Evaluate the available moves and choose the best one
        for i in available_moves:
            dist = game.is_collision_after_move(i, last_head)[1]
            # Randomly choose between moves if the distance to food is equal
            if dist == less_distance:
                if random.random() > 0.5:
                    less_distance = dist
                    final_move = i
            # If a shorter distance to the food is found, choose that move
            if dist < less_distance:
                less_distance = dist
                final_move = i

        # Update the position of the head for the next iteration
        last_head = [head_x, head_y]
        # Execute the chosen move and collect the game's response
        reward, done, score = game.play_step(final_move)
        if done:
            game.reset()

if __name__ == '__main__':
    train()
