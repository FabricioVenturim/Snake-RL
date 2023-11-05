from agent_model2 import Agent
from game.snake_without_growing import SnakeGame
from helper.plot import plot
from model.model import Linear_QNet, QTrainer

# Define the training function
def train():
    """ Trains the agent to play the Snake game.
    """

    # Lists to store scores and mean scores for plotting
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0  # Record score

    # Initialize the agent and the Snake game
    agent = Agent()
    game = SnakeGame()

    # Training loop
    while True:
        # Get the current state of the game
        state_old = agent.get_state(game)

        # Decide on an action based on the current state
        final_move = agent.get_action(state_old)

        # Play a step in the game with the chosen action
        reward, done, score = game.play_step(final_move)

        # Get the new state after taking the action
        state_new = agent.get_state(game)

        # Train the model with recent experience (short-term memory)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store the experience for later training (long-term memory)
        agent.remember(state_old, final_move, reward, state_new, done)

        # If the game is over
        if done:
            # Reset the game for a new episode
            game.reset()

            # Increment the number of games played
            agent.n_games += 1

            # Train the model using experiences from long-term memory
            agent.train_long_memory()

            # Update the record score if the current score exceeds it
            if score > record:
                record = score
                # Save the model with the new record
                agent.model.save()

            # Display game statistics
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update scores list and calculate mean score for plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Plot the scores and mean scores
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()

