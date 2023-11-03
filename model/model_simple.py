import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# for visualization of the neural network
import matplotlib.pyplot as plt

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 
        super().__init__()
        # Define the first linear layer taking input_size and outputting to hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        
        # Define the second linear layer taking hidden_size and outputting to output_size
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        # Applying the first linear layer followed by a ReLU activation function
        x = F.relu(self.linear1(x))
        
        # Applying the second linear layer
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'): 
        # Define the folder path for saving the model
        model_folder_path = './model'
        
        # Create the folder if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Join the folder path with the file name to get the complete path
        file_name = os.path.join(model_folder_path, file_name)
        
        # Save the model's state dictionary to the specified file
        torch.save(self.state_dict(), file_name)



class QTrainer:
    def __init__(self, model, lr, gamma): 
        # Initialize learning rate, gamma value, and model
        self.lr = lr
        self.gamma = gamma
        self.model = model
        
        # Set the optimizer using the Adam optimization algorithm
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) 
        
        # Set the loss function to Mean Squared Error
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done): 
        # Convert inputs to tensors with appropriate data types
        # A representation of the current situation or configuration 
        # (the positions of the snake segments, the direction of movement, and the position of the food)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # Choice made by the agent (in Snake, might be to move up, down, left, or right)
        action = torch.tensor(action, dtype=torch.long)
        # Scalar feedback received after taking an action in a state (measure of how good or bad that action was)
        reward = torch.tensor(reward, dtype=torch.float)

        # If the input states are 1D, add an extra dimension
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Get the Q-values predictions for current states using the model
        pred = self.model(state)

        # Clone the predictions to create a target tensor for loss calculation
        target = pred.clone()

        # Calculate the updated Q-values using the Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Update the target tensor at the action taken with the new Q-value
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Zero out the gradients, compute the loss, and perform backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        # Update the model parameters using the optimizer
        self.optimizer.step()
     
