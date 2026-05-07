"""
Imitation learning controller for surface trajectory following.

Predicts delta (displacement) given current position and goal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImitationController(nn.Module):
    """Feedforward controller with residual connections for imitation learning.
    
    Input: [current_pos(3), goal(3)] = 6 dimensions
    Output: delta(3) = displacement to add to current position
    
    Architecture improvements:
    - Predicts delta instead of absolute position (better generalization)
    - Skip connections for smoother gradients
    - Tanh output to limit maximum displacement per step
    """
    def __init__(self, input_size=6, hidden_size=16, output_size=3, max_step=0.2):
        super(ImitationController, self).__init__()
        
        self.max_step = max_step  # Maximum displacement per step
        
        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Residual blocks
        self.block1_fc = nn.Linear(hidden_size, hidden_size)
        self.block1_norm = nn.LayerNorm(hidden_size)
        self.block1_dropout = nn.Dropout(0.1)
        
        self.block2_fc = nn.Linear(hidden_size, hidden_size)
        self.block2_norm = nn.LayerNorm(hidden_size)
        self.block2_dropout = nn.Dropout(0.1)
        
        # Output layers
        self.output_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_norm = nn.LayerNorm(hidden_size // 2)
        self.output_fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass with residual connections.
        
        Args:
            x: (batch, 6) tensor [current_x, current_y, current_z, goal_x, goal_y, goal_z]
            
        Returns:
            (batch, 3) tensor [delta_x, delta_y, delta_z] - displacement to add to current position
        """
        # Input projection
        h = F.relu(self.input_norm(self.input_layer(x)))
        
        # Residual block 1 (skip connection)
        residual = h
        h = self.block1_fc(h)
        h = self.block1_norm(h)
        h = F.relu(h)
        h = self.block1_dropout(h)
        h = h + residual  # Skip connection
        
        # Residual block 2 (skip connection)
        residual = h
        h = self.block2_fc(h)
        h = self.block2_norm(h)
        h = F.relu(h)
        h = self.block2_dropout(h)
        h = h + residual  # Skip connection
        
        # Output layers
        h = F.relu(self.output_norm(self.output_fc1(h)))
        delta = self.output_fc2(h)
        
        # Tanh to limit displacement range
        delta = torch.tanh(delta) * self.max_step
        
        return delta

