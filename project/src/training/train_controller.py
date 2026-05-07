"""
Training utilities for imitation controller.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_controller(controller, states_train, targets_train, states_test=None, targets_test=None, num_epochs=300, batch_size=32, lr=0.001):
    """Train imitation controller with optional test set evaluation.
    
    Args:
        controller: Controller model (nn.Module)
        states_train: (N, 6) training inputs [current_pos(3) + goal(3)]
        targets_train: (N, 3) training targets [next_pos(3)]
        states_test: Optional (M, 6) test inputs
        targets_test: Optional (M, 3) test targets
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        train_losses: List of training loss values per epoch
        test_losses: List of test loss values per epoch (if test data provided)
        eval_epochs: List of evaluation epochs
    """
    print('\n========== TRAINING IMITATION CONTROLLER ==========')
    
    # Convert to PyTorch tensors
    states_tensor = torch.from_numpy(states_train.astype(np.float32))
    targets_tensor = torch.from_numpy(targets_train.astype(np.float32))
    
    
    states_test_tensor = torch.from_numpy(states_test.astype(np.float32))
    targets_test_tensor = torch.from_numpy(targets_test.astype(np.float32))

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(states_tensor, targets_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f'\nController architecture: {controller.__class__.__name__}')
    print(f'Parameters: {sum(p.numel() for p in controller.parameters())}')
    print(f'Training samples: {len(states_train)}')
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(controller.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    train_losses = []
    test_losses = []
    eval_epochs = []
    
    print(f'\nTraining for {num_epochs} epochs...')
    for epoch in range(num_epochs):
        # Training
        controller.train()
        epoch_loss = 0.0
        
        for batch_states, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = controller(batch_states)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        eval_epochs.append(epoch)
        
    
        controller.eval()
        with torch.no_grad():
            test_predictions = controller(states_test_tensor)
            test_loss = criterion(test_predictions, targets_test_tensor).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)
    
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}')
          
    
    print(f'\nTraining complete! Final train loss: {train_losses[-1]:.6f}')
    print(f'Final test loss: {test_losses[-1]:.6f}')
    return train_losses, test_losses, eval_epochs