import torch
from utils import to_device, compute_accuracy
from losses import compute_train_loss, compute_valid_loss  # Make sure this function is defined in losses.py

def train(model, train_dataloader, valid_dataloader, criterion, optimizer, config):
    """
    Train the model
    model: The model object
    train_dataloader: Training data loader
    valid_dataloader: Validation data loader
    criterion: Loss function
    optimizer: Optimizer
    config: Configuration object
    """

    # Move the model to the device
    device = config.device
    model.to(device)

    for epoch in range(config.num_epochs):
        # Set the model to training mode
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            # Move data to the device
            batch = to_device(batch, device)

            # Compute loss
            loss = compute_train_loss(model, batch, criterion)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Evaluate the model on the validation set
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_dataloader:
                # Move data to the device
                batch = to_device(batch, device)

                # Compute loss
                loss = compute_valid_loss(model, batch, criterion)

                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)

        # Print training and validation loss
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}')

    return model
###################
# You may need to modify the compute_train_loss and compute_valid_loss functions according to your actual needs.
###################