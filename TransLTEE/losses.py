import torch

def compute_train_loss(model, batch, criterion):
    """
    Compute training loss
    model: The model object
    batch: A batch of training data
    criterion: Loss function
    """
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss

def compute_valid_loss(model, batch, criterion):
    """
    Compute validation loss
    model: The model object
    batch: A batch of validation data
    criterion: Loss function
    """
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss
######################
# If the model returns multiple outputs, or if the loss function requires additional parameters, 
# you might need to make corresponding adjustments in these functions
######################