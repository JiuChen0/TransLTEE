class Config:
    """
    Configuration for the model and the training process
    """
    def __init__(self):
        # Training parameters
        self.epochs = 100  # Number of training epochs
        self.batch_size = 64  # Batch size
        self.learning_rate = 0.001  # Learning rate
        self.weight_decay = 0.0001  # Weight decay
        self.save_dir = './checkpoints'  # Path to save the model

        # Model parameters
        self.input_dim = 128  # Input dimension
        self.hidden_dim = 256  # Hidden layer dimension