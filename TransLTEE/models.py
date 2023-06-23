import torch
from torch import nn
from transformers import TransformerModel
from losses import WassersteinLoss

class DoubleHeadRNN(nn.Module):
    """ 
    Double-headed RNN structure
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Constructor
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        """
        super(DoubleHeadRNN, self).__init__()
        # Define two RNN structures
        self.rnn0 = nn.RNN(input_dim, hidden_dim)
        self.rnn1 = nn.RNN(input_dim, hidden_dim)
    
    def forward(self, x):
        """
        Forward propagation
        x: Input data
        """
        # Process the input data to obtain the outputs of the two RNNs
        output0, _ = self.rnn0(x)
        output1, _ = self.rnn1(x)
        return output0, output1

class TransformerEncoder(nn.Module):
    """
    Transformer encoder structure
    """
    def __init__(self, hidden_dim):
        """
        Constructor
        hidden_dim: Hidden layer dimension
        """
        super(TransformerEncoder, self).__init__()
        # Use pre-trained bert as Transformer encoder
        self.transformer = TransformerModel.from_pretrained('bert-base-uncased')
    
    def forward(self, x):
        """
        Forward propagation
        x: Input data
        """
        # Process the input data to obtain the output of the Transformer encoder
        outputs = self.transformer(x)
        return outputs.last_hidden_state

class MyModel(nn.Module):
    """
    Our model structure, including the double-headed RNN and Transformer encoder
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Constructor
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        """
        super(MyModel, self).__init__()
        # Define the double-headed RNN
        self.double_head_rnn = DoubleHeadRNN(input_dim, hidden_dim)
        # Define the Transformer encoder
        self.transformer_encoder = TransformerEncoder(hidden_dim)
        # Define the loss function, here we use the Wasserstein distance
        self.criterion = WassersteinLoss()
    
    def forward(self, x):
        """
        Forward propagation
        x: Input data
        """
        # Get the outputs of the double-headed RNN
        output0, output1 = self.double_head_rnn(x)
        # Get the output of the Transformer encoder
        encoded0 = self.transformer_encoder(output0)
        encoded1 = self.transformer_encoder(output1)
        # Calculate the Wasserstein distance between the two outputs as the model's loss
        loss = self.criterion(encoded0, encoded1)
        return loss


###################
# This code does not implement the attention mechanism or the stacking of multiple loss functions
###################