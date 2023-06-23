import torch
from models import MyModel
from config import Config
from train import train
from data import get_dataloader  # import function defined in data.py

def main():
    # Create a configuration object
    config = Config()

    # Check if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model and move it to the GPU
    model = MyModel(config.input_dim, config.hidden_dim).to(device)

    # Create the data loaders
    train_dataloader = get_dataloader('train.csv', config.batch_size)
    valid_dataloader = get_dataloader('valid.csv', config.batch_size)  # If there is a validation set

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Train the model
    train(model, train_dataloader, valid_dataloader, criterion, optimizer, config)  # The train function needs to be modified to accept the data loader for the validation set

    # Save the model
    torch.save(model.state_dict(), f'{config.save_dir}/model.pt')

if __name__ == '__main__':
    main()