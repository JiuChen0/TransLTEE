from torch.utils.data import DataLoader
from models import MyModel
from losses import compute_train_loss, compute_valid_loss
from data import get_dataloader
from config import Config

def train():
    config = Config()
    train_loader = get_dataloader('train.csv', config.batch_size)
    valid_loader = get_dataloader('valid.csv', config.batch_size)

    model = MyModel(config.input_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        for batch in train_loader:
            loss = compute_train_loss(model, batch, config.loss_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                loss = compute_valid_loss(model, batch, config.loss_weights)

        # Save the model
        torch.save(model.state_dict(), f'{config.save_dir}/model_{epoch}.pth')
