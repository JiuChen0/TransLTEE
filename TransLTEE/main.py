import tensorflow as tf
from models import MyModel  # Please replace with your actual model
from config import Config
from data import get_dataloader

def main():
    # Create a configuration object
    config = Config()

    # Create the model
    model = MyModel(config.input_dim, config.hidden_dim)

    # Create the data loaders
    train_dataloader = get_dataloader('train.csv', config.batch_size)
    valid_dataloader = get_dataloader('valid.csv', config.batch_size)  # If there is a validation set

    # Compile the model with the optimizer and loss function
    model.compile(optimizer=config.optimizer, 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_dataloader, 
              validation_data=valid_dataloader, 
              epochs=config.epochs, 
              batch_size=config.batch_size)

    # Save the model
    model.save(f'{config.save_dir}/model')

if __name__ == '__main__':
    main()
