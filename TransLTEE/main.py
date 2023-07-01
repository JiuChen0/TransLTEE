import tensorflow as tf
from models import MyModel  # Please replace with your actual model
from config import Config
from data import get_dataloader
import logging

# Create a logger object.
logger = logging.getLogger(__name__)

# Set the log level to INFO.
logger.setLevel(logging.INFO)

# Create a console handler.
handler = logging.StreamHandler()

# Add the handler to the logger.
logger.addHandler(handler)

def main():
    # Create a configuration object
    logging.info('CREATING CONFIG OBJECT...')
    config = Config()
    logging.info('CONFIG OBJECT SUCCESSFULLY CREATED!')

    # Create the model
    logging.info('BUILDING MODELS...')
    model = MyModel(config.input_dim, config.hidden_dim)
    logging.info('MODELS SUCCESSFULLY BUILT!')

    # Create the data loaders
    logging.info('CREATING DATA LOADERS...')
    train_dataloader = get_dataloader('train.csv', config.batch_size)
    valid_dataloader = get_dataloader('valid.csv', config.batch_size)  # If there is a validation set
    logging.info('DATA LOADERS SUCCESSFULLY CREATED!')

    # Compile the model with the optimizer and loss function
    logging.info('COMPILING MODEL WITH THE OPTIMIZER AND LOSS FUNCTION')
    model.compile(optimizer=config.optimizer, 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])
    logging.info('OPTIMIZER AND LOSS FUNCTION SUCCESSFULLY COMPILED')

    # Train the model
    logging.info('TRAINING MODEL...')
    model.fit(train_dataloader, 
              validation_data=valid_dataloader, 
              epochs=config.epochs, 
              batch_size=config.batch_size)
    logging.info('MODEL SUCCESSFULLY TRAINED!')

    # Save the model
    logging.info('SAVING MODEL...')
    model.save(f'{config.save_dir}/model')
    logging.info('MODEL SUCCESSFULLY SAVED!')

if __name__ == '__main__':
    main()
