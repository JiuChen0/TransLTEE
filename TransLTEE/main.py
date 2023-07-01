import tensorflow as tf
from models import MyModel  # Please replace with your actual model
from config import Config
import numpy as np
from data import get_dataloader
import logging

# Create a logger object.
logger = logging.getLogger(__name__)

# Set the log level to INFO.
logger.setLevel(logging.INFO)

# Create a console handler.
# handler = logging.StreamHandler()

# Add the handler to the logger.
# logger.addHandler(handler)

def main():
    # Create a configuration object
    logger.info('CREATING CONFIG OBJECT...')
    config = Config()
    logger.info('CONFIG OBJECT SUCCESSFULLY CREATED!')

    # Create the model
    logger.info('BUILDING MODELS...')
    model = MyModel(config.input_dim, config.hidden_dim)
    logger.info('MODELS SUCCESSFULLY BUILT!')

    # Load and process the data
    t0=10

    for j in range(1, 11):

        TY = np.loadtxt('data/IHDP/csv/ihdp_npci_' + str(j) + '.csv', delimiter=',')
        matrix = TY[:, 5:]
        N = TY.shape[0]

        out_treat = np.loadtxt('data/IHDP/Series_y_' + str(j) + '.txt', delimiter=',')
        ts = out_treat[:, 0]
        ts = np.reshape(ts, (N, 1))
        ys = np.concatenate((out_treat[:, 1:(t0 + 1)], out_treat[:, -1].reshape(N, 1)), axis=1)
        from sklearn.model_selection import train_test_split

        matrix_rep = np.repeat(matrix[:, np.newaxis, :], t0, axis=1)
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(matrix_rep, ys, ts, test_size=0.2)

        print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test), np.shape(t_train), np.shape(t_test))

    # Create the data loaders
    # logger.info('CREATING DATA LOADERS...')
    # train_dataloader = get_dataloader('train.csv', config.batch_size)
    # valid_dataloader = get_dataloader('valid.csv', config.batch_size)  # If there is a validation set
    # logger.info('DATA LOADERS SUCCESSFULLY CREATED!')

    # Compile the model with the optimizer and loss function
    logger.info('COMPILING MODEL WITH THE OPTIMIZER AND LOSS FUNCTION')
    model.compile(optimizer=config.optimizer, 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])
    logger.info('OPTIMIZER AND LOSS FUNCTION SUCCESSFULLY COMPILED')

    # Train the model
    logger.info('TRAINING MODEL...')
    model.fit(train_dataloader, 
              validation_data=valid_dataloader, 
              epochs=config.epochs, 
              batch_size=config.batch_size)
    logger.info('MODEL SUCCESSFULLY TRAINED!')

    # Save the model
    logger.info('SAVING MODEL...')
    model.save(f'{config.save_dir}/model')
    logger.info('MODEL SUCCESSFULLY SAVED!')

if __name__ == '__main__':
    main()
