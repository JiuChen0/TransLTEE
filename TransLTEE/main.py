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
handler = logging.StreamHandler()

# Add the handler to the logger.
logger.addHandler(handler)

def main():
    # Create a configuration object
    logger.info('CREATING CONFIG OBJECT...')
    config = Config()
    logger.info('CONFIG OBJECT SUCCESSFULLY CREATED!')

    # Create the model
    logger.info('BUILDING MODELS...')
    model = MyModel(config.input_dim)
    logger.info('MODELS SUCCESSFULLY BUILT!')

    # Load and process the data
    t0=10

    # for j in range(1, 11):

    TY = np.loadtxt('../data/IHDP/csv/ihdp_npci_' + "1" + '.csv', delimiter=',')
    matrix = TY[:, 5:]
    N = TY.shape[0]

    out_treat = np.loadtxt('../data/IHDP/Series_y_' + "1" + '.txt', delimiter=',')
    ts = out_treat[:, 0]
    ts = np.reshape(ts, (N, 1))
    # ys = np.concatenate((out_treat[:, 1:(t0 + 1)], out_treat[:, -1].reshape(N, 1)), axis=1)
    ys = out_treat[:, 1:(t0 + 1)]
    from sklearn.model_selection import train_test_split

    matrix_rep = np.repeat(matrix[:, np.newaxis, :], t0, axis=1)
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(matrix_rep, ys, ts, test_size=0.2)

    # print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test), np.shape(t_train), np.shape(t_test))

    input_x = tf.convert_to_tensor(X_train.reshape(-1, 25))

    # Compile the model with the optimizer and loss function
    logger.info('COMPILING MODEL WITH THE OPTIMIZER AND LOSS FUNCTION...')
    model.compile(optimizer=config.optimizer, 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])
    logger.info('OPTIMIZER AND LOSS FUNCTION SUCCESSFULLY COMPILED')

    # Train the model
    logger.info('TRAINING MODEL...')
    #get phi(X), surrogate representation
    regularizer = tf.keras.regularizers.l2(l2=1.0)
    phi_X_train = tf.keras.layers.Dense(config.dim_in, activation='relu', kernel_regularizer=regularizer)(X_train)
    # print(phi_X_train)
    
    output = model(
    phi_X_train,
    training = True,

    )
    # encoded = model(
    # phi_X_train,
    # training = True,
    # )
    # decoded = model(
    # phi_X_train,
    # training = True,
    # )
    # print(f"Encoder Output: {encoded}")
    # print(f"Decoder Output: {decoded}")
    print(f"Model Output: {output}")
    logger.info('MODEL SUCCESSFULLY TRAINED!')

    # Save the model
    # logger.info('SAVING MODEL...')
    # model.save(f'{config.save_dir}/model')
    # logger.info('MODEL SUCCESSFULLY SAVED!')

if __name__ == '__main__':
    main()
