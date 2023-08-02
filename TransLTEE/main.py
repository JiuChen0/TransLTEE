import tensorflow as tf
from models import MyModel  # Please replace with your actual model
from config import Config
import numpy as np
from data import get_dataloader
import logging
import random
import time


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
    t0=50

    # for j in range(1, 11):

    TY = np.loadtxt('./data/IHDP/csv/ihdp_npci_' + "1" + '.csv', delimiter=',')
    matrix = TY[:, 5:]
    N = TY.shape[0]

    out_treat = np.loadtxt('./data/IHDP/Series_y_' + "1" + '.txt', delimiter=',')
    ts = out_treat[:, 0]
    ts = np.reshape(ts, (N, 1))
    # ys = np.concatenate((out_treat[:, 1:(t0 + 1)], out_treat[:, -1].reshape(N, 1)), axis=1)
    ys = out_treat[:, 1:(t0 + 2)]

    groundtruth0 = np.loadtxt('./data/IHDP/Series_groundtruth_' + '1' + '.txt')
    groundtruth0 = np.reshape(groundtruth0[:t0], (t0, 1))
    print(groundtruth0.shape)

    from sklearn.model_selection import train_test_split

    matrix_rep = np.repeat(matrix[:, np.newaxis, :], t0, axis=1)
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(matrix_rep, ys, ts, test_size=0.2)

    n_train, n_test = X_train.shape[0], X_test.shape[0]

    print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test), np.shape(t_train), np.shape(t_test))

    input_x = tf.convert_to_tensor(X_train.reshape(-1, 25))


    # print(tar_train.shape, tar_real.shape)

    # t_train = tf.expand_dims(t_train,-1)
    # t_test = tf.expand_dims(t_test,-1)

    # print(np.shape(tar_train))

    # Compile the model with the optimizer and loss function
    logger.info('COMPILING MODEL WITH THE OPTIMIZER AND LOSS FUNCTION...')
    # model.compile(optimizer=config.optimizer, 
    #               loss=tf.keras.losses.CategoricalCrossentropy(), 
    #               metrics=['accuracy'])
    logger.info('OPTIMIZER AND LOSS FUNCTION SUCCESSFULLY COMPILED')

    # Train the model
    logger.info('TRAINING MODEL...')
    #get phi(X), surrogate representation
    # regularizer = tf.keras.regularizers.l2(l2=1.0)
    # phi_X_train = tf.keras.layers.Dense(config.dim_in, activation='relu', kernel_regularizer=regularizer)(X_train)
    # print(phi_X_train)


    ## Gradient Descent
    optimizer = config.optimizer
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def train_step(X_train, t0, t_train, tar_train, tar_real, groundtruth, gama = config.gama):

        with tf.GradientTape() as tape:
            predictions, predict_error, distance = model(
                X_train, t0, t_train, tar_train, tar_real,
                training = True,
                )
            
            CF_predictions, _, _ = model(
                X_train, t0, (1-t_train), tar_train, tar_real,
                training = True,
                )
            # loss_groundtruth = train_loss(abs(CF_predictions), groundtruth)
            # loss = predict_error + loss_groundtruth + gama*distance
            pred_effect = tf.reduce_mean(abs(predict_error-CF_predictions),axis=0)
            # print(pred_effect)
            loss = predict_error + gama*distance

        gradients = tape.gradient(loss, model.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        # train_accuracy(pred_effect, groundtruth)
        train_accuracy(tar_real, predictions)
        # print([CF_predictions])

## Training
    for epoch in range(config.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        start = time.time()

        # Let's split the data into batches
        num_batches = n_train // config.batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = start_idx + config.batch_size

            x_batch = X_train[start_idx:end_idx, :, :]
            t_batch = t_train[start_idx:end_idx, :]
            y_batch = y_train[start_idx:end_idx, :]
            tar_batch = tf.expand_dims(y_batch[:,:-1],-1)
            tar_real_batch = tf.expand_dims(y_batch[:, 1:],-1)

            train_step(x_batch, t0, t_batch, tar_batch, tar_real_batch, groundtruth0, gama=0.01)

        # If there are any remaining samples, pack them into a mini-batch and use them for training
        if n_train % config.batch_size != 0:
            start_idx = num_batches * config.batch_size
            x_batch = X_train[start_idx:, :, :]
            t_batch = t_train[start_idx:, :]
            y_batch = y_train[start_idx:, :]
            tar_batch = tf.expand_dims(y_batch[:,:-1],-1)
            tar_real_batch = tf.expand_dims(y_batch[:, 1:],-1)

            train_step(x_batch, t0, t_batch, tar_batch, tar_real_batch, groundtruth0, gama=0.01)

        # Timing and printing happens here after each epoch
        epoch_time = time.time() - start
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {epoch_time:.2f} secs\n')

    # output = model(
    # X_train, t0, t_train, tar_train, tar_real,
    # training = True,
    # )
    # print(f"Model Output: {output}")

    # pred_y = output[4]
    # pred_y = tf.squeeze(pred_y)
    # tar_real = tf.squeeze(tar_real)
    # tar_real = tf.cast(tar_real,tf.float32)
    # # print(pred_y)
    # print(pred_y.shape, tar_real.shape,pred_y.dtype, tar_real.dtype)
    # # print(tf.subtract(pred_y,tar_real))
    # pred_error = tf.reduce_mean(tf.square(pred_y - tar_real))
    # print(pred_error,pred_error.shape)

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


    logger.info('MODEL SUCCESSFULLY TRAINED!')

    # Save the model
    # logger.info('SAVING MODEL...')
    # model.save(f'{config.save_dir}/model')
    # logger.info('MODEL SUCCESSFULLY SAVED!')

if __name__ == '__main__':
    main()