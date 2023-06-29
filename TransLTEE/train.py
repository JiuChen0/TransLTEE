import tensorflow as tf
from models import MyModel
from losses import compute_train_loss, compute_valid_loss
from config import Config
import numpy as np

def get_data(file):
    data = np.loadtxt(file, delimiter=',')
    treatment = data[:, 0:1]
    y = data[:, 1:]
    return treatment, y

def train():
    config = Config()

    for i in range(1, 11): # Loop through all generated datasets
        # Load data
        treatment_train, y_train = get_data('data/NEWS/Series_y_' + str(i) + '.txt')
        # treatment_valid, y_valid = get_data('valid.csv')

        model = MyModel(config.input_dim, config.hidden_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        for epoch in range(config.epochs):
            # Training
            model.train()
            with tf.GradientTape() as tape:
                x_train = treatment_train
                t_train = np.ones((x_train.shape[0], 1, 1)) # Assuming treatment is always 1 for training
                y_train_ = np.ones((x_train.shape[0], 1)) # Assuming y_ is always 1 for training
                loss = compute_train_loss(model, (x_train, t_train, y_train_), config.loss_weights)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # # Validation
            # model.eval()
            # loss = compute_valid_loss(model, (x_valid, t_valid, y_valid_), config.loss_weights)

            # Save the model
            model.save_weights(f'{config.save_dir}/model_{epoch}.h5')
