import tensorflow as tf
from models import MyModel
from losses import compute_train_loss, compute_valid_loss
from data import get_dataloader
from config import Config

def train():
    config = Config()
    train_loader = get_dataloader('train.csv', config.batch_size)
    valid_loader = get_dataloader('valid.csv', config.batch_size)

    model = MyModel(config.input_dim, config.hidden_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    for epoch in range(config.epochs):
        # Training
        model.train()
        for batch in train_loader:
            with tf.GradientTape() as tape:
                loss = compute_train_loss(model, batch, config.loss_weights)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Validation
        model.eval()
        for batch in valid_loader:
            loss = compute_valid_loss(model, batch, config.loss_weights)

        # Save the model
        model.save_weights(f'{config.save_dir}/model_{epoch}.h5')
