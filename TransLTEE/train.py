from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np

def train():
    config = Config()
        treatment_train, y_train = get_data('data/NEWS/Series_y_' + '1' + '.txt')

        matrix_rep = np.repeat(matrix[:, np.newaxis, :], t0, axis=1)
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(matrix_rep, ys, ts, test_size=0.2)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        model = MyModel(config.input_dim, config.hidden_dim)
        optimizer = Adam(learning_rate=config.learning_rate)
        model.compile(optimizer=optimizer, loss='mse') # Assuming 'mse' loss here, change as needed

        for epoch in range(config.epochs):
            for i in range(0, n_train, config.batch_size):
                x_batch = X_train[i:i+config.batch_size]
                t_batch = t_train[i:i+config.batch_size]
                y_batch = y_train[i:i+config.batch_size]
                
                loss = model.train_on_batch(x_batch, y_batch) # This will update the weights of the model
                
                # Check the performance every output_delay epochs
                if epoch % config.output_delay == 0 or epoch == config.epochs - 1:
                    y_pred = model.predict(X_test)
                    # Compute any metrics you want here
                    # RMSE example
                    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
                    print(f'Epoch {epoch} - RMSE: {rmse}')
                
            # Save the model
            model.save_weights(f'{config.save_dir}/model_{epoch}.h5')

