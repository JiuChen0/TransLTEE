import tensorflow as tf

def compute_short_term_loss(surr_rep, encoded):
    # TODO: Implement the calculation of short-term loss
    # MSE Loss
    loss = tf.keras.losses.MeanSquaredError()(surr_rep, encoded)
    return loss

def compute_primary_outcome_loss(surr_rep, targets):
    # TODO: Implement the calculation of primary outcome loss
    # Cross Entropy Loss
    loss = tf.keras.losses.CategoricalCrossentropy()(surr_rep, targets)
    return loss

def compute_IPM_loss(encoded0, encoded1):
    # TODO: Implement the calculation of IPM loss (Wasserstein-1 distance)
    # Here we use a placeholder
    loss = tf.constant(0.0)
    return loss

def compute_train_loss(model, inputs, targets, weights):
    """
    Compute training loss
    model: The model object
    batch: A batch of training data
    weights: Weights for the three losses
    """
    surr_rep, encoded0, encoded1 = model(inputs, training=True)
    loss1 = compute_short_term_loss(surr_rep, encoded0)
    loss2 = compute_primary_outcome_loss(surr_rep, targets)
    loss3 = compute_IPM_loss(encoded0, encoded1)
    loss = weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3
    return loss

def compute_valid_loss(model, inputs, targets, weights):
    """
    Compute validation loss
    model: The model object
    batch: A batch of validation data
    weights: Weights for the three losses
    """
    surr_rep, encoded0, encoded1 = model(inputs, training=False)
    loss1 = compute_short_term_loss(surr_rep, encoded0)
    loss2 = compute_primary_outcome_loss(surr_rep, targets)
    loss3 = compute_IPM_loss(encoded0, encoded1)
    loss = weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3
    return loss
