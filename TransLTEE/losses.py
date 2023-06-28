import tensorflow as tf

class WassersteinCalculator:
    def __init__(self, lam=10, its=10, sq=False, backpropT=False):
        self.lam = lam
        self.its = its
        self.sq = sq
        self.backpropT = backpropT

    def safe_sqrt(self, x, lbound=1e-10):
        return tf.sqrt(tf.clip_by_value(x, lbound, float('inf')))

    def pdist2sq(self, A, B):
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
        D = tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0)
        return D

    def wasserstein(self, seq_len, X_ls, t, p):
        D = 0
        for i in range(seq_len):
            X = X_ls[:, i, :]
            it = tf.where(t > 0)[:, 0]
            ic = tf.where(t < 1)[:, 0]
            Xc = tf.gather(X, ic)
            Xt = tf.gather(X, it)
            nc = tf.to_float(tf.shape(Xc)[0])
            nt = tf.to_float(tf.shape(Xt)[0])

            if self.sq:
                M = self.pdist2sq(Xt, Xc)
            else:
                M = self.safe_sqrt(self.pdist2sq(Xt, Xc))

            M_mean = tf.reduce_mean(M)
            M_drop = tf.nn.dropout(M, 10 / (nc * nt))
            delta = tf.stop_gradient(tf.reduce_max(M))
            eff_lam = tf.stop_gradient(self.lam / M_mean)

            Mt = M
            row = delta * tf.ones_like(M[0:1, :])
            col = tf.concat([delta * tf.ones_like(M[:, 0:1]), tf.zeros((1, 1))], 0)
            Mt = tf.concat([M, row], 0)
            Mt = tf.concat([Mt, col], 1)

            a = tf.concat([p * tf.ones_like(tf.where(t > 0)[:, 0:1]) / nt, (1 - p) * tf.ones((1, 1))], 0)
            b = tf.concat([(1 - p) * tf.ones_like(tf.where(t < 1)[:, 0:1]) / nc, p * tf.ones((1, 1))], 0)

            Mlam = eff_lam * Mt
            K = tf.exp(-Mlam) + 1e-6
            U = K * Mt
            ainvK = K / a

            u = a
            for _ in range(self.its):
                u = 1.0 / tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K))))
            v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

            T = u * (tf.transpose(v) * K)
            E = T * Mt
            D += tf.reduce_sum(E)
        return D

def compute_short_term_loss(surr_rep, encoded):
    # MSE Loss
    loss = tf.keras.losses.MeanSquaredError()(surr_rep, encoded)
    return loss

def compute_primary_outcome_loss(surr_rep, targets):
    # Cross Entropy Loss
    loss = tf.keras.losses.CategoricalCrossentropy()(surr_rep, targets)
    return loss

def compute_IPM_loss(seq_len, X_ls, t, p):
    w_calculator = WassersteinCalculator()
    loss = w_calculator.wasserstein(seq_len, X_ls, t, p)
    return loss

def compute_train_loss(model, inputs, targets, weights, seq_len, X_ls, t, p):
    surr_rep, encoded0, encoded1 = model(inputs, training=True)
    loss1 = compute_short_term_loss(surr_rep, encoded0)
    loss2 = compute_primary_outcome_loss(surr_rep, targets)
    loss3 = compute_IPM_loss(seq_len, X_ls, t, p)
    loss = weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3
    return loss

def compute_valid_loss(model, inputs, targets, weights, seq_len, X_ls, t, p):
    surr_rep, encoded0, encoded1 = model(inputs, training=False)
    loss1 = compute_short_term_loss(surr_rep, encoded0)
    loss2 = compute_primary_outcome_loss(surr_rep, targets)
    loss3 = compute_IPM_loss(seq_len, X_ls, t, p)
    loss = weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3
    return loss
