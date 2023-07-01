import tensorflow as tf
import os
import numpy as np

SQRT_CONST = 1e-10

FLAGS = tf.compat.v1.flags

def to_device(data, device):
    """
    Move tensors to the specified device.
    data: Data to be moved. It can be a single tensor or a list or dictionary of tensors.
    device: Target device.
    """
    # Use .to() method to move data to device in TensorFlow
    if tf.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        raise TypeError('Unsupported data type {}'.format(type(data)))

def compute_accuracy(output, target):
    """
    Compute classification accuracy.
    output: Output from the model, shape=[batch_size, num_classes].
    target: Actual labels, shape=[batch_size].
    """
    pred = tf.argmax(output, axis=1)
    correct = tf.equal(pred, target)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy.numpy()

def save_model(model, config, epoch):
    """
    Save the model's state dict and optimizer's state dict.
    model: The model instance.
    config: The config instance.
    epoch: The current epoch number.
    """
    model_dir = os.path.join(config.save_dir, f'model_epoch_{epoch}.pt')
    tf.keras.models.save_model(model, model_dir)
    print(f"Model saved at {model_dir}")

def load_model(model, config, device, epoch):
    """
    Load the model's state dict and optimizer's state dict.
    model: The model instance.
    config: The config instance.
    device: The device instance.
    epoch: The epoch number to load.
    """
    model_dir = os.path.join(config.save_dir, f'model_epoch_{epoch}.pt')
    if os.path.exists(model_dir):
        model = tf.keras.models.load_model(model_dir)
        model = to_device(model, device)
        print(f"Model loaded from {model_dir}")
    else:
        print(f"No model found at {model_dir}, please check the epoch number.")
    return model

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.compat.v1.sqrt(tf.compat.v1.clip_by_value(x, lbound, np.inf))

def lindisc(X,p,t):
    ''' Linear MMD '''

    it = tf.compat.v1.where(t>0)[:,0]
    ic = tf.compat.v1.where(t<1)[:,0]

    Xc = tf.compat.v1.gather(X,ic)
    Xt = tf.compat.v1.gather(X,it)

    mean_control = tf.compat.v1.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.compat.v1.reduce_mean(Xt,reduction_indices=0)

    c = tf.compat.v1.square(2*p-1)*0.25
    f = tf.compat.v1.sign(p-0.5)

    mmd = tf.compat.v1.reduce_sum(tf.compat.v1.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = tf.compat.v1.where(t>0)[:,0]
    ic = tf.compat.v1.where(t<1)[:,0]

    Xc = tf.compat.v1.gather(X,ic)
    Xt = tf.compat.v1.gather(X,it)

    mean_control = tf.compat.v1.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.compat.v1.reduce_mean(Xt,reduction_indices=0)

    mmd = tf.compat.v1.reduce_sum(tf.compat.v1.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.compat.v1.where(t>0)[:,0]
    ic = tf.compat.v1.where(t<1)[:,0]

    Xc = tf.compat.v1.gather(X,ic)
    Xt = tf.compat.v1.gather(X,it)

    Kcc = tf.compat.v1.exp(-pdist2sq(Xc,Xc)/tf.compat.v1.square(sig))
    Kct = tf.compat.v1.exp(-pdist2sq(Xc,Xt)/tf.compat.v1.square(sig))
    Ktt = tf.compat.v1.exp(-pdist2sq(Xt,Xt)/tf.compat.v1.square(sig))

    m = tf.compat.v1.to_float(tf.compat.v1.shape(Xc)[0])
    n = tf.compat.v1.to_float(tf.compat.v1.shape(Xt)[0])

    mmd = tf.compat.v1.square(1.0-p)/(m*(m-1.0))*(tf.compat.v1.reduce_sum(Kcc)-m)
    mmd = mmd + tf.compat.v1.square(p)/(n*(n-1.0))*(tf.compat.v1.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.compat.v1.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.compat.v1.matmul(X,tf.compat.v1.transpose(Y))
    nx = tf.compat.v1.reduce_sum(tf.compat.v1.square(X),1,keep_dims=True)
    ny = tf.compat.v1.reduce_sum(tf.compat.v1.square(Y),1,keep_dims=True)
    D = (C + tf.compat.v1.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t):
    it = tf.compat.v1.where(t>0)[:,0]
    ic = tf.compat.v1.where(t<1)[:,0]
    Xc = tf.compat.v1.gather(X,ic)
    Xt = tf.compat.v1.gather(X,it)
    nc = tf.compat.v1.to_float(tf.compat.v1.shape(Xc)[0])
    nt = tf.compat.v1.to_float(tf.compat.v1.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(seq_len, X_ls,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """
    D=0
    for i in range(seq_len):
        X=X_ls[:,i,:]
        it = tf.compat.v1.where(t>0)[:,0]
        ic = tf.compat.v1.where(t<1)[:,0]
        Xc = tf.compat.v1.gather(X,ic)
        Xt = tf.compat.v1.gather(X,it)
        nc = tf.compat.v1.to_float(tf.compat.v1.shape(Xc)[0])
        nt = tf.compat.v1.to_float(tf.compat.v1.shape(Xt)[0])

        ''' Compute distance matrix'''
        if sq:
            M = pdist2sq(Xt,Xc)
        else:
            M = safe_sqrt(pdist2sq(Xt,Xc))

        ''' Estimate lambda and delta '''
        M_mean = tf.compat.v1.reduce_mean(M)
        M_drop = tf.compat.v1.nn.dropout(M,10/(nc*nt))
        delta = tf.compat.v1.stop_gradient(tf.compat.v1.reduce_max(M))
        eff_lam = tf.compat.v1.stop_gradient(lam/M_mean)

        ''' Compute new distance matrix '''
        Mt = M
        row = delta*tf.compat.v1.ones(tf.compat.v1.shape(M[0:1,:]))
        col = tf.compat.v1.concat([delta*tf.compat.v1.ones(tf.compat.v1.shape(M[:,0:1])),tf.compat.v1.zeros((1,1))],0)
        Mt = tf.compat.v1.concat([M,row],0)
        Mt = tf.compat.v1.concat([Mt,col],1)

        ''' Compute marginal vectors '''
        a = tf.compat.v1.concat([p*tf.compat.v1.ones(tf.compat.v1.shape(tf.compat.v1.where(t>0)[:,0:1]))/nt, (1-p)*tf.compat.v1.ones((1,1))],0)
        b = tf.compat.v1.concat([(1-p)*tf.compat.v1.ones(tf.compat.v1.shape(tf.compat.v1.where(t<1)[:,0:1]))/nc, p*tf.compat.v1.ones((1,1))],0)

        ''' Compute kernel matrix'''
        Mlam = eff_lam*Mt
        K = tf.compat.v1.exp(-Mlam) + 1e-6 # added constant to avoid nan
        U = K*Mt
        ainvK = K/a

        u = a
        for i in range(0,its):
            u = 1.0/(tf.compat.v1.matmul(ainvK,(b/tf.compat.v1.transpose(tf.compat.v1.matmul(tf.compat.v1.transpose(u),K)))))
        v = b/(tf.compat.v1.transpose(tf.compat.v1.matmul(tf.compat.v1.transpose(u),K)))

        T = u*(tf.compat.v1.transpose(v)*K)

        if not backpropT:
            T = tf.compat.v1.stop_gradient(T)

        E = T*Mt
        D += 2*tf.compat.v1.reduce_sum(E)

    return D

def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w
