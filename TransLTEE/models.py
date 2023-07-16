from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, GRUCell
from transformers import TFBertModel, TFBertForMaskedLM
import tensorflow as tf
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, MultiHeadAttention

'''
num_layers is the parameter for specifying how many iterations the encoder block should 
have. d_model is the dimensionality of the input, num_heads is the number of attention heads, 
and dff is the dimensionality of the feed-forward network. The rate parameter is for the dropout rate.
'''

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),  
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask=None):  # Added default value for mask
        if mask is None:
            mask = tf.ones_like(x)  # Use default mask of ones if none provided
        attn_output = self.mha(x, x, x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  

        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  

        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [TransformerEncoderBlock(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

    def call(self, x, training, mask):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),  
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask=None):  
        attn1 = self.mha1(x, x, x, look_ahead_mask)  # Self attention
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # Encoder-decoder attention
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [TransformerDecoderBlock(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask=None):
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        return x



class MyModel(Model):
    def __init__(self, input_dim, num_layers=7, num_heads=5, dff=50, dropout_rate=0.1):
        super(MyModel, self).__init__()
        self.regularizer = tf.keras.regularizers.l2(l2=1.0)
        self.input_phi = tf.keras.layers.Dense(input_dim, activation='relu', kernel_regularizer=self.regularizer)
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, d_model=input_dim, num_heads=num_heads, dff=dff, rate=dropout_rate)
        self.transformer_decoder = TransformerDecoder(num_layers=num_layers, d_model=input_dim, num_heads=num_heads, dff=dff, rate=dropout_rate)
        self.dense = tf.keras.layers.Dense(100)
        self.linear = tf.keras.layers.Dense(1)
        # self.softmax = tf.keras.layers.Softmax()

    def pdist2sq(self, X,Y):
        """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
        C = -2*tf.matmul(X,tf.transpose(Y))
        nx = tf.reduce_sum(tf.square(X),1,True)
        ny = tf.reduce_sum(tf.square(Y),1,True)
        D = (C + tf.transpose(ny)) + nx
        return D
    
    def safe_sqrt(self, x, lbound=1e-10):
        ''' Numerically safe version of TensorFlow sqrt '''
        return tf.sqrt(tf.clip_by_value(x, lbound, np.inf)) # x in [10^-10,+∞]
    
    def wasserstein(self, seq_len, X_ls,t,p,lam=10,its=10,sq=True,backpropT=False):
        """ Returns the Wasserstein distance between treatment groups """
        D=0
        for i in range(seq_len):
            X=X_ls[:,i,:]
            it = tf.where(t>0)[:,0]
            ic = tf.where(t<1)[:,0]
            Xc = tf.gather(X,ic)
            Xt = tf.gather(X,it)
            nc = tf.cast(tf.shape(Xc)[0],tf.float32)
            nt = tf.cast(tf.shape(Xt)[0],tf.float32)
    
            ''' Compute distance matrix'''
            if sq:
                M = self.pdist2sq(Xt,Xc)
            else:
                M = self.safe_sqrt(self.pdist2sq(Xt,Xc))
    
            ''' Estimate lambda and delta '''
            M_mean = tf.reduce_mean(M)
            M_drop = tf.nn.dropout(M,10/(nc*nt))
            delta = tf.stop_gradient(tf.reduce_max(M))
            eff_lam = tf.stop_gradient(lam/M_mean)
    
            ''' Compute new distance matrix '''
            Mt = M
            row = delta*tf.ones(tf.shape(M[0:1,:]))
            col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],0)
            Mt = tf.concat([M,row],0)
            Mt = tf.concat([Mt,col],1)
    
            ''' Compute marginal vectors '''
            a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))],0)
            b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))],0)
    
            ''' Compute kernel matrix'''
            Mlam = eff_lam*Mt
            K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
            U = K*Mt
            ainvK = K/a
    
            u = a
            for i in range(0,its):
                u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
            v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))
    
            T = u*(tf.transpose(v)*K)
    
            if not backpropT:
                T = tf.stop_gradient(T)
    
            E = T*Mt
            D += 2*tf.reduce_sum(E)
    
        return D

    def safe_sqrt(x, lbound=1e-10):
        ''' Numerically safe version of TensorFlow sqrt '''
        return tf.sqrt(tf.clip_by_value(x, lbound, np.inf)) # x in [10^-10,+∞]

    def pred_error(self, tar_out, tar_real):
        pred_y = tf.squeeze(tar_out)
        real_y = tf.cast(tf.squeeze(tar_real), tf.float32)
        # print(pred_y)
        # print(pred_y.shape, real_y.shape,pred_y.dtype, real_y.dtype)
        # print(tf.subtract(pred_y,tar_real))
        pred_error = tf.reduce_mean(tf.square(pred_y - real_y))
        print(pred_error,pred_error.shape)
        return pred_error
    
    def distance(self, encoded, seq_len, t):
        t = tf.cast(t, tf.float32)
        p_t = np.mean(t)
        dis = self.wasserstein(seq_len,encoded,t,p_t,lam=1,its=20)
        return dis

    def call(self, x, t0, t, tar_input, tar_real, training=False, mask=None):
        phi_x = self.input_phi(x)
        seq_len = tf.shape(phi_x)[1]
        if mask is None:
            mask = tf.ones((seq_len, seq_len))
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((tf.shape(tar_input)[1], tf.shape(tar_input)[1])), -1, 0)
        encoded = self.transformer_encoder(phi_x, training, mask)
        decoded = self.transformer_decoder(tar_input, encoded, training, look_ahead_mask)
        output = self.linear(decoded)
        print(encoded.shape, decoded.dtype, output.dtype)
        predicted_error = self.pred_error(output, tar_real)
        distance = self.distance(encoded, t0, t)
        # output = self.softmax(linear_output)
        # return self.dense(encoded), self.linear(decoded), encoded, decoded, output
        return predicted_error, distance

