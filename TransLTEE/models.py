from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, GRUCell
from transformers import TFBertModel, TFBertForMaskedLM
import tensorflow as tf

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
        attn_output, _ = self.mha(x, x, x, mask)  
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

class SurrogateRepresentation(Model):
    def __init__(self, dim_in):
        super(SurrogateRepresentation, self).__init__()
        self.phi = Dense(dim_in)
        
    def call(self, x):
        return tf.nn.relu(self.phi(x))

# class DoubleHeadRNN(Model):
#     def __init__(self, input_dim, hidden_dim):
#         super(DoubleHeadRNN, self).__init__()
#         self.rnn0 = RNN(GRUCell(hidden_dim))  # GRU is used here, replace with LSTMCell for LSTM
#         self.rnn1 = RNN(GRUCell(hidden_dim))

#     def call(self, x):
#         output0 = self.rnn0(x)
#         output1 = self.rnn1(x)
#         return output0, output1

# class TransformerEncoder(Model):
#     def __init__(self):
#         super(TransformerEncoder, self).__init__()
#         self.transformer = TFBertModel.from_pretrained('bert-base-uncased')

#     def call(self, x):
#         outputs = self.transformer(x)
#         return outputs.last_hidden_state

# class TransformerDecoder(Model):
#     def __init__(self):
#         super(TransformerDecoder, self).__init__()
#         self.transformer = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

#     def call(self, x):
#         outputs = self.transformer(x)
#         return outputs.logits

class MyModel(Model):
    def __init__(self, input_dim, num_layers=2, num_heads=5, dff=50, dropout_rate=0.1):
        super(MyModel, self).__init__()
        self.surr_rep = SurrogateRepresentation(input_dim)
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, d_model=input_dim, num_heads=num_heads, dff=dff, rate=dropout_rate)
        self.dense = tf.keras.layers.Dense(10)

    def call(self, x, training=False, mask=None):
        x = self.surr_rep(x)
        print(x)
        # output0, output1 = self.double_head_rnn(x)
        encoded = self.transformer_encoder(x, training, mask)
        return self.dense(encoded)
