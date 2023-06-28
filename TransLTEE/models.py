from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, GRUCell
from transformers import TFBertModel, TFBertForMaskedLM

class SurrogateRepresentation(Model):
    def __init__(self, dim_in):
        super(SurrogateRepresentation, self).__init__()
        self.phi = Dense(dim_in)
        
    def call(self, x):
        return tf.nn.relu(self.phi(x))

class DoubleHeadRNN(Model):
    def __init__(self, input_dim, hidden_dim):
        super(DoubleHeadRNN, self).__init__()
        self.rnn0 = RNN(GRUCell(hidden_dim))  # GRU is used here, replace with LSTMCell for LSTM
        self.rnn1 = RNN(GRUCell(hidden_dim))

    def call(self, x):
        output0 = self.rnn0(x)
        output1 = self.rnn1(x)
        return output0, output1

class TransformerEncoder(Model):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.transformer = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, x):
        outputs = self.transformer(x)
        return outputs.last_hidden_state

class TransformerDecoder(Model):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.transformer = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

    def call(self, x):
        outputs = self.transformer(x)
        return outputs.logits

class MyModel(Model):
    def __init__(self, input_dim, hidden_dim):
        super(MyModel, self).__init__()
        self.surr_rep = SurrogateRepresentation(input_dim)
        self.double_head_rnn = DoubleHeadRNN(input_dim, hidden_dim)
        self.transformer_encoder = TransformerEncoder()

    def call(self, x):
        x = self.surr_rep(x)
        output0, output1 = self.double_head_rnn(x)
        encoded0 = self.transformer_encoder(output0)
        encoded1 = self.transformer_encoder(output1)
        return x, encoded0, encoded1
