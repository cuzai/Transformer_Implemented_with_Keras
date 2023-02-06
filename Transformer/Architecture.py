import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout

class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, rate, max_seq_len, **kargs):
        super().__init__(**kargs)
        self.d_model = d_model
        self.positional_encoded = self.positional_encoding(max_seq_len)

        self.dropout = Dropout(rate)
 
    def positional_encoding(self, max_seq_len):
        # PE(pos, 2i) = sin(pos/10000^{2i/d_model}), 
        # PE(pos, 2i+1) = cos(pos/10000^{2i/d_model})

        # it is difficult to itemize a tensor, e.g. [:, 0::2], we will use numpy for this one only)
        # it is also impossible numpy to deal with tensor object(e.g seq_len from a tensor), we need to explicitly offer seq_len from the beggining

        position = np.arange(max_seq_len)[..., np.newaxis]
        i = np.arange(self.d_model) // 2
        i = 1 / np.power(10000, 2*i/self.d_model)[np.newaxis, ...]
        positional_encoded = np.matmul(position, i)

        positional_encoded[:, 0::2] = np.sin(positional_encoded[:, 0::2])
        positional_encoded[:, 1::2] = np.cos(positional_encoded[:, 1::2])

        return tf.cast(positional_encoded, tf.float32)
  
    def call(self, input):
        embedded = input * tf.sqrt(tf.cast(self.d_model, tf.float32)) 
        out = embedded + self.positional_encoded[:tf.shape(input)[1], :]
        out = self.dropout(out)
        
        return out


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kargs):
        super().__init__(**kargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads; assert d_model % num_heads == 0

        self.query_dense, self.key_dense, self.value_dense = Dense(d_model), Dense(d_model), Dense(d_model)
        self.dense = Dense(d_model)
    
    def split_heads(self, x):
        x = tf.reshape(x, shape=(tf.shape(x)[0], -1, self.num_heads, self.depth)) # first split d_model into num_heads and depth
        x = tf.transpose(x, perm=[0,2,1,3]) # and then transpose
        return x

    def undo_split_heads(self, x):
        x = tf.transpose(x, perm=[0,2,1,3])
        x = tf.reshape(x, shape=(tf.shape(x)[0], -1, self.d_model))
        return x

    def scaled_dot_product_attention(self, q, k, v, mask):
        # softmax(QK^T/sqrt(d_k))·V
        qk = tf.matmul(q, k, transpose_b=True)
        d_k = tf.cast(tf.shape(k), tf.float32)[-1]
        logits = qk / tf.sqrt(d_k)

        if mask is not None:
            logits += mask * -1e9
        
        attention_weight = tf.nn.softmax(logits, axis=-1) # row wise softmax
        output = tf.matmul(attention_weight, v)

        return output
        
    def call(self, query, key, value, mask):
        query_weight, key_weight, value_weight = self.query_dense(query), self.key_dense(key), self.value_dense(value)
        query_splitted, key_splitted, value_splitted = self.split_heads(query_weight), self.split_heads(key_weight), self.split_heads(value_weight)
        out = self.scaled_dot_product_attention(query_splitted, key_splitted, value_splitted, mask)
        out = self.undo_split_heads(out)
        out = self.dense(out)

        return out

class PointwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_pff, d_model, **kargs):
        super().__init__(**kargs)
        self.dense1 = Dense(d_pff, activation="relu")
        self.dense2 = Dense(d_model)
    
    def call(self, input):
        out = self.dense1(input)
        out = self.dense2(out)
        return out
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)

        self.multihead_self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1, self.dropout2 = Dropout(rate), Dropout(rate)
        self.layer_norm1, self.layer_norm2 = LayerNormalization(epsilon=epsilon), LayerNormalization(epsilon=epsilon)
        self.pff = PointwiseFeedForward(d_pff, d_model)

    def call(self, embedded_input, padding_mask, training):
        attentioned_output = self.multihead_self_attention(embedded_input, embedded_input, embedded_input, padding_mask)
        attentioned_output = self.dropout1(attentioned_output, training=training)
        attentioned_output = self.layer_norm1(embedded_input + attentioned_output)

        pff_output = self.pff(attentioned_output)
        pff_output = self.dropout2(pff_output, training=training)
        pff_output = self.layer_norm2(attentioned_output + pff_output)
        
        return pff_output

class StackedEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)
        self.encoder_li = [Encoder(d_model, num_heads, rate, epsilon, d_pff) for _ in range(num_layers)]
    
    def call(self, x, padding_mask, training):
        for encoder in self.encoder_li:
            x = encoder(x, padding_mask, training)

        return x
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)
        self.multihead_self_attention = MultiHeadAttention(d_model, num_heads)
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.pff = PointwiseFeedForward(d_pff, d_model)
        
        self.dropout1, self.dropout2, self.dropout3 = Dropout(rate), Dropout(rate), Dropout(rate)
        self.layer_norm1, self.layer_norm2, self.layer_norm3 = LayerNormalization(epsilon=epsilon), LayerNormalization(epsilon=epsilon), LayerNormalization(epsilon=epsilon)
    
    def call(self, embedded_input, lookahead_mask, training, enc_output, padding_mask):
        attentioned_output1 = self.multihead_self_attention(embedded_input, embedded_input, embedded_input, lookahead_mask) # teacher_force
        attentioned_output1 = self.dropout1(attentioned_output1, training=training)
        attentioned_output1 = self.layer_norm1(attentioned_output1 + embedded_input)

        attentioned_output2 = self.multihead_attention(attentioned_output1, enc_output, enc_output, padding_mask)
        attentioned_output2 = self.dropout2(attentioned_output2, training=training)
        attentioned_output2 = self.layer_norm2(attentioned_output1 + attentioned_output2)

        pff_output = self.pff(attentioned_output2)
        pff_output = self.dropout3(pff_output, training=training)
        pff_output = self.layer_norm3(attentioned_output2 + pff_output)
        
        return pff_output

class StackedDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, rate, epsilon, d_pff, **kargs):
        super().__init__(**kargs)
        self.decoder_li = [Decoder(d_model, num_heads, rate, epsilon, d_pff) for _ in range(num_layers)]
    
    def call(self, x, lookahead_mask, training, enc_output, padding_mask):
        for decoder in self.decoder_li:
            x = decoder(x, lookahead_mask, training, enc_output, padding_mask)
        return x