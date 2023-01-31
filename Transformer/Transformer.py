import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout

from Transformer.Architecture import *

class Transformer(tf.keras.layers.Layer):
    def __init__(self, enc_vocab_size, dec_vocab_size, d_model, max_seq_len, rate, num_layers, num_heads, epsilon, d_pff, **kargs):
        super().__init__(**kargs)
        self.max_seq_len = max_seq_len

        self.enc_embedded = Embedding(enc_vocab_size, d_model)
        self.dec_embedded = Embedding(dec_vocab_size, d_model)

        self.positional_encoder = PositionalEncoder(enc_vocab_size, d_model, rate, max_seq_len)
        
        self.stacked_encoder = StackedEncoder(num_layers, d_model, num_heads, rate, epsilon, d_pff)
        self.stacked_decoder = StackedDecoder(num_layers, d_model, num_heads, rate, epsilon, d_pff)

        self.final_dense = Dense(dec_vocab_size)
    
    def create_padding_mask(self, x):
        # tf.print(tf.shape(x))
        padding_mask = tf.cast(tf.equal(0., x), tf.float32) # each row is a sentence
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :] # split in order to apply to look ahead mask as well as scaled dot product attention
        return padding_mask

    def create_lookahead_mask(self, x):
        size = tf.shape(x)[-1]
        lookahead_mask = 1 - tf.linalg.band_part(tf.ones(shape=(size, size)), -1, 0) # Look ahead mask is applied to K·Q (seq_len, seq_len). Note the shape of the mask
        padding_mask = self.create_padding_mask(x) 

        return tf.maximum(lookahead_mask, padding_mask)

    def call(self, enc_input, dec_input, training):
        enc_padding_mask = self.create_padding_mask(enc_input) # padding mask for encoder
        dec_lookahead_mask = self.create_lookahead_mask(dec_input) # look ahead mask for decoder sub layer1 (self attention)
        dec_padding_mask = self.create_padding_mask(enc_input) # padding mask for decoder sub layer2 masks encoder output. so the shape of it refers to enc_input

        enc_embed = self.enc_embedded(enc_input)
        enc_embed = self.positional_encoder(enc_embed)
        enc_output = self.stacked_encoder(enc_embed, enc_padding_mask, training)

        dec_embed = self.dec_embedded(dec_input)
        dec_embed = self.positional_encoder(dec_embed)
        dec_output = self.stacked_decoder(dec_embed, dec_lookahead_mask, training, enc_output, dec_padding_mask)
        
        output = self.final_dense(dec_output)

        return output