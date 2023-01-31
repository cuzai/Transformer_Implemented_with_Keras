# Transformer Implemented with Tensorflow/Keras
## 1. Objective: Implement Transformer from scratch using Tensorflow/Keras
 - Transformer is one of the most critical back bone architecture for SOTA NLP models (BERT, GPT...)
 - Above can be applied not only to NLP problems but also to any other sequential data forecasting problems
 - Rather than using existing libraries such as `torch.nn.Transformer` or `huggingface.transformer`, I decided to implement it from scratch because of two reasons below.
   - It will help me understand transformer much deeper.
   - It allows much more customization so I can make the model best fit to any other situations(or problems).

## 2. Usage
 - It seems there is no existing transformer related layer or model implemented with Tensorflow/Keras so hopefully this code would help somebody in trouble with using transformer with Tensorflow/Keras.  
 - Simply follow the code below and if necessary, refer to `Test(fr_en_translation)` folder.
 ```python
 from Transformer.Transformer import Transformer

# Transformer hyperparameters. Numbers in comment is the hyperparameters suggested from the papaer
BATCH_SIZE = 64
D_MODEL = 128 # 512
RATE = 0.1
NUM_LAYERS = 4 # 6
NUM_HEADS = 4 # 8
EPSILON = 1e-6
D_PFF = 512 # 2048

# Input layers
enc_input = Input(shape=(None,), name="enc_input")
dec_input = Input(shape=(None,), name="dec_input")

# Make transformer with above-defined hyperparameters
transformer = Transformer(**params, name="dec_output")
dec_output = transformer(enc_input, dec_input, training=True)

model = tf.keras.models.Model(inputs=(enc_input, dec_input), outputs=dec_output)
 ```