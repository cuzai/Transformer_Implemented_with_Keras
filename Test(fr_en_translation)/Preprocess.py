import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sentencepiece as spm

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Tokenizer():
    def __init__(self, language):
        self.language = language

    def train(self, data): # Train sentencepeice
        # Save data for sentence piece to train
        with open(f'{self.language}_stpc.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(data))
        
        # Train spm
        spm.SentencePieceTrainer.Train(f"--input={self.language}_stpc.txt" +
                                        f" --model_prefix={self.language}" + 
                                        " --vocab_size=8000" +
                                        " --model_type=bpe" +
                                        " --max_sentence_length=999999"
                                        " --pad_id=0 --pad_piece=<PAD>" + # pad (0)
                                        " --unk_id=1 --unk_piece=<UNK>" + # unknown (1)
                                        " --bos_id=2 --bos_piece=<SOS>" + # begin of sequence (2)
                                        " --eos_id=3 --eos_piece=<EOS>" # end of sequence (3)
                                        )

    def tokenize(self, data, training):
        # Train sentencepiece only for the first time
        if not os.path.isfile("english_stpc.txt") or not os.path.isfile("french_stpc.txt") or training==True: 
            self.train(data)

        # Load
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{self.language}.model")
        
        return sp

class Preprocess():
    def __init__(self, BATCH_SIZE, train, valid, test, training=False):
        self.BATCH_SIZE = BATCH_SIZE
        self.train_dataset, self.valid_dataset, self.test = self.preprocess(train, valid, test, training)

    def get_tokenized(self, fr_tokenizer, en_tokenizer, train, valid, test): # Tokenize data
        train["fr_tokens"], train["en_tokens"] = train["French words/sentences"].apply(lambda x: [fr_tokenizer.bos_id()] + fr_tokenizer.encode_as_ids(x) + [fr_tokenizer.eos_id()]), train["English words/sentences"].apply(lambda x: [en_tokenizer.bos_id()] + en_tokenizer.encode_as_ids(x) + [en_tokenizer.eos_id()])
        valid["fr_tokens"], valid["en_tokens"] = valid["French words/sentences"].apply(lambda x: fr_tokenizer.encode_as_ids(x)), valid["English words/sentences"].apply(lambda x: en_tokenizer.encode_as_ids(x))
        test["fr_tokens"], test["en_tokens"] = test["French words/sentences"].apply(lambda x: fr_tokenizer.encode_as_ids(x)), test["English words/sentences"].apply(lambda x: en_tokenizer.encode_as_ids(x))
        return train, valid, test

    def get_padded(self, train, valid, test): # Pad data
        self.train = train
        fr_train = pad_sequences(train["fr_tokens"], padding="post", dtype=np.float32)
        fr_valid = pad_sequences(valid["fr_tokens"], padding="post", dtype=np.float32)

        en_train = pad_sequences(train["en_tokens"], padding="post", dtype=np.float32)
        en_valid = pad_sequences(valid["en_tokens"], padding="post", dtype=np.float32)

        return fr_train, fr_valid, en_train, en_valid

    def get_buffer_size(self, num): # Get data.shape[0] and ceil for sufficient buffer size.
        # Make it as a string
        num = str(num) 
        first_digit = num[0]

        # Make ceil array
        arr = np.zeros(shape= (len(num),), dtype=np.int16)
        arr[0] = int(first_digit) + 1

        # Return it to a number again
        arr = int("".join([str(i) for i in arr]))
        
        return arr
    
    def get_dataset(self):
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {"enc_input":self.fr_train,
            "dec_input":self.en_train[:, :-1] # Decoder input does not need <EOS>
            },
            {"dec_output":self.en_train[:, 1:]} # Decoder output does not need <SOS>
            ))

        valid_dataset = tf.data.Dataset.from_tensor_slices((
            {"enc_input":self.fr_train,
            "dec_input":self.en_train[:, :-1] # Decoder input does not need <EOS>
            },
            {"dec_output":self.en_train[:, 1:]} # Decoder output does not need <SOS>
            ))
    
        train_buffer_size = self.get_buffer_size(max(self.fr_train.shape[0], self.en_train.shape[0]))
        train_dataset = train_dataset.cache().shuffle(buffer_size=train_buffer_size).batch(batch_size=self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

        valid_buffer_size = self.get_buffer_size(max(self.fr_valid.shape[0], self.en_valid.shape[0]))
        valid_dataset = valid_dataset.cache().shuffle(buffer_size=valid_buffer_size).batch(batch_size=self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        
        return train_dataset, valid_dataset

    def get_shape(self, name):
        if name == "fr_train": return self.fr_train.shape
        elif name == "fr_valid": return self.fr_valid.shape
        elif name == "en_train": return self.en_train.shape
        elif name == "en_valid": return self.en_valid.shape

    def preprocess(self, train, valid, test, training):
        # Tokenize
        self.fr_tokenizer, self.en_tokenizer = Tokenizer("french").tokenize(train["French words/sentences"], training=training), Tokenizer("english").tokenize(train["English words/sentences"], training)
        train, valid, test = self.get_tokenized(self.fr_tokenizer, self.en_tokenizer, train, valid, test)

        # Pad_sequence
        self.fr_train, self.fr_valid, self.en_train, self.en_valid = self.get_padded(train, valid, test)

        # Make tensorflow dataset
        train_dataset, valid_dataset = self.get_dataset()
        
        return train_dataset, valid_dataset, test

if __name__ == "__main__":
    BATCH_SIZE = 64

    gpus = tf.config.list_physical_devices(device_type="GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)

    data = pd.read_csv("./eng_-french.csv")
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    valid, test = train_test_split(train, test_size=0.1, random_state=0)

    Preprocess(BATCH_SIZE, train, valid, test)