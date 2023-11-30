import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
from keras.layers import Layer
import json
from keras.layers import Attention
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Input
from keras import Model
from keras.layers import Dot
from keras.layers import Concatenate


def create_input_sequences():
    queries = pd.read_json("train.jsonl", lines=True)
    tables = pd.read_json("train.tables.jsonl", lines=True)
    tables.index = tables["id"]
    sql_vocab = ["SYMSYMS", "SYMSELECT", "SYMWHERE", "SYMAND", "SYMCOL", "SYMTABLE", "SYMCAPTION", "SYMPAGE",
                 "SYMSECTION", "SYMOP", "SYMCOND", "SYMQUESTION", "SYMAGG", "SYMAGGOPS", "SYMCONDOPS", "SYMAGGOPS",
                 "MAX", "MIN", "COUNT", "SUM", "AVG", "SYMCONDOPS", "=", ">", "<", "OP", "SYMTABLE", "SYMCOL"]
    sequences = []
    for q, table in zip(queries['question'], queries['table_id']):
        question = q.split()
        header = tables.loc[table]['header']
        seq = []
        for vocab in sql_vocab:
            seq.append(vocab)
        for col in header:
            seq.append(col)
            seq.append("SYMCOL")
        seq.append("SYMQEUSTION")
        sequences.append(seq + question)
    input_sequences = pd.DataFrame(sequences)
    input_sequences.to_csv("input_seqs")

def create_output_sequences():
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    queries = pd.read_json("train.jsonl", lines=True)
    tables = pd.read_json("train.tables.jsonl", lines=True)
    tables.index = tables["id"]
    sequences = []
    for sql, table in zip(queries['sql'], queries['table_id']):
        seq = []
        seq.append("SYMSELECT")
        if sql['agg'] == 0:
            seq.append("SYMAGG")
            seq.append("SYMCOL")
        else:
            seq.append("SYMAGG")
            seq.append(agg_ops[sql['agg']])
            seq.append("SYMCOL")

        seq.append(tables.loc[table]['header'][sql['sel']])
        if len(sql['conds']) > 0:
            seq.append("SYMWHERE")
            seq.append("SYMCOL")
            three_cond_ops = sql['conds'][0]
            seq.append(tables.loc[table]['header'][three_cond_ops[0]])
            seq.append("SYMOP")
            seq.append(cond_ops[three_cond_ops[1]])
            seq.append("SYNCOND")
            seq.append(three_cond_ops[2])
        seq.append("SYMEND")
        sequences.append(seq)
    output = pd.DataFrame(sequences)
    output.to_csv("output_seqs")
    print(sequences[0])


def attention_layer(encoder_outputs, decoder_lstm_outputs):
    # Using built-in Attention layer
    attention = Attention(use_scale=True)
    context_vector, attention_scores = attention([decoder_lstm_outputs, encoder_outputs], return_attention_scores=True)

    return context_vector, attention_scores


class SwitchingNetwork(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SwitchingNetwork, self).__init__(**kwargs)
        self.units = units
        self.dense1 = tf.keras.layers.Dense(units, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        context_vector, decoder_state = inputs
        # Expand the dimensions of decoder_state to match context_vector
        decoder_state_expanded = tf.expand_dims(decoder_state, 1)
        decoder_state_expanded = tf.tile(decoder_state_expanded, [1, tf.shape(context_vector)[1], 1])

        # Now concatenate
        x = tf.keras.layers.concatenate([context_vector, decoder_state_expanded], axis=-1)
        x = self.dense1(x)
        switch = self.dense2(x)

        return switch


def transform_attention_scores(attention_scores, input_to_output_index, max_output_vocab_size, inp_seq, output_tokenizer):
    batch_size, seq_len_output, seq_len_input = tf.shape(attention_scores)[0], tf.shape(attention_scores)[1], \
                                                tf.shape(attention_scores)[2]
    transformed_scores = tf.zeros((batch_size, seq_len_output, max_output_vocab_size))

    for i in tf.range(batch_size):
        for j in tf.range(seq_len_output):
            for k in tf.range(seq_len_input):
                input_idx = inp_seq[i][k]
                output_idx = input_to_output_index.get(input_idx, output_tokenizer.word_index['<OOV>'])
                update_value = attention_scores[i, j, k]
                indices = tf.reshape(tf.convert_to_tensor([i, j, output_idx]), [1, 3])
                update = tf.SparseTensor(indices, [update_value], [batch_size, seq_len_output, max_output_vocab_size])
                transformed_scores += tf.sparse.to_dense(update)

    return transformed_scores





def vectorize():
    inp = pd.read_csv("input_seqs", dtype=str)
    outp = pd.read_csv("output_seqs", dtype=str)

    inp.drop('Unnamed: 0', axis=1, inplace=True)
    inp.fillna('PAD', inplace=True)

    outp.drop('Unnamed: 0', axis=1, inplace=True)
    outp.fillna('PAD', inplace=True)

    input_sequences = [' '.join(row) for row in inp.values]
    output_sequences = [' '.join(row) for row in outp.values]

    x_train, x_valid, y_train, y_valid = train_test_split(input_sequences, output_sequences, test_size=0.2)

    input_tokenizer = Tokenizer(oov_token="<OOV>")
    output_tokenizer = Tokenizer(oov_token="<OOV>")

    input_tokenizer.fit_on_texts(x_train)
    output_tokenizer.fit_on_texts(y_train)

    input_to_output_index = {}
    for word, index in input_tokenizer.word_index.items():
        if word in output_tokenizer.word_index:
            input_to_output_index[index] = output_tokenizer.word_index[word]
        else:
            input_to_output_index[index] = output_tokenizer.word_index['<OOV>']

    x_train_seq = input_tokenizer.texts_to_sequences(x_train)
    x_valid_seq = input_tokenizer.texts_to_sequences(x_valid)
    y_train_seq = output_tokenizer.texts_to_sequences(y_train)
    y_valid_seq = output_tokenizer.texts_to_sequences(y_valid)

    x_train_pad = pad_sequences(x_train_seq, maxlen=142, padding='post')
    x_valid_pad = pad_sequences(x_valid_seq, maxlen=142, padding='post')
    y_train_pad = pad_sequences(y_train_seq, maxlen=13, padding='post')
    y_valid_pad = pad_sequences(y_valid_seq, maxlen=13, padding='post')

    input_vocab_size = len(input_tokenizer.word_index) + 1
    output_vocab_size = len(output_tokenizer.word_index) + 1

    embedding_dim = 128
    lstm_units = 128

    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_state=True, return_sequences=True)  # return_sequences=True added
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=output_vocab_size, output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm_outputs, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(decoder_embedding,
                                                                                            initial_state=encoder_states)
    decoder_context, scores = attention_layer(encoder_outputs, decoder_lstm_outputs)
    print("Attention scores shape:", scores.shape)
    print("Decoder context shape:", decoder_context.shape)
    print("Attention scores shape:", scores.shape)
    print(scores)

    switching_network = SwitchingNetwork(units=lstm_units)
    switch = switching_network([decoder_context, state_h])
    distribution = transform_attention_scores(scores, input_to_output_index, output_vocab_size, x_train_pad, output_tokenizer)

    decoder_outputs = Dense(output_vocab_size, activation='softmax')(decoder_context)
    combined_distribution = switch * decoder_outputs + (1 - switch) * distribution

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit([x_train_pad, y_train_pad], y_train_pad, epochs=10, batch_size=32,
              validation_data=([x_valid_pad, y_valid_pad], y_valid_pad))

    model.save('encoder_decoder.h5')



if __name__ == "__main__":
    #create_input_sequences()
    #create_output_sequences()
    vectorize()

