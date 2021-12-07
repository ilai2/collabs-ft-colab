import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from music21 import *
from preprocess import read_int_dict, read_song, deprocess_midi, read_element_dict


class Model(tf.keras.Model):
    def __init__(self, vocab_size, num_instruments=1):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.window_size = 100 # DO NOT CHANGE!
        self.embedding_size = 60 #TODO
        self.batch_size = 25 #TODO 
        self.num_instruments = num_instruments

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.E = tf.Variable(tf.random.normal(shape=[self.vocab_size,self.embedding_size], stddev=.01, dtype=tf.float32))
        self.LSTM = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
        self.LSTM2 = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.D1 = tf.keras.layers.Dense(512, activation="relu")
        self.D2 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")
        self.Dropout = tf.keras.layers.Dropout(0.3)


    def call(self, notes, input_state, is_generating=False):
        """
        :param ndv: as a tensor (num_instruments * batch_size * window_size) (each element (a note) is a tuple of (pitch, duration, volume))
        :return: as a tensor (num_instruments * batch_size * vocab_size)
        """
        
        #TODO: Fill in 
        if not is_generating:
            notes = tf.cast(tf.reshape(notes, [self.num_instruments * self.batch_size, self.window_size]), tf.int32)
        notes = tf.cast(notes, tf.int32)
        notes_embedded = tf.nn.embedding_lookup(self.E, notes)

        lstm1_output, _, _ = self.LSTM(notes_embedded)
        lstm1_output = self.Dropout(lstm1_output)
        lstm2_output, state1, state2 = self.LSTM2(lstm1_output)

        logits = self.D1(lstm2_output)
        logits = self.Dropout(logits)
        logits = self.D2(logits)

        if not is_generating:
            logits = tf.reshape(logits, [self.num_instruments, self.batch_size, self.window_size, self.vocab_size])
        
        return logits, (state1, state2)

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for i in range(0, np.shape(train_labels)[1], model.batch_size):
        if (i + model.batch_size <= np.shape(train_labels)[1]):
            input = train_inputs[:,i:model.batch_size+i]
            input = np.array(input)
            label = train_labels[:,i:model.batch_size+i]
            label = np.array(label)

            with tf.GradientTape() as tape:
                logits, _ = model.call(input, None)
                loss = model.loss(logits, label)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_sentence(length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    next_input = np.empty((model.num_instruments, 1))
    score_val = []

    for i in range(model.num_instruments):
        first_note_index = np.random.randint(0, len(vocab) - 1)
        next_input[i] = [first_note_index]
        note = reverse_vocab[first_note_index]
        note = (i, note[1], note[2], note[3])
        score_val.append(note)

    score = [score_val]

    for _ in range(length):
        logits, previous_state = model.call(next_input, previous_state, is_generating=True)
        logits = np.array(logits[:,0,:])

        one_note = []
        next_input = np.empty((model.num_instruments, 1))

        for i in range(len(logits)):
        #     for j in range(len(logits[0])):
        #         new_logits.append(logits[i][j])
        # logits = np.array(new_logits)
        # print(np.shape(logits))
            top_n = np.argsort(logits[i])[-sample_n:]
            n_logits = np.exp(logits[i][top_n])/np.exp(logits[i][top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)
            note = reverse_vocab[out_index]
            note = (i, note[1], note[2], note[3])
            one_note.append(note)
            next_input[i] = [out_index]

        score.append(one_note)

    return score

def main():
    # initialize lists for three attributes
    pitches = []
    durations = []
    volumes = []

    # load in 500 songs
    for a in range(50):
        pitches_i, durations_i, volumes_i = read_song('classical.txt', a)
        pitches_i = pitches_i[13]
        durations_i = durations_i[13]
        volumes_i = volumes_i[13]
        pitches.append(pitches_i)
        durations.append(durations_i)
        volumes.append(volumes_i)

    # reformat songs into 1-d list
    flattened_pitches = []
    flattened_durations = []
    flattened_volumes = []
    for ii in range(len(pitches)):
        for jj in range(len(pitches[ii])):
            # for kk in range(len(pitches[ii][jj])):
            flattened_pitches.append(pitches[ii][jj])
            flattened_durations.append(durations[ii][jj])
            flattened_volumes.append(volumes[ii][jj])

    
    pitches = np.reshape(np.array(flattened_pitches), (1, -1))
    durations = np.reshape(np.array(flattened_durations), (1, -1))
    volumes = np.reshape(np.array(flattened_volumes), (1, -1))


    notes = []
    for i in range(0, len(pitches)):
        for j in range(len(pitches[0])):
            notes.append((i, pitches[i][j], durations[i][j], volumes[i][j]))
        
    vocab = set(list(notes))
    vocab = {w: i for i, w in enumerate(list(vocab))}

    # notes = [vocab[x] for x in notes]
    new_notes = np.empty((len(pitches), len(pitches[0])))
    for k in range(len(notes)):
        note = notes[k]
        instrument = note[0]
        note_pdv = (note[0], note[1], note[2], note[3])
        new_notes[instrument][k % len(notes[0])] = vocab[note_pdv]

    notes = tf.convert_to_tensor(new_notes)

    # TODO: initialize model
    model = Model(len(vocab))

    # turn notes tensor into windows
    train_inputs_indices = notes[:,:-1]
    train_labels_indices = notes[:,1:]
    remainder_inputs = np.shape(train_inputs_indices)[1] % model.window_size
    remainder_labels = np.shape(train_labels_indices)[1] % model.window_size
    train_inputs = train_inputs_indices[:,:-remainder_inputs]
    train_labels = train_labels_indices[:,:-remainder_labels]
    notes = tf.reshape(train_inputs, [len(train_inputs), -1, model.window_size])
    labels = tf.reshape(train_labels, [len(train_inputs), -1, model.window_size])
    # TODO: Set-up the training step
    for b in range(15):
        print ("Epoch Number: ", b)
        train(model, notes, labels)

    raw_score = generate_sentence(300, vocab, model)
    score_pitches = np.empty((model.num_instruments, 301))
    score_durations = np.empty((model.num_instruments, 301))
    score_volumes = np.empty((model.num_instruments, 301))
    for k in range(len(raw_score)):
        for l in raw_score[k]:
            score_pitches[l[0]][k] = l[1]
            score_durations[l[0]][k] = l[2]
            score_volumes[l[0]][k] = l[3]

    _, idict = read_int_dict("dict.txt")
    midi_score = deprocess_midi(score_pitches, score_durations, score_volumes, idict)
    midi_out = midi_score.write('midi', fp='test_good_music_mac.mid')
    model.save_weights("weights")

if __name__ == '__main__':
    main()
