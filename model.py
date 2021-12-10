import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from music21 import *
from preprocess import read_int_dict, read_song, deprocess_midi
import sys


class Model(tf.keras.Model):
    def __init__(self, vocab_size, num_instruments=1):
        """
        The Model class predicts the next notes in a score

        :param vocab_size: the size of the vocab
        :param num_instruments: the number of instruments to train on (default is 1)
        """

        super(Model, self).__init__()

        # hyperparameters
        self.vocab_size = vocab_size
        self.window_size = 100 
        self.embedding_size = 75 
        self.batch_size = 25 
        self.num_instruments = num_instruments

        # model architecture
        self.E = tf.Variable(tf.random.normal(shape=[self.vocab_size,self.embedding_size], stddev=.01, dtype=tf.float32))
        self.LSTM = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
        self.LSTM2 = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.D1 = tf.keras.layers.Dense(1024, activation="relu")
        self.D2 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")
        self.Dropout = tf.keras.layers.Dropout(0.3)


    def call(self, notes, is_generating=False):
        """
        :param notes: a (num_instruments * batch_size * window_size) tensor of notes
                      where each note is represented as a (pitch, duration, volume)
        : param is_generating: boolean indicating whether the model is generating music
        :return: a (num_instruments * batch_size * window_size * vocab_size) tensor of probabilities
        """        
        # flatten input across instruments axis if training
        if not is_generating:
            notes = tf.cast(tf.reshape(notes, [self.num_instruments * self.batch_size, self.window_size, 1]), tf.int32)
        notes = tf.cast(notes, tf.int32)

        # embed notes and run through RNN
        # notes_embedded = tf.nn.embedding_lookup(self.E, notes)
        # lstm1_output, _, _ = self.LSTM(notes_embedded)
        print(np.shape(notes))
        lstm1_output, _, _ = self.LSTM(notes)
        lstm1_output = self.Dropout(lstm1_output)
        lstm2_output, _, _ = self.LSTM2(lstm1_output)

        # predict probabiliies using Dense layers
        logits = self.D1(lstm2_output)
        logits = self.Dropout(logits)
        logits = self.D2(logits)

        # reshape if training
        if not is_generating:
            logits = tf.reshape(logits, [self.num_instruments, self.batch_size, self.window_size, self.vocab_size])
        
        return logits

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param logits: a tensor of shape (num_instruments * batch_size * window_size * vocab_size)
                       containing probabilities
        :param labels: a tensor of shape (num_instruments * batch_size * window_size) 
                       containing labels
        :return: the loss of the model as a tensor of size 1
        """

        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_instruments * num_inputs)
    :param train_labels: train labels (all labels for training) of shape (num_instruments * num_labels)
    :return: None
    """
    # define an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # calculate loss
    total_loss = 0

    # complete one epoch, taking batch_size shape chunks of the training data
    for i in range(0, np.shape(train_labels)[1], model.batch_size):
        if (i + model.batch_size <= np.shape(train_labels)[1]):
            input = train_inputs[:,i:model.batch_size+i]
            input = np.array(input)
            label = train_labels[:,i:model.batch_size+i]
            label = np.array(label)
            with tf.GradientTape() as tape:
                logits = model.call(input)
                loss = model.loss(logits, label)
            total_loss += loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # print perplexity
    print(np.exp(total_loss / (np.shape(train_labels)[1] / model.batch_size)))

def generate_score(length, vocab, model, sample_n=5):
    """
    Takes a model, vocab, selects the most likely next note

    :param length: number of notes in score
    :param vocab: dictionary, note to id mapping
    :param model: trained RNN model
    :sample_n: number of closest predictions to choose note from
    :return: None
    """

    reverse_vocab = {idx: word for word, idx in vocab.items()}

    next_input = np.empty((model.num_instruments, 1))
    score_val = []

    # generate start for each instrument
    for i in range(model.num_instruments):
        first_note_index = np.random.randint(0, len(vocab) - 1)
        next_input[i] = [first_note_index]
        note = reverse_vocab[first_note_index]
        note = (i, note[1], note[2], note[3])
        score_val.append(note)

    score = [score_val]

    # call model on is_generating to get most likely next note for each instrument
    for _ in range(length):
        logits = model.call(next_input, is_generating=True)
        logits = np.array(logits[:,0,:])

        one_note = []
        next_input = np.empty((model.num_instruments, 1))

        for i in range(len(logits)):
            top_n = np.argsort(logits[i])[-sample_n:]
            n_logits = np.exp(logits[i][top_n])/np.exp(logits[i][top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)
            note = reverse_vocab[out_index]
            note = (i, note[1], note[2], note[3])
            one_note.append(note)
            next_input[i] = [out_index]

        # add new note to score
        score.append(one_note)

    return score

def main():
    if sys.argv[1] not in {"--train", "--load"}:
        print("USAGE: python assignment.py <task>")
        print("<task>: [train/load]")
        exit()

    # initialize lists for three attributes
    pitches = []
    durations = []
    volumes = []

    # load in songs 
    for a in range(100):
        pitches_f, durations_f, volumes_f = read_song('classical.txt', a)
        pitches_f = pitches_f[13]
        durations_f = durations_f[13]
        volumes_f = volumes_f[13]
        pitches.append(pitches_f)
        durations.append(durations_f)
        volumes.append(volumes_f)
        # pitches_s, durations_s, volumes_s = read_song('jazz.txt', a)
        # pitches_s = pitches_s[13]
        # durations_s = durations_s[13]
        # volumes_s = volumes_s[13]
        # pitches.append(pitches_s)
        # durations.append(durations_s)
        # volumes.append(volumes_s)

    # reformat songs into 1-d list
    flattened_pitches = []
    flattened_durations = []
    flattened_volumes = []
    for ii in range(len(pitches)):
        for jj in range(len(pitches[ii])):
            flattened_pitches.append(pitches[ii][jj])
            flattened_durations.append(durations[ii][jj])
            flattened_volumes.append(volumes[ii][jj])

    
    pitches = np.reshape(np.array(flattened_pitches), (1, -1))
    durations = np.reshape(np.array(flattened_durations), (1, -1))
    volumes = np.reshape(np.array(flattened_volumes), (1, -1))

    # combine features into tuple
    notes = []
    for i in range(0, len(pitches)):
        for j in range(len(pitches[0])):
            notes.append((i, pitches[i][j], durations[i][j], volumes[i][j]))
    
    # create vocab
    vocab = set(list(notes))
    vocab = {w: i for i, w in enumerate(list(vocab))}

   # create notes matrix, reshaping "flattened" matrix for each instrument
    new_notes = np.empty((len(pitches), len(pitches[0])))
    for k in range(len(notes)):
        note = notes[k]
        instrument = note[0]
        note_pdv = (note[0], note[1], note[2], note[3])
        new_notes[instrument][k % len(notes[0])] = vocab[note_pdv]

    notes = tf.convert_to_tensor(new_notes)

    # initialize model
    model = Model(len(vocab))

    # initialize number of epochs depending on if training or using pretrained weights
    epoch_num = 0

    if sys.argv[1] == "--load":
        epoch_num = 1
    elif sys.argv[1] == "--train":
        epoch_num = 0

    # turn notes tensor into windows
    train_inputs_indices = notes[:,:-1]
    train_labels_indices = notes[:,1:]
    remainder_inputs = np.shape(train_inputs_indices)[1] % model.window_size
    remainder_labels = np.shape(train_labels_indices)[1] % model.window_size
    train_inputs = train_inputs_indices[:,:-remainder_inputs]
    train_labels = train_labels_indices[:,:-remainder_labels]
    notes = tf.reshape(train_inputs, [len(train_inputs), -1, model.window_size])
    labels = tf.reshape(train_labels, [len(train_inputs), -1, model.window_size])

    # save weights 
    for b in range(epoch_num):
        if (b+1) % 10 == 0 and b >= 9 and epoch_num != 1:
            model.save_weights(str(b) + '.h5')

        print ("Epoch Number: ", b)
        if sys.argv[1] == "--load":
            train(model, notes[:,0:model.batch_size], labels[:,0:model.batch_size])
        else:    
            train(model, notes, labels)

    if sys.argv[1] == "--load":
        model.built = True
        model.load_weights(sys.argv[2])
        print("Weights loaded.")

    raw_score = generate_score(300, vocab, model)

    # reconvert score to format for deprocessing
    score_pitches = np.empty((model.num_instruments, 301))
    score_durations = np.empty((model.num_instruments, 301))
    score_volumes = np.empty((model.num_instruments, 301))
    for k in range(len(raw_score)):
        for l in raw_score[k]:
            score_pitches[l[0]][k] = l[1]
            score_durations[l[0]][k] = l[2]
            score_volumes[l[0]][k] = l[3]

    # write score to midi
    _, idict = read_int_dict("dict.txt")
    midi_score = deprocess_midi(score_pitches, score_durations, score_volumes, idict)
    midi_score.write('midi', fp='test_good_music_classical.mid')

if __name__ == '__main__':
    main()
