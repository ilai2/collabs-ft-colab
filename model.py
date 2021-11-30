import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 32 #TODO
        self.batch_size = 250 #TODO 

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.E = tf.Variable(tf.random.normal(shape=[self.vocab_size,self.embedding_size], stddev=.01, dtype=tf.float32))
        self.LSTM = tf.keras.layers.LSTM(2 * self.embedding_size, return_sequences=True, return_state=True)
        self.D1 = tf.keras.layers.Dense(1000, activation="relu")
        self.D2 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")
        self.Dropout = tf.keras.layers.Dropout(0.3)

    def call(self, notes):
        """
        :param ndv: as a tensor (batch_size * window_size) (each element (a note) is a tuple of (pitch, duration, volume))
        :return: as a tensor (batch_size * vocab_size)
        """
        
        #TODO: Fill in 
        notes_embedded = tf.nn.embedding_lookup(self.E, notes)

        lstm1_output, state1, state2 = self.LSTM(notes_embedded)
        lstm1_output = self.Dropout(lstm1_output)

        logits = self.D1(lstm1_output)
        logits = self.Dropout(logits)
        logits = self.D2(logits)
        
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(0, np.shape(train_labels)[0], model.batch_size):
        input = train_inputs[i:model.batch_size+i]
        input = np.array(input)
        label = train_labels[i:model.batch_size+i]
        label = np.array(label)

        with tf.GradientTape() as tape:
            logits, _ = model.call(input)
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

    first_note_index = np.random.randint(0, len(vocab) - 1)
    next_input = [[first_note_index]]
    score = [vocab[first_note_index]]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        score.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(score))

def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    
    # TO-DO:  Separate your train and test data into inputs and labels
    #TODO: Figure out how to actually get the data oop
    pitches, durations, volumes = get_data('../../data/train.txt')
    notes = []
    for i in range(0, len(pitches)):
        notes.append((pitches[i], durations[i], volumes[i]))

    vocab = set(notes)
    vocab = {w: i for i, w in enumerate(list(vocab))}

    notes = tf.convert_to_tensor(notes)
    notes = [vocab[x] for x in notes]

    # TODO: initialize model
    model = Model(len(vocab))

    # turn notes tensor into windows
    train_inputs_indices = notes[:-1]
    train_labels_indices = notes[1:]
    remainder_inputs = np.shape(train_inputs_indices)[0] % model.window_size
    remainder_labels = np.shape(train_labels_indices)[0] % model.window_size
    train_inputs = train_inputs_indices[:-remainder_inputs]
    train_labels = train_labels_indices[:-remainder_labels]
    notes = tf.reshape(train_inputs, [-1, model.window_size])
    labels = tf.reshape(train_labels, [-1, model.window_size])
    # TODO: Set-up the training step
    train(model, notes, labels)
    generate_sentence(300, vocab, model)

if __name__ == '__main__':
    main()
