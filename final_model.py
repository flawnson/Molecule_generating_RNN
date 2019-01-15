"""
OK LISTEN UP

This is the LSTM model that will receive one-hot encoded "grammarfied" molecules (No, grammarfied is not a word, but for the sake of clarity)
grammarfied: the antithesis of characterized, essentially, instead of being broken down into individual characters, it is broken down to grammar rules
Anyways, this model will receive ^^^ as input, and output a set molecules.
These outputed molecules, depending on the accuracy of the model (which is dependant on the training), may or may not chemically valid.
It is important to note that it is entirely possible to create a model that received characterized inputs.
It has been empirically and experimentally proved that the grammarfied model outputs notably more valid molecules.
Find the research paper "Grammar Variational Autoencoders" by Hernández-Lobato et al. for more details.
As with all supervized neural network models, there are 2 main datatypes.
The aforementioned one-hot encoded grammarfied inputs, and a dataset of SMILE strings.
Each training step will take a specified number of SMILE strings to determine the next one-hot encoded grammarfied production rule.
Comment out parts of the code depending on what you want to use the code for (Training, Saving checkpoints, Sampling, etc.)
PIVOT

Yeah so the above idea isn't working out
New plan: we're gonna do the characterized version
"""
import sys
import numpy as np
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

"""DATA PREPROCESSING"""

# Opening files, extracting data, and automatically closing them (SMILES strings are conjoined together with the "\n" metatag)
filename = "100k_rndm_zinc_drugs_clean.txt"
with open(filename) as f:
	# f = [next(filename) for x in range(10000)]
    	raw_text = "\n".join(line.strip() for line in f)

# Creating mapping from character to integer and vice versa (a mapping for "\n" metatag is manually inserted into the dictionaries)
unique_chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(unique_chars))
char_to_int.update({-1 : "\n"})
int_to_char = dict((i, c) for i, c in enumerate(unique_chars))
int_to_char.update({"\n" : -1})

mapping_size = len(char_to_int)
reverse_mapping_size = len(int_to_char)

print ("Size of the character to integer dictionary is: ", mapping_size)
print ("Size of the integer to character dictionary is: ", reverse_mapping_size)

assert mapping_size == reverse_mapping_size

# Summarize the loaded data to provide lengths for preparing datasets
n_chars = len(raw_text)
n_vocab = len(unique_chars)

print ("Total number of characters in the file is: ", n_chars)

# Preparring datasets by matching the dataset lengths (dataX will be the SMILES strings and dataY will be individual characters in the SMILE string)
seq_length = 137
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)

# Reshape X to be [samples, time steps, features], the expected input format for recurrent models
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# Normalize the integers in X by dividing by the number of unique SMILES characters (a.k.a vocabulary)
X = X / float(n_vocab)

# One-hot encode the output variable (so that they can be used to generate new SMILES after training)
Y = np_utils.to_categorical(dataY)

"""CREATING THE LSTM MODEL"""

# Create the model (simple 2 layer LSTM)
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))

print(model.summary())

# Compile the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam')

# # Define checkpoints (used to save the weights at each epoch, so that the model doesn't need to be retrained)
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
# callbacks_list = [checkpoint]

# # Fit the model
# model.fit(X, Y, epochs = 19, batch_size = 512, callbacks = callbacks_list)

# """TO TRAIN FROM SAVED CHECKPOINT"""
# # Load weights
# model.load_weights("weights-improvement-75-1.8144.hdf5")

# # load the model
# new_model = load_model("model.h5")
# assert_allclose(model.predict(x_train),
#                 new_model.predict(x_train),
#                 1e-5)

# # fit the model
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# new_model.fit(x_train, y_train, epochs = 100, batch_size = 64, callbacks = callbacks_list)

"""GENERATING NEW SMILES"""

# Load the pre-trained network weights
filename = "weights-improvement-24-0.7524.hdf5"
model.load_weights(filename)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Pick a random seed from the SMILES strings
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# Generate specified number of characters in range
for i in range(137):
	x = np.reshape(pattern, (1, len(pattern), 1))
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")

"""
The model is complete.

Further improvements include:
1. Creating two unique tokens that indicate the beginning and end of a SMILE string (and therefore, the beginning and end of a molecule)
2. Using a more powerful Neural Memoery Network (Stack-Augmented RNN, Neural Turing Machines, Differentiable Neural Computers, etc)
3. Creating a GPU friendly version of the code (although I don't have a GPU)
4. Could have used a different input format, other than integer representations, there are one-hot vectors (probably just as good) and character embeddings (probably much better, reducing complexity and increasing generalization)
5. As indicated in the preamble before the code, the outputs of this model could have had more valid molecules, had I used the grammar model.
"""