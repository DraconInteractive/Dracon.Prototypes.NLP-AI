#pip install -U spacy
#spacy download en_core_web_sm
#spacy download en_core_web_trf
#pip install tensorflow

import os
import time
import spacy
from spacy import displacy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Sequential, Model

nlp = spacy.load("en_core_web_lg")
vocab = nlp.vocab
word_to_int = {word.text: i for i, word in enumerate(vocab)}
vocab_array = [word_to_int[word.text] for word in vocab]
"""
# Find the index of a specific word
word = 'example'
word_index = word_to_int[word]
"""

def get_x_vector(doc):
	word_indices = [word_to_int.get(token.text, 0) for token in doc]
	# 20 is the maximum sequence length
	padded_indices = np.pad(word_indices, (0, 20 - len(word_indices)), mode='constant')
	feature_vector = np.concatenate((padded_indices, doc.vector))
	return feature_vector

## TODO Convert 'keywords' into [0,0,0] where each integer is of a word in the input
training_data = [
	(nlp("Whats the weather like in Perth today"), ('information', 'search_weather', 'perth')),
	(nlp("Whats the date today"), ('information', 'search_date', 'today')),
	(nlp("What day is it today"), ('information', 'search_date', 'today')),
	(nlp("Create a JSON document in root"), ('action', 'create_doc', 'json')),
	(nlp("Run a test sequence"), ('action', 'run', 'test')),
	(nlp("Open google for me"), ('action', 'run', 'google'))
]

intents = ['information', 'action']
actions = ['search_weather', 'search_date', 'create_doc', 'run']
keywords = ['perth', 'today', 'json', 'test', 'google']

num_intents = len(intents)
num_actions = len(actions)
num_keywords = len(keywords)

X = []
Y = []
for doc, (intent, action, keyword) in training_data:
	X.append(get_x_vector(doc))

	# One-hot encode the label vectors
	intent_index = intents.index(intent)
	action_index = actions.index(action)
	keyword_index = keywords.index(keyword)

	label = np.concatenate((tf.one_hot(intent_index, num_intents), tf.one_hot(action_index, num_actions), tf.one_hot(keyword_index, num_keywords)))
	Y.append(label)

X = np.array(X)
Y = np.array(Y)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_intents + num_actions + num_keywords, activation='sigmoid', name='output'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the dataset
model.fit(X, Y, epochs=250)

def test_model(phrase, model):
	test_vector = get_x_vector(nlp(phrase))
	print(f"Running test...\"{phrase}\"")
	prediction = model.predict(np.array([test_vector]))
	# Extract the predicted label indices
	intent_pred, action_pred, keyword_pred = np.split(prediction[0], [num_intents, num_intents + num_actions])

	intent_index = np.argmax(intent_pred)
	action_index = np.argmax(action_pred)
	keyword_index = np.argmax(keyword_pred)

	# Look up the corresponding labels in the original lists
	intent_label = intents[intent_index]
	action_label = actions[action_index]
	keyword_label = keywords[keyword_index]

	# Print the predicted labels
	print("\tIntent: ", intent_label)
	print("\tAction: ", action_label)
	print("\tKeyword: ", keyword_label)

test_model("when is the perth derby", model)
test_model("make me a word doc", model)
test_model("open that file I was using", model)


