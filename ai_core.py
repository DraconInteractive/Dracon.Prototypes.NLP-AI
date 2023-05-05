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

nlp = spacy.load("en_core_web_sm")

training_data = [
    (nlp("Whats the weather like in Boston today"), ('information', 'search_weather', 'boston')),
    (nlp("Whats the date today"), ('information', 'search_date', 'today')),
    (nlp("What day is it today"), ('information', 'search_date', 'today')),
    (nlp("Create a JSON document in root"), ('action', 'create_doc', 'json')),
    (nlp("Run a test sequence"), ('action', 'run', 'test')),
    (nlp("Open google for me"), ('action', 'run', 'google'))
]

intents = ['information', 'action']
actions = ['search_weather', 'search_date', 'create_doc', 'run']
keywords = ['boston', 'today', 'json', 'test', 'google']

num_samples = len(training_data)
num_intents = len(intents)
num_actions = len(actions)
num_keywords = len(keywords)

X = []
y_intent = []
y_action = []
y_keyword = []
for doc, (intent, action, keyword) in training_data:
	# Convert doc to a feature vector
	feature_vector = doc.vector
	X.append(feature_vector)

	# One-hot encode the label vectors
	intent_vec = np.zeros(num_intents)
	intent_index = intents.index(intent)
	intent_vec[intent_index] = 1
	y_intent.append(intent_vec)

	action_vec = np.zeros(num_actions)
	action_index = actions.index(action)
	action_vec[action_index] = 1
	y_action.append(action_vec)

	keyword_vec = np.zeros(num_keywords)
	keyword_index = keywords.index(keyword)
	keyword_vec[keyword_index] = 1
	y_keyword.append(keyword_vec)


# Convert the target variables to 2D numpy arrays
X = np.array(X);
input_shape = (6, 1, 96)
inputs = Input(shape=input_shape)
lstm = LSTM(64)(inputs)


intent_output = Dense(num_intents, activation='softmax', name='intent')(lstm)
action_output = Dense(num_actions, activation='softmax', name='action')(lstm)
keyword_output = Dense(num_keywords, activation='softmax', name='keyword')(lstm)

# Define the model
model = Model(inputs=inputs, outputs={'intent': intent_output, 'action': action_output, 'keyword': keyword_output})

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

intent_arr = np.array(y_intent).reshape(num_samples, len(intents))
action_arr = np.array(y_action).reshape(num_samples, len(actions))
keyword_arr = np.array(y_keyword).reshape(num_samples, len(keywords))

Y = {'intent': intent_arr, 'action': action_arr, 'keyword': keyword_arr}
#Y = [y_intent, y_action, y_keyword]

# Train the model on the dataset
model.fit(X, Y, epochs=10)

test_phrase = "cook me a wonderful dinner"
test_vector = nlp(test_phrase).vector
prediction = model.predict(np.array([test_vector]))

intent_label = np.argmax(prediction['intent'])
action_label = np.argmax(prediction['action'])
keyword_label = np.argmax(prediction['keyword'])

print("Intent: ", intent_label)
print("Action: ", action_label)
print("Keyword: ", keyword_label)