import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np;
import matplotlib.pyplot as plt

class TF:
	def run(self):
		# Need a dataset
		# Will come from Spacy in form of NLP elements

		pass

	def run__basic_example(self):
		print("TensorFlow version: ", tf.__version__)

		# Notes
		# Logits: Vector of raw (non-normalized) predictions that a classification model generates, normally then passed to a normalization function

		# Load dataset
		# mnist is 60k hand-drawn digits
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		# Images are 0->255 in value (black and white)
		x_train, x_test = x_train / 255.0, x_test / 255.0

		# Build model
		# Input shape (28,28) is a matrix vector 28x28, likely the resolution of the image
		# Dense (128) define a layer of 128 nodes
		# Dropout (0.2) defines the fraction of input units to drop (my guess = 20%)
		model = tf.keras.models.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(10)
		])

		# Define loss function
		loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

		# Compile and configure model
		# Adam algorithm is "omputationally efficient, has little memory requirement, invariant to diagonal rescaling of gradients, and is well suited for problems that are large in terms of data/parameters"
		model.compile(optimizer='adam',
			  loss=loss_fn,
			  metrics=['accuracy'])

		# Train and evaluate
		model.fit(x_train, y_train, epochs=2)
		model.evaluate(x_test,  y_test, verbose=2)

		# Get probability
		probability_model = tf.keras.Sequential([
		  model,
		  tf.keras.layers.Softmax()
		])

		pMod = probability_model(x_test[:2])
		print (f"Probability: {pMod}")
		#print (f"Probability Zero: "{pMod[0]})	

	def run_text_rnn_example(self):
		# Data location
		path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
		# Open data file
		text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
		# Unique characters in file
		vocab = sorted(set(text))

		ids_from_chars = tf.keras.layers.StringLookup(
			vocabulary=list(vocab), mask_token=None)

		chars_from_ids = tf.keras.layers.StringLookup(
			vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

		# "Convert text vector into stream of character indicies"
		all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
		ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


		# Function to join characters back into strings (used *much later)
		def text_from_ids(ids):
			return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

		# Batch individual characters to sequences of the desired size
		seq_length = 100
		sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
		# Training requires a dataset of (input, label) pairs where both are sequences
		# Function takes a sequence as input, duplicates and shifts it to align the input and label for each typeset
		# "Tensorflow" becomes "Tensorflo" and "ensorflow"
		def split_input_target(sequence):
			input_text = sequence[:-1]
			target_text = sequence[1:]
			return input_text, target_text
		dataset = sequences.map(split_input_target)

		class ExampleRNNModel(tf.keras.Model):
			# Model has 3 layers
			# Embedding: the input layer. Trainable lookup table that will map each character-ID to a vector with embedding_dim dimensions
			# GRU: A type of RNN with size units=rnn_units (could possibly also use an LSTM layer here)
			# Dense: The output layer with vocab_size outputs. One logit for each character in vocabulary. These are the log-likelihood of each character according to the model
			def __init__(self, vocab_size, embedding_dim, rnn_units):
				super().__init__(self)
				self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
				self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
				self.dense = tf.keras.layers.Dense(vocab_size)

			def call(self, inputs, states=None, return_state=False, training=False):
				x = inputs
				x = self.embedding(x, training=training)
				if states is None:
					states = self.gru.get_initial_state(x)
				x, states = self.gru(x, initial_state=states, training=training)
				x = self.dense(x, training=training)

				if return_state:
					return x, states
				else:
					return x

		class CustomTraining(ExampleRNNModel):
			@tf.function
			def train_step(self, inputs):
				inputs, labels = inputs
				with tf.GradientTape() as tape:
					predictions = self(inputs, training=True)
					loss = self.loss(labels, predictions)
				grads = tape.gradient(loss, self.trainable_variables)
				self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

				return {'loss': loss}	

		# Before feeding data into model, shuffle and pack into batches
		BATCH_SIZE = 64
		# TF data is design to work with possibly infinite sequences, so it doesnt attempt to shuffle entire sequence in memory. 
		# Instead, it maintains a buffer in which it shuffles elements
		BUFFER_SIZE = 10000
		dataset = (
			dataset
			.shuffle(BUFFER_SIZE)
			.batch(BATCH_SIZE, drop_remainder=True)
			.prefetch(tf.data.experimental.AUTOTUNE))

		# Length of the vocabulary in StringLookup Layer
		vocab_size = len(ids_from_chars.get_vocabulary())
		# The embedding dimension
		embedding_dim = 256
		# Number of RNN units
		rnn_units = 1024

		# Use TF.ExampleRNNModel for original setup
		# Custom training is for gradient training set
		model = TF.CustomTraining(
			vocab_size=vocab_size,
			embedding_dim=embedding_dim,
			rnn_units=rnn_units)

		# Testing phase
		# Get shape of output
		for input_example_batch, target_example_batch in dataset.take(1):
			example_batch_predictions = model(input_example_batch)
			print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

		# print summary
		model.summary()

		# Sample the output distribution to get actual character indicies
		# Distribution is defined by the logits over the character vocabulary
		sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
		sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

		# Decode these to see the text predicted by the untrained model
		print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
		print()
		print(f"Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
		print()

		# Loss function (model returns logits, so set from_logits flag)
		loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
		example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
		print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
		print("Mean loss:        ", example_batch_mean_loss.numpy())

		# New model shouldnt be too 'sure' of itself. Mean loss should be ~= to vocab size
		# High loss means model is sure of the wrong answers and is badly initialized (example 66.14)
		print ("Exp Mean loss: ", tf.exp(example_batch_mean_loss).numpy())
		print()

		# Configure / compile with adam algorithm and defined loss function
		model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss)

		# Configure checkpoints
		# Directory where the checkpoints will be saved
		checkpoint_dir = './training_checkpoints'
		# Name of the checkpoint files
		checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_prefix,
			save_weights_only=True)

		# Keep to 10-20 for reasonable training time (seeing approx 5-6 min per epoch)
		EPOCHS = 2

		# Train
		history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
		
		"""
		# Alternate, manual form of training
		mean = tf.metrics.Mean()

		for epoch in range(EPOCHS):
			start = time.time()

			mean.reset_states()
			for (batch_n, (inp, target)) in enumerate(dataset):
				logs = model.train_step([inp, target])
				mean.update_state(logs['loss'])

				if batch_n % 50 == 0:
					template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
					print(template)
			# saving (checkpoint) the model every 5 epochs
			if (epoch + 1) % 5 == 0:
				model.save_weights(checkpoint_prefix.format(epoch=epoch))
			print()
			print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
			print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
			print("_"*80)

		model.save_weights(checkpoint_prefix.format(epoch=epoch))
		"""

		# Define submodel for single step text generation
		class OneStep(tf.keras.Model):
			def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
				super().__init__()
				self.temperature = temperature
				self.model = model
				self.chars_from_ids = chars_from_ids
				self.ids_from_chars = ids_from_chars

				# Create a mask to prevent "[UNK]" from being generated.
				skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
				sparse_mask = tf.SparseTensor(
					# Put a -inf at each bad index.
					values=[-float('inf')]*len(skip_ids),
					indices=skip_ids,
					# Match the shape to the vocabulary
					dense_shape=[len(ids_from_chars.get_vocabulary())])
				self.prediction_mask = tf.sparse.to_dense(sparse_mask)

			@tf.function
			def generate_one_step(self, inputs, states=None):
				# Convert strings to token IDs.
				input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
				input_ids = self.ids_from_chars(input_chars).to_tensor()

				# Run the model.
				# predicted_logits.shape is [batch, char, next_char_logits]
				predicted_logits, states = self.model(inputs=input_ids, states=states,
													  return_state=True)
				# Only use the last prediction.
				predicted_logits = predicted_logits[:, -1, :]
				predicted_logits = predicted_logits/self.temperature
				# Apply the prediction mask: prevent "[UNK]" from being generated.
				predicted_logits = predicted_logits + self.prediction_mask

				# Sample the output logits to generate token IDs.
				predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
				predicted_ids = tf.squeeze(predicted_ids, axis=-1)

				# Convert from token ids to characters
				predicted_chars = self.chars_from_ids(predicted_ids)

				# Return the characters and model state.
				return predicted_chars, states

		one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

		# Loop to generate text
		start = time.time()
		states = None
		next_char = tf.constant(['ROMEO:'])
		result = [next_char]

		for n in range(1000):
		  next_char, states = one_step_model.generate_one_step(next_char, states=states)
		  result.append(next_char)

		result = tf.strings.join(result)
		end = time.time()
		print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
		print('\nRun time:', end - start)