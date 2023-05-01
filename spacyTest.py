#pip install -U spacy
#spacy download en_core_web_sm
#spacy download en_core_web_trf
#pip install tensorflow

import os
import spacy
from spacy import displacy
import validatePath as vp
import tensorflow as tf

class Verb:
	def __init__(self, token):
		self.lemma = token.lemma_
		self.objects = self.parseObjects(token)

	def parseObjects(self, token):
		objects = []
		for child in token.children:
			if (child.dep_ == "dobj"):
				objects.append(DirectObject(child))		
			elif(child.dep_ == "prep"):
				objects.append(PrepObject(child))			
		return objects;

	def debug(self):
		print(" Verb: ", self.lemma)
		for obj in self.objects:
			obj.debug()

class VerbObject:
	def __init__(self, token):
		self.lemma = ""
		self.compoundLemma = ""
		self.label = ""

	def debug(self):
		if (self.compoundLemma != None):
			print(f"  {type(self).__name__}:\t[{self.compoundLemma}] {self.lemma}")
		else:
			print(f"  {type(self).__name__}:\t{self.lemma}")

class DirectObject(VerbObject):
	def __init__(self, token):
		VerbObject.__init__(self, token)
		self.lemma = token.lemma_
		compound = next((x for x in token.children if x.dep_ == "compound"), None)
		if compound != None:
			self.compoundLemma = compound.lemma_
		else:
			self.compoundLemma = ""
		self.label = "dobj"

class PrepObject(VerbObject):
	def __init__(self, token):
		VerbObject.__init__(self, token)
		self.prepLemma = token.lemma_
		pobj = next(x for x in token.children if x.dep_ == "pobj")

		self.lemma = pobj.lemma_
		compound = next((x for x in pobj.children if x.dep_ == "compound"), None)
		if compound != None:
			self.compoundLemma = compound.lemma_
		else:
			self.compoundLemma = ""
		self.label = "pobj"
		# In long sentences, there can be multiple prepositions
		for child in pobj.children:
			if child.dep_ == "prep":
				self.childPrep = PrepObject(child)
				break

class ActionProcessor:
	def __init__(self, actions, defaultAction):
		self.actions = actions
		self.defaultAction = defaultAction

	def processNLP(self, data, verbs):
		# Use first verb as root action for now
		# Fails if no verbs present. TODO: Add handling for that
		if (len(verbs) > 0):
			baseAction = next((x for x in self.actions if verbs[0].lemma in x.keys), None)
		else:
			baseAction = None
		
		if baseAction != None:
			baseAction.run(data, verbs)
		else:
			self.defaultAction.run(data, verbs)
			
class Action:
	def __init__(self, keys):
		self.keys = keys

	def run(self, data, verbs):
		print(f"[Action] Running {type(self).__name__}\n")

class CreateAction(Action):
	def run(self, data, verbs):
		Action.run(self, data, verbs)
		objects = verbs[0].objects
		dobj = next((x for x in objects if x.label == "dobj"), None);
		pobj = next((x for x in objects if x.label == "pobj"), None);
		if (dobj.lemma == "file"):
			path = os.path.dirname(__file__)
			print (path)
			file = ""
			fileName = "testfile"
			for verb in verbs:
				if verb.lemma == "add":
					primaryObject = verb.objects[0];
					if (primaryObject.lemma == "content"):
						if (primaryObject.compoundLemma == "json"):
							print("[Create][Add] Adding json content to file.")
							file = "{\"content\":\"test\"}"
						else:
							print("[Create][Add] Adding basic content to file.")
							file = "This is testing content"
					else:
						print("[Create][Add] Unknown add type")
						return
			print(f"[Create] Content: \n{file}")
			extension = ".txt"
			fullPath = f"{path}\\{fileName}{extension}"
			print(f"[Create] Create {fullPath}? (y/n)")
			confirmedCreate = input()
			if confirmedCreate == "y":
				
				f = open(fullPath, "w")
				f.write(file)
				f.close()
				print("[Create] File created")
		else:
			print("[Create] Unknown create type")

class OutputAction(Action):
	def run(self, data, verbs):
		Action.run(self, data, verbs)
		verbCount = 0
		nounCount = 0
		for subj in data:
			if subj.pos_ == "VERB":
				verbCount += 1
			elif subj.pos_ == "NOUN":
				nounCount += 1
		print(f"Verbs: {verbCount}, Nouns: {nounCount}")

class TF:
	def run(self):
		# Need a dataset
		# Will come from Spacy in form of NLP elements
		
		pass

	def run_example(self):
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


def main():
	TF().run_example()
	return
	nlp = spacy.load("en_core_web_sm")
	#nlp = spacy.load("en_core_web_trf")
	while True:
		text = input('> ')
		doc = nlp(text)

		verbs = []

		print("Looking for verbs...")

		for subj in doc:
			if (subj.pos_ == "VERB"):
				verbs.append(Verb(subj))

		print (f"\nFound {len(verbs)} verbs\n")
		print("Debugging...\n")

		for verb in verbs:
			verb.debug()

		print("Setting up actions...\n")
		actions = []
		actions.append(CreateAction(["create"]))
		outputAction = OutputAction([])
		processor = ActionProcessor(actions, outputAction)

		print("Running processor...\n")
		processor.processNLP(doc, verbs)

		#serve = input('Serve? (y/n): ')
		#if (serve == "y"):
			#displacy.serve(doc)

main()