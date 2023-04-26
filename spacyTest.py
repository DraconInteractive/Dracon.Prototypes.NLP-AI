#pip install -U spacy
#spacy download en_core_web_sm
#spacy download en_core_web_trf

import os
import spacy
from spacy import displacy
import validatePath as vp

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

def main():
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