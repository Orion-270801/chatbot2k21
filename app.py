from flask import Flask, render_template, request, jsonify
import nltk
nltk.download('punkt')
import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import csv
stemmer = LancasterStemmer()
seat_count = 50
#import request
with open("intents.json") as file:
	data = json.load(file,strict = False)
with open("data.pickle","rb") as f:
	words, labels, training, output = pickle.load(f)

#Function to process input
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)
tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

#Loading existing model from disk
model = tflearn.DNN(net)
model.load("model.tflearn")
from flask_restful import reqparse, abort, Api, Resource

#import joblib
#model = joblib.load('model.sav')
app = Flask(__name__)
api = Api(app)
import re
regex = '^[0-9]+$.'
message = []
def get_bot_response(message):
	message = message.lower()
	results = model.predict([bag_of_words(message,words)])[0]
	result_index = np.argmax(results)
	tag = labels[result_index]
	for tg in data['intents']:
		if tg['tag'] == tag:
			responses = tg['response']
	response = random.choice(responses)
	return str(response)
x = []
class mj(Resource):
	@app.route('/',methods=['POST'])
	#def get(self,message):
	def get():
		string = ""
		#for i in message:
		#	string = string + i
		#print(string)
		data = request.get_json()
		print(data)
		msg = data['message']
		for i in msg:
			string=string+i
		result = get_bot_response(string)
#		result = {'Name: '+name+'<br/>'+'Age: '+age+'<br/>'+'Weight: '+weight+'<br/>'+'Blood Pressure: '+bp+'<br/>'+'Ailments Noted: '+". ".join(message)+'<br/>'+'Analysis: '+"".join(result)+'<br/>'}
		return jsonify({"result": result})
api.add_resource(mj,"/mj/<string:message>")
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000)

