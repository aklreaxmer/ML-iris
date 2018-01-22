import random
import numpy


#parse data file
data_file = open("iris.data")

data = [] # input space
for line in data_file:
	if len(line) > 1:
		line = line.split(",")
		line[-1] = line[-1].strip()
		line.insert(0,1) #add a bias of 1 at index 0 of each input vector
		data.append(line)


#return a best hypothesis "w" based on input space "data"
def learn(n, rate, data):
	w = numpy.random.rand(5) #weight vector has four features + bias
	for i in range(n):
		x = random.choice(data) #choose an input vector from input space
		correct = 0 if x[-1] == "Iris-versicolor" else 1 # "versicolor" species = 0, "setosa" = 1
		x = x[:-1]
		x = numpy.array(x).astype(numpy.float) 

		prediction = numpy.dot(w, x) #make our prediction with the current hypothesis "w"
		prediction = 0 if prediction < 0 else 1	 #step function to classify as 0 or 1

		error = correct - prediction
		w = w + (rate * error) * x #update hypothesis
	return w

#predict the species of "x" using our learned "w"
def predict(x, w):
	prediction = numpy.dot(w, x)
	prediction = 0 if prediction < 0 else 1
	species = "Iris-versicolor" if prediction == 0 else "Iris-setosa"
	print("Predicted species: " + species)


rate = .1
n = 100 #sample size (50 versicolor, 50 setosa)
	
#Find a best hypothesis
w = learn(n, rate, data)

#Test it out
predict([1, 5.0, 3.9, 1.2, 0.3], w) #Should be setosa
predict([1, 5.4, 3.0, 4.5, 1.5], w) #Should be verticolor

