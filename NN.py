# Logistic regression and one layer Neural Network approach to classify the reviews

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix

#importing dataset
dataset = pd.read_csv('r.tsv', delimiter = '\t')

#cleaning of the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1105):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#feature extraction
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#LOGISTIC CLASSIFIER 
logclf = sklearn.linear_model.LogisticRegressionCV()
logclf.fit(X_train, Y_train)
Y_pred = logclf.predict(X_test)

print(' Train Set Score = ', logclf.score(X_train, Y_train))
print(' Test Set Score = ', logclf.score(X_test, Y_test))



#NN
Y = y.reshape(1105,1) 
Y = Y.T
X = X.T

#sigmoid activation function
def sigmoid (z):
    ans = 1/(1+np.exp(-z))
    return ans

n_x = X.shape[0]
n_h = 5
n_y = Y.shape[0]

#parameters intialization function
def initialize_parameters(n_x, n_h, n_y):
   
    W1=np.random.randn(n_h,n_x)*.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*.01
    b2=np.zeros((n_y,1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

#forward propagation function
def forward_propagation(X, parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
   
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache    
    
#cost evaluation function
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]

    cost=(-1/m)*(np.sum((np.multiply(Y,np.log(A2))+(np.multiply((1-Y),np.log(1-A2))))))  
    cost = float(np.squeeze(cost))  
    
    assert(isinstance(cost, float))
    
    return cost

#backward propagation function
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    Z1=cache["Z1"]
    A1=cache["A1"]
    Z2=cache["Z2"]
    A2=cache["A2"]
    
    dZ2=A2-Y
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.multiply(np.dot(parameters["W2"].T,dZ2),1-np.power(A1,2))
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
   
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

#gradient descent
def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1=W1-np.multiply(learning_rate,dW1)
    b1=b1-np.multiply(learning_rate,db1)
    W2=W2-np.multiply(learning_rate,dW2)
    b2=b2-np.multiply(learning_rate,db2)
    
    parameters = {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2}
    return parameters

#NN model function
def model(X,Y, iter = 10000, print_cost = False):
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(0, iter):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

#prediction function
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    #predictions = A2
    
    return predictions


parameters = model(X_train,Y_train, iter = 10000, print_cost = True)
nn_pred = predict(parameters, X_test)

cm = confusion_matrix(Y_test, nn_pred.T)
acc = cm[0][0] + cm[1][1] / n_x

print("Neural Network accuracy : ",acc)
    
    
