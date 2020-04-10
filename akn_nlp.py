# RACS

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def backend():
    # Importing the dataset
    dataset = pd.read_csv('r.tsv', delimiter = '\t')
    test_data = pd.read_csv('testing.csv', delimiter = ',')
           
    c = []
    ct = []
            
    # Cleaning the texts
    for i in range(0, 1105):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        c.append(review)
             
    col = test_data.shape[0] 
          
    for i in range(0, col):
        review = re.sub('[^a-zA-Z]', ' ', test_data['Review'][i])  
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        ct.append(review)
            
    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1510)
    X = cv.fit_transform(c).toarray()
    X_test = cv.transform(ct).toarray()
    y = dataset.iloc[:, 1].values
            
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X, y)
            
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    ans, count = np.unique(y_pred, return_counts=True)
            
    #Writing into file
    df = pd.DataFrame(count)
    df.to_csv('ans.csv')
        

    neg = count[0]
    pos = count[1]    
    
    plt.switch_backend('agg')
    labels = 'Negative Reviews' , 'Positive Reviews'
    size = [neg,pos]
    colors = ['lavender','cornflowerblue']
    explode = (0,0.1)
    
    plt.pie(size, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True)

    plt.axis('equal')
    plt.savefig('C:/Users/nirma/Desktop/College/6 sem/Minor 2/static/img/res.png')    
    
backend()
