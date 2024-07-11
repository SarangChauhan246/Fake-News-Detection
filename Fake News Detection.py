# Importing Necessary libraries


import pandas as pd 
import numpy as np 
import re # regular expression 
from nltk.corpus import stopwords # remove words that do not provide much value to data
from nltk.stem.porter import PorterStemmer #To perform stemming - only root words 
from sklearn.feature_extraction.text import TfidfVectorizer # convert text to valuable data 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 
import seaborn as sns


import nltk
nltk.download('stopwords')

#printing stopwords
print(stopwords.words('english'))

#loading dataset to pandas data frame
news_dataset = pd.read_csv('train_fake.csv')

news_dataset.shape

#printing first 5 rows of dataset
news_dataset.head()


sns.countplot(data =news_dataset,x='label',order=news_dataset['label'].value_counts().index)

# PreProccessing of dataset 


#counting number of missing values
news_dataset.isnull().sum()

#replacing missing values with empty string
news_dataset = news_dataset.fillna('')

#merging author name and title to predict according to that
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

print(news_dataset['content'])

#separating data and label
X = news_dataset.drop(columns = 'label', axis = 1) # axis = 1 for column| 0 for row
Y = news_dataset['label']

print(X)
print(Y)

#Stemming process - finding root word 
#example = playing -> play
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # remove all the words that are not letters or words
    stemmed_content = stemmed_content.lower()         
    stemmed_content = stemmed_content.split()     
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]  # remove stopwords and perform stemming on the words in the list
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)

print(Y)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

#spliting the dataset into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

# Deploying machine learning models on dataset

#Training the model

#LOGISTIC REGRESSION
LR = LogisticRegression()

LR.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = LR.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
predict_lr = LR.predict(X_test)
accuracy_lr = accuracy_score(predict_lr, Y_test)

X_new = X_test[3]

prediction = LR.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

print(Y_test[3])

from sklearn.metrics import classification_report
print('Accuracy score of the logistic regression: ', accuracy_lr)
print(classification_report(predict_lr, Y_test))

#GRADIENT BOOSTER

from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier()
GB.fit(X_train, Y_train)

predict_gb = GB.predict(X_test)

accuracy_gb = GB.score(X_test, Y_test)

print('Accuracy score of the Gradient Booster: ', accuracy_gb)
print(classification_report(Y_test, predict_gb))

#RANDOM FOREST 

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
RF.fit(X_train, Y_train)

predict_rf = RF.predict(X_test)

accuracy_rf = RF.score(X_test, Y_test)

print('Accuracy score of the Random Forest: ', accuracy_rf)
print(classification_report(Y_test, predict_rf))

#DECISION TREE

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)

predict_dt = DT.predict(X_test)

accuracy_dt = DT.score(X_test, Y_test)

print('Accuracy score of the Decision Tree: ', accuracy_dt)
print(classification_report(Y_test, predict_dt))

import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Actual': Y_test,
    'Decision Tree': predict_dt,
    'Random Forest': predict_rf,
    'Logistic Regression': predict_lr,
    'Gradient Booster': predict_gb,
})

accuracy_data = {
    'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Gradient Booster'],
    'Accuracy': [accuracy_dt, accuracy_rf, accuracy_lr, accuracy_gb]
}

df_accuracy = pd.DataFrame(accuracy_data)

# Plot the accuracy scores
plt.figure(figsize=(10, 6))
plt.bar(df_accuracy['Model'], df_accuracy['Accuracy'], color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of Different Models')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
for index, value in enumerate(df_accuracy['Accuracy']):
    plt.text(index, value, f"{value:.2f}", ha='center', va='bottom')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(Y_test, predict_dt)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()

