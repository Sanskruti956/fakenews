import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier

data = pd.read_csv('news.csv')

print (data.columns)
print (data.head())


print (data.head())

data.head()

data.shape

data.isnull().sum()

labels=data.label

labels

labels.head()

data['text'] = data['text'].str.lower()

X_train, X_test, y_train, y_test = train_test_split(data['text'], labels, test_size=0.2, random_state=42)

X_train.head()

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

nb_classifier = MultinomialNB()
nb_classifier.fit(tfidf_train, y_train)

MultinomialNB()

predictions = nb_classifier.predict(tfidf_test)

accuracy = accuracy_score(y_test,predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test,predictions))

print(f"Accuracy : {round(score*100,2)}%")

confusion_matrix(y_test,y_pred,labels=["FAKE","REAL"])

import pickle
file='FakeNews.pkl'
pickle.dump(pac,open(file,'wb'))

# Input new text for prediction
new_text = "Kerry to go to Paris in gesture of sympathy"

# Preprocess the new text (similar to the preprocessing steps in the training phase)
new_text = new_text.lower()  # Example: Convert to lowercase

# Vectorize the new text using the same TF-IDF vectorizer
tfidf_new = tfidf_vectorizer.transform([new_text])

# Make predictions
new_prediction = nb_classifier.predict(tfidf_new)

predicted_label = "Fake" if new_prediction[0] == 0 else "Real"

# Display the result
print(f"Predicted Label for New Text: {new_prediction[0]}")
