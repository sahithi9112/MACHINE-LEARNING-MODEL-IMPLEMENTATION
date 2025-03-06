import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
# For this example, we will use the 'spam.csv' dataset which can be found online.
# Make sure to download the dataset and place it in the same directory as your notebook.
data = pd.read_csv('spam.csv', encoding='latin-1')

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Drop unnecessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Encode labels: 'ham' as 0 and 'spam' as 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into features and target variable
X = data['message']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Vectorization
# Convert text data into numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Training
# Using Naive Bayes classifier for spam detection
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Model Prediction
y_pred = model.predict(X_test_vectorized)

# Model Evaluation
# Print the classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
