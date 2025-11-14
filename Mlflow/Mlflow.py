import json
import pandas as pd
import numpy as np
from numpy.ma.extras import average
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from sklearn.metrics import (classification_report,
                             recall_score, precision_score,f1_score, accuracy_score)
from catboost import CatBoostClassifier

import nltk
from nltk.corpus import stopwords
import mlflow
import re
import random
from tqdm import tqdm

mlflow.set_experiment("Mlflow_experiments")
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Download Russian stopwords
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))
path = "kinopoisk.jsonl"
column_to_processed = "content"


def experiment(test_size, seed, vectorizer,model, average, run_name, russian_stopwords, path, column_to_processed):

   # Load the JSONL file

   def load_jsonl(file_path):
      data = []
      with open(file_path, 'r', encoding='utf-8') as f:
          for line in f:
              data.append(json.loads(line))
      return pd.DataFrame(data)

   # Preprocess text
   def preprocess_text(text):
      # Convert to lowercase
         text = text.lower()
         # Remove punctuation
         text = re.sub(r'[^\w\s]', '', text)
         # Remove stopwords
         words = text.split()
         words = [word for word in words if word not in russian_stopwords]
         return ' '.join(words)

   with mlflow.start_run(run_name=run_name):
      mlflow.log_param("test_size",test_size)
      mlflow.log_param("random_state",  seed)
      mlflow.log_param("vectorizer", type(vectorizer).__name__)
      mlflow.log_param("model",  type(model).__name__)
      mlflow.log_param("average", average)

      # Load the data
      df = load_jsonl(path)

      # Preprocess the content
      df['processed_content'] = df[column_to_processed].apply(preprocess_text)

      # Split the data into train and test sets
      X_train, X_test, y_train, y_test = train_test_split(
         df['processed_content'], df['grade3'], test_size=test_size, random_state=seed
      )

      # Create bag of words representation
      X_train_bow = vectorizer.fit_transform(X_train)
      X_test_bow = vectorizer.transform(X_test)

      # Train the classifier

      model.fit(X_train_bow, y_train)

      # Make predictions
      y_pred = model.predict(X_test_bow)

      mlflow.log_metric("recall", recall_score(y_test, y_pred, average=average ))
      mlflow.log_metric("precision", precision_score(y_test, y_pred, average=average))
      mlflow.log_metric("f1", f1_score(y_test, y_pred, average=average))
      mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

      mlflow.sklearn.log_model(model, "model")

      # Evaluate the model
      #print(classification_report(y_test, y_pred))

      return classification_report(y_test, y_pred)

default_results = experiment(0.2, 42, CountVectorizer(),MultinomialNB(),
                             "macro","default_results",russian_stopwords,
                             path, column_to_processed)
for i in tqdm(range(7)):
   test_size = random.choice([0.2,0.25,0.3])
   seed = random.choice([42,128,2])
   vectorizer = random.choice([CountVectorizer(dtype=np.float32), TfidfVectorizer()])
   model =  random.choice([MultinomialNB(), LGBMClassifier(verbose = -1), CatBoostClassifier(iterations = 50, verbose =False)])
   average= random.choice(["macro", "weighted"])
   experiment(test_size,seed, vectorizer,model,average,f"experiment_{i+1}",
              russian_stopwords, path,column_to_processed)

def classify_review(review_text):
   processed_text = preprocess_text(review_text)
   bow_representation = vectorizer.transform([processed_text])
   prediction = clf.predict(bow_representation)
   return prediction[0]
      # Function to classify new reviews


   # Example usage
   #example_review = "Ваш текст рецензии на русском языке"
#print(f"Классификация: {classify_review(example_review)}")