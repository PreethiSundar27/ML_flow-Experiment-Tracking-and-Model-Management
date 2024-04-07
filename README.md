# ML_flow-Experiment-Tracking-and-Model-Management

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import numpy as np

@task(name="loading_data")

def load_data(file_path):

    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)


@task(name='preprocess_data_task')

def preprocess_data(X_train, X_test):

    """
    Preprocess text data using CountVectorizer.
    """
    vectorizer = CountVectorizer()
    
    X_train_vec = vectorizer.fit_transform(X_train)
    
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec

@task(name='train_model_task')

def train_model(X_train, y_train):

    """
    
    Train a Naive Bayes classifier.
    """
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf

@task(name='evaluate_model_task')

def evaluate_model(model, X_test, y_test):

    """
    Evaluate the model on test data.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

@flow(name="NB_Training_Flow")


def workflow():

    # Load data
    data_path=("file_path")
    data = load_data(data_path)
    data.fillna("", inplace=True)

# Convert Ratings to sentiment categories
    data['Sentiment'] = np.where(data['Ratings'].isin([1, 2]), 0, np.where(data['Ratings'] == 3, 1, 2))

    X = data['Review text']
    y = data['Sentiment']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess text data
    X_train_vec, X_test_vec = preprocess_data(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_vec, y_train)
    train_accuracy = evaluate_model(model, X_train_vec, y_train)
    test_accuracy = evaluate_model(model, X_test_vec, y_test)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test_vec, y_test)
    print("Train Accuracy:", train_accuracy)

    print("Test Accuracy:", accuracy)

if __name__ == "__main__":

    workflow()
