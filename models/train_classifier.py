import sys

import numpy as np
import pandas as pd
import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def load_data(database_filepath):
    '''
    Reads from a specified database and splits the data into messages and categories
    Input:
        database_filepath: (string) database file path
    Returns:
        X: (pandas dataframe) messages (to be convverted into features)
        y: (pandas dataframe) categories (labels)
        y.columns: (list of string) category (label) names
    '''
    
    # read in the file
    table_name = 'messages'
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, engine)
    
    # define features and label arrays
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1).copy()
    
    return X, y, y.columns



def tokenize(text):
    '''
    Tokenizes text
    Input:
        text: (string) text to be tokenized
    Return:
        tokens: (list of strings) tokens
    '''
    
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    #print(text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    #print(text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    
    #print(tokens)
    
    return tokens



def build_model():
    '''
    Creates a text processing and model pipeline with grid search
    Inputs: none
    Return: (object) grid search
    '''
    
    # define parameters for grid search
    parameters = {'clf__estimator__alpha': [0.5,1,1.5,2]}
    
    # create a text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RidgeClassifier()))
    ])
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model



def make_classification_report(y_true, y_pred):
    '''
    Create a pandas dataframe containing the precision, recall, fscore and support metrics
    for each category
    Inputs:
        y_true: label values in the data set (ground truth)
        y_pred: label values predicted by the model
    Return:
        df: (pandas dataframe) metrics
    '''
    df = pd.DataFrame(np.array(precision_recall_fscore_support(y_true, y_pred)).T)
    df.columns = ['precision','recall','fscore','support']
    df['category'] = y_true.columns
    df = df.reindex(columns=['category' , 'precision', 'recall', 'fscore', 'support'])
    print(df)
    
    return df
                  


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model predictions versus test data
    Inputs:
        model: (object) model
        X_test: (dataframe) feature 
        Y_test: 
        category_names: 
    Return: None
    '''
    Y_test_pred = model.predict(X_test)
    make_classification_report(Y_test, Y_test_pred)
    
    return



def save_model(model, model_filepath):
    '''
    Export model to a pickle file
    Inputs:
        model: (onbect) trained model
        model_filepath: (string) pickle file path
      Returns: None
    '''
    # export model as a pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    
    return



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()