import sys
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle



def load_data(database_filepath):
    """
    Load dataset from database and split it into feature and target DataFrames

    Args:
        database_filepath (str): path to database

    Returns:
        X (DataFrame): DataFrame with messages
        Y (DataFrame): DataFrame with categories of messages
        category_names (array of str): Array of category names
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("merged_data", con=engine)
    X = df.message
    Y = df.drop(["id","message","original","genre"], axis=1)
    
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenizes given text
    
    Args:
        text (str): text for tokenization
    
    Returns:
        tokens (array of str): array of words
    
    """
    # place "urlplaceholder" instead of url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    
    return tokens

def build_model():
    """
    Build a machine learning pipeline using grid search to find better parameters
    Args:
        None
    Return:
        cv (GridSearch object): model
    """

    #train pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #tested multiple parameters options
#    parameters = {
#                    'vect__max_df': (0.2, 0.5),
#                  'clf__estimator__n_estimators': [100,200],
#                  'clf__estimator__max_features': [2000, 4000]
#                 }

    #choose the best one and faster one to speed up model training 
    parameters = {
                     'vect__max_df': [0.2],
                   'clf__estimator__n_estimators': [100],
                   'clf__estimator__max_features': [2000]
                  }

    # use grid search to find better parameters. 
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)

    return cv
#    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model: prints precision, recall, f1-score and support for each category
    
    Args:
        model: pipeline
        X_test (DataFrame): test messages
        Y_test (DataFrame): test categories
        category_names (array of str): array of names of categories

    Return:
        None
    """

    #obtain predictions
    Y_pred = model.predict(X_test)

    #print classification report
    print(classification_report(Y_test, Y_pred, target_names = category_names))
    

def save_model(model, model_filepath):
    """
    Export the  model as a pickle file 

    Args:
        model: trained model
        model_filepath (str): path to pickle model

    Return:
        None
    """

    pickle.dump(model, open(model_filepath, 'wb'))

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
