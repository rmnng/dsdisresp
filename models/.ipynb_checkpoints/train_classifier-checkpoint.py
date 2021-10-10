import sys

# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from termcolor import colored

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """
    Load data containing messages and categries from DB files. 
    
    Arguments:
        database_filepath -> DB file containing data        
    
    Output:
       X -> Predictors
       y -> Targets
       cat_cols -> categories to be predicted
    """    
    
    # load table form DB
    engine = create_engine("sqlite:///{file}".format(file=database_filepath))
    df = pd.read_sql_table('messages', engine)
    
    # select message as predictor    
    X = df.message.values
    
    # select categories as targets
    cat_cols = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    y = df[cat_cols].values    

    return X, y, cat_cols

def tokenize(text):
    """
    Function to replace URL strings, tokenize and lemmatize sentences. 
    
    Arguments:
        text -> one message from the database as text        
    
    Output:
       clean_tokens -> prepared text 
    """     
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    Class StartingVerbExtractor
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Building model/pipeline for multi-output category prediction. 
    This can be a model, a pipeline or even GridSearch (which would take a looong time to execute) 
    
    Arguments:
              
    
    Output:
       model -> built model/pipeline/grid search 
    """     
    
    #build pipeline   
    # after testing some pipelines and classifiers 
    pipe = Pipeline([
        ('featunion', FeatureUnion([
            ('txtpipe', Pipeline([
                ('v', CountVectorizer(tokenizer=tokenize)),
                ('t', TfidfTransformer())
            ])),
            ('txtstverb', StartingVerbExtractor())
        ])),
        ('cl', MultiOutputClassifier(RandomForestClassifier()))    

    ])   
 
    
    # This is the place to create GridSearchCV.
    # I commented it out due to a very long execution time of the fit function
    
#    parameters = {
#        'featunion__txtpipe__v__min_df': (0.1, 0.2, 0.3),
#        'cl__estimator__min_samples_split': (2, 3, 4),
#        'cl__estimator__n_estimators': (50, 100, 150)
#    }

#    return GridSearchCV(pipe, parameters)
    
    ## BTW: The best n_estimators param was 150    
        
    return pipe


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Fuction scoring X_test and calculating F1 score for the prediction results category by category. 
    Score for each category will be printed
    
    Arguments:
        model -> model to be used for prediction
        X_test -> predictors of the test part of the dataset
        Y_test -> targets of the test part of the dataset
        category_names -> categories to be predicted
    
    Output:
       
    """            
    
    y_pred = model.predict(X_test)
        
    score = []
    for i, col in enumerate(category_names):    
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[:, i], y_pred[:, i], average='weighted')
        score.append(fscore)
        print("f-score for category {} is {}".format(colored(col, 'red'), colored(fscore, 'blue')))

    #average of all f1 scores
    print("{}".format(colored('================================================', 'green')))
    print("average f-score for all columns is {}".format(colored(sum(score)/len(score), 'red', attrs=['bold'])))    
    

def save_model(model, model_filepath):    
    """
    Save model to a pickle file
    
    Arguments:
        model -> model to be saved
        model_filepath -> path to the pickle file    
    
    Output:
    
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