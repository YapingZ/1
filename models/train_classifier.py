
# import packages
import sys
# import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, fbeta_score
import nltk
import pickle
nltk.download('punkt')
nltk.download('wordnet')


# load data from database
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessage', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    labels = list(Y)
    return X, Y, labels


# text processing
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

# build our model
def build_model():
    # text processing and model pipeline
    pipeline = Pipeline([
                         ('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                         ])

    # define parameters for GridSearchCV
    parameters = {'tfidf__norm': ['l1', 'l2'],
                  'clf__estimator__n_estimators': [50, 100, 200]}
    # create gridsearch object and return as final model pipeline
    model = GridSearchCV(pipeline, param_grid=parameters, cv=4)
    return model


# evaluate the model
def evaluate_model(model,X_test, Y_test, labels):

    # output model test results
    Y_pred = model.predict(X_test)

    for i, col in enumerate(labels):
        print('Category {} metrics:'.format(col))
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))




def export_model(model, model_filepath):
    # Export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) ==3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n Database: {}'.format(database_filepath))
        X, Y, labels = load_data(database_filepath)
        # train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        # build model pipeline
        print('Building model...')
        model = build_model()

        # train model
        print('Training model...')
        model.fit(X_train, Y_train)

        # evaluate model
        print('Evaluating model...')
        evaluate_model(model,X_test, Y_test, labels)

        # save model
        print('Saving model...')
        export_model(model, model_filepath)
        print('Congratulations! Model trained and saved')

    else:
        print('Please type the correct input!')


if __name__ == '__main__':
    main()


