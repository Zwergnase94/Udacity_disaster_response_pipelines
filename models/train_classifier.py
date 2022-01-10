import sys

import joblib
import nltk
import pandas as pd
from sqlalchemy import create_engine
import string

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#download nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

STOP_WORDS = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)


def load_data(database_filepath):

    #load Data from Database to DataFrame
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('labeled_messages', engine)
    engine.dispose()

    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):

    # normalize case and remove punctuation
    text = text.translate(PUNCTUATION_TABLE).lower()

    # tokenize text
    tokens = nltk.word_tokenize(text)

    # lemmatize and remove stop words
    return [lemmatizer.lemmatize(word) for word in tokens
                                                    if word not in STOP_WORDS]


def build_model():

    clf = RandomForestClassifier(n_estimators=100)

    # Pipeline for transforming data, fitting to model and predicting
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', RandomForestClassifier())
                      ])

    # The param grid still very long to search
    parameters = {
        'clf__n_estimators': [50, 100, 150, 200, 250],
        'clf__min_samples_split': [2, 4, 6, 8, 10]
        
    }

    #Initialize a gridsearch object that is parallelized
    cv = GridSearchCV(pipeline, parameters, cv=3, verbose=10, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    # Generate predictions
    Y_pred = model.predict(X_test)

    # Print out the full classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    
    #sace the training model
    joblib.dump(model, model_filepath)


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