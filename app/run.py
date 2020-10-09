import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Line, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# load model training and test statistics
df_train_stats = pd.read_sql_table('train_stats', engine).sort_values(['category'])
df_test_stats = pd.read_sql_table('test_stats', engine).sort_values(['category'])
df_cat_stats = pd.read_sql_table('pct_non_zero', engine).sort_values(['category'])



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    ###
    graphs.append(
        {
            'data': [
                Bar(
                    x=df_cat_stats['category'],
                    y=df_cat_stats['pct_not_zero'],
                    name='Pct non zero (all data)'
                ),
                Scatter(
                    x=df_train_stats['category'],
                    y=df_train_stats['fscore'],
                    name='F1 (training set)'
                ),
                Scatter(
                    x=df_test_stats['category'],
                    y=df_test_stats['fscore'],
                    name='F1 (test set)'
                )
            ],

            'layout': {
                'title': 'Training and Test F-scores and Category Occurrence Rate',
                #'yaxis': {
                #    'title': "Count"
                #},
                #'xaxis': {
                #    'title': "Category"
                #}
                'margin': {'b':160}
            }
        }
    )

    graphs.append(
        {
            'data': [
                Bar(
                    x=df_cat_stats['category'],
                    y=df_cat_stats['pct_not_zero'],
                    name='Pct non zero (all data)'
                ),
                Scatter(
                    x=df_train_stats['category'],
                    y=df_train_stats['precision'],
                    name='Precision (training set)'
                ),
                Scatter(
                    x=df_test_stats['category'],
                    y=df_test_stats['precision'],
                    name='Precision (test set)'
                )
            ],

            'layout': {
                'title': 'Training and Test Model Precision and Category Occurrence Rate',
                #'yaxis': {
                #    'title': "Count"
                #},
                #'xaxis': {
                #    'title': "Category"
                #}
                'margin': {'b':160}
            }
        }
    )

    graphs.append(
        {
            'data': [
                Bar(
                    x=df_cat_stats['category'],
                    y=df_cat_stats['pct_not_zero'],
                    name='Pct non zero (all data)'
                ),
                Scatter(
                    x=df_train_stats['category'],
                    y=df_train_stats['recall'],
                    name='Precision (training set)'
                ),
                Scatter(
                    x=df_test_stats['category'],
                    y=df_test_stats['recall'],
                    name='Precision (test set)'
                )
            ],

            'layout': {
                'title': 'Training and Test Model Recall and Category Occurrence Rate',
                #'yaxis': {
                #    'title': "Count"
                #},
                #'xaxis': {
                #    'title': "Category"
                #}
                'margin': {'b':160}
            }
        }
    )
###
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()