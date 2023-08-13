import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from textblob import TextBlob

def get_normalized_data(df):
    '''
    To perform normalization in redacted_sentences of dataframe.
    Parameter
    ----------
    df : Dataframe
        data from .tsv file
    Returns
    -------
        dataframe after normalization.
    '''
    df['redacted_sentence'] = df['redacted_sentence'].map(lambda sentence : sentence.replace('"',''))
    df['redacted_sentence'] = df['redacted_sentence'].map(lambda sentence : sentence.replace('',''))
    df['redacted_sentence'] = df['redacted_sentence'].map(lambda sentence : sentence.replace('(',''))
    df['redacted_sentence'] = df['redacted_sentence'].map(lambda sentence : sentence.replace(')',''))
    df['redacted_sentence'] = df['redacted_sentence'].map(lambda sentence : sentence.replace('<br />',''))
    return df

def get_data():
    '''
    To get data in .tsv file from github link and convert it into dataframe.

    Returns
    -------
        dataframe after performing normalization.
    '''
    df = pd.read_csv('https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv',sep='\t',on_bad_lines='skip',names=['github_name','type','redacted_name','redacted_sentence'])
    #df = pd.read_csv('docs/unredactor.tsv',sep='\t',on_bad_lines='skip', names=['github_name','type','redacted_name','redacted_sentence'])
    return get_normalized_data(df)

def calculate_score(sentence_text):
    '''
    To calculate polarity of a sentence
    Parameter
    ----------
    sentence : string
        redacted_sentence
    Returns
    -------
        polarity score
    '''
    blob = TextBlob(sentence_text)
    total_polarity_score = 0
    count = 0
    for sentence in blob.sentences:
        count += 1
        total_polarity_score += sentence.sentiment.polarity
    return round(total_polarity_score/count,2)

def feature(df):
    '''
    To create features
    Parameter
    ----------
    df : Dataframe
        data from .tsv file
    Returns
    -------
        dataframe after creating features.
    '''
    df['name_length'] = df['redacted_name'].map(lambda name: len(name))
    df['sentence_length'] = df['redacted_sentence'].map(lambda sentence: len(sentence))
    df['sentiment_score'] = df['redacted_sentence'].map(lambda sentence: calculate_score(sentence))
    return df

def get_features_data(df,type):
    '''
    To get data for training and validation
    Parameter
    ----------
    df : Dataframe
        data from .tsv file
    type : str
        to get data type
    Returns
    -------
        list of dict
    '''
    training_features_list = []
    redacted_name = list(df['redacted_name'])
    name_length = list(df['name_length'])
    sentence_length = list(df['sentence_length'])
    sentiment_score = list(df['sentiment_score'])
    if type.lower() == 'training':
        for row in zip(redacted_name,name_length,sentence_length,sentiment_score):
            features_dict = {}
            features_dict['redacted_name'] = row[0]
            features_dict['name_length'] = row[1]
            features_dict['sentence_length'] = row[2]
            features_dict['sentiment_score'] = row[3]
            training_features_list.append(features_dict)
    else:
        for row in zip(redacted_name,name_length,sentence_length,sentiment_score):
            features_dict = {}
            features_dict['name_length'] = row[1]
            features_dict['sentence_length'] = row[2]
            features_dict['sentiment_score'] = row[3]
            training_features_list.append(features_dict)
    return training_features_list,redacted_name

def get_vectorized_data(df,training_features_list):
    '''
    To vectorize
    Parameter
    ----------
    df : Dataframe
        data from .tsv file
    training_features_list : list
        list of dict
    Returns
    -------
        data to use in fit and fit_transform
    '''
    vectorizer = DictVectorizer()
    name_data_features = vectorizer.fit_transform(training_features_list).toarray()
    labels = list(df['redacted_name'])
    return name_data_features, labels

def get_scores(validation_labels, predicted_labels):
    '''
    To get precision,recall and f1 scores
    Parameter
    ----------
    validation_labels :
        y_true
    predicted_labels :
        y-pred
    Returns
    -------
        precision,recall and f1 scores
    '''
    precision = metrics.precision_score(y_true=validation_labels, y_pred=predicted_labels,average='weighted')
    recall = metrics.recall_score(y_true=validation_labels, y_pred=predicted_labels,pos_label="positive",average='weighted')
    f1 = metrics.f1_score(y_true=validation_labels, y_pred=predicted_labels,pos_label="positive",average='weighted')
    print('Precision: ', precision)
    print('Recall: ' , recall)
    print('f1: ', f1)

def main():
    '''
    To take data from .tsv file and print scores
    Returns
    -------
        prints precision,recall and f1 scores
    '''
    warnings.filterwarnings("ignore")
    normalized_df = get_data()
    df = feature(normalized_df)
    training_features_list,labels = get_features_data(df,'training')

    #Vectorize the data
    vectorizer = DictVectorizer()
    name_data_features = vectorizer.fit_transform(training_features_list).toarray()

    classifier = KNeighborsClassifier(n_neighbors=20)
    classifier.fit(name_data_features,labels)

    validation_features_list,labels = get_features_data(df,'validation')
    input = vectorizer.transform(validation_features_list).toarray()
    predicted_labels = classifier.predict(input)
    get_scores(labels,predicted_labels)

if __name__ == "__main__":
    main()