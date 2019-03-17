from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import scipy as sc
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import RegexpTokenizer
#TODO: Убрать стоп-слова
# Убрать символы !n числам или словам
# Перенести импорты в начало файла
# Запилить класс для препроцессинга текста(для финала)
# Протестить склерн модельки и запилить нейронку для классификации
# Заюзать self-attention для текста(вроде круто работает)
# Оформить нормально файл
stopWords = set(stopwords.words('english'))

example = 'All work and no play makes jack dull boy. All work and no play makes jack a dull boy.'
df = pd.read_csv('NLP/train.csv')
df = df.drop('qid',axis=1)

#train , test , y_train, y_test = train_test_split(df.question_text,df.target,test_size=0.3)



def standardize_text(df, text_field = 'question_text'):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def delete_stopw(token_list):
    """
    Удаляем стоп-слова из листа после токенизации
    """
    clear_text = []
    for word in token_list:
        if word not in stopWords:
            clear_text.append(word)
    return clear_text



dataset = standardize_text(df)
tokenizer = RegexpTokenizer(r'\w+')

dataset.question_text = dataset.question_text.apply(tokenizer.tokenize)
dataset.question_text = dataset.question_text.apply(lambda x: delete_stopw(x))

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
stemmer.stem('romans')


ex = ['dogs','romans','kills']

def stemming(token_list):
    """
    Просто проходим по токенам
    и аппендим чистые в новый лист
    """
    stemmed_list = []
    for word in token_list:
        stemmed_list.append(stemmer.stem(word))
    return stemmed_list

dataset.question_text = dataset.question_text.apply(lambda x:stemming(x))
