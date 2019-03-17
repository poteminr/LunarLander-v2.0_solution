import pandas as pd
import scipy as sc
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer


class TextPreproc(object):
    """
    Класс для препроцессинга текста
    """
    def __init__(self,language='english'):
        self.stopWords = set(stopwords.words(language))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = SnowballStemmer(language)

    def standardize_text(self,df, column_name):
        """
        Удаляем лишние символы из текста
        """
        df[column_name] = df[column_name].str.replace(r"http\S+", "")
        df[column_name] = df[column_name].str.replace(r"http", "")
        df[column_name] = df[column_name].str.replace(r"@\S+", "")
        df[column_name] = df[column_name].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
        df[column_name] = df[column_name].str.replace(r"@", "at")
        df[column_name] = df[column_name].str.lower()
        return df

    def tokinizer(self,df,column_name):
        """
        Преобразуем строки текста из pandas-столбика в лист с токенами
        """
        df[column_name] = df[column_name].apply(self.tokenizer.tokenize)
        return df


    def delete_stopw(self,token_list):
        """
        Удаляем стоп-слова из листа после токенизации
        """
        clear_text = []
        for word in token_list:
            if word not in self.stopWords:
                clear_text.append(word)
        return clear_text

    def stemming(self,token_list):
        """
        Просто проходим по токенам
        и аппендим чистые в новый лист
        """
        stemmed_list = []
        for word in token_list:
            stemmed_list.append(self.stemmer.stem(word))
        return stemmed_list
#
#Example
# df = pd.read_csv('NLP/train.csv')
# df = df.drop('qid',axis=1)
#
# txt_preproc = TextPreproc(language='english')
#
# df = txt_preproc.standardize_text(df=df,column_name='question_text')
#
# df = txt_preproc.tokinizer(df=df,column_name='question_text')
#
# df['question_text'] = df['question_text'].apply(lambda x: txt_preproc.delete_stopw(x))
#
# df['question_text'] = df['question_text'].apply(lambda x: txt_preproc.stemming(x))
