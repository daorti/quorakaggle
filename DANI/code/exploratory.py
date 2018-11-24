import os
os.chdir('C:/Users/danie/Desktop/Archivos/Kaggle/Quora/DANI/code')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import nltk
nltk.download()
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB

# Leemos los datos
data=pd.read_csv('../data/input/train.csv')
validation=pd.read_csv('../data/input/test.csv')

# Tipos de datos (comprobamos que el target es numérico) y la pinta del dataset
data.head()
data.dtypes
data.shape
data.columns

# ¿Qué tipo de targets hay?
data.groupby(['target']).agg(['count'])

# Eliminamos NA's
data = data.dropna()


########## STEMMING ##########
# Sirve para quedarnos con las raíces de las palabras
stemmer = SnowballStemmer("english", ignore_stopwords=True)
def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

data['question_text'].head()
data['question_text'] = data['question_text'].apply(stemming)
data['question_text'].head()

# Features & Label
X = data['question_text']
y = data['target']


########## TRAIN, TEST ##########

division=0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=division, random_state=42)
print(X_train.shape)
print(X_test.shape)


########## VECTORIZAMOS ##########

# Probamos con el CoutVectorizer.
# Crea una columna nueva para cada palabra diferente.
# Formando así una matrix mxn, donde m es el número de registros del df de train y m el total de palabras diferentes del df
ctv = CountVectorizer(analyzer='word',encoding='ascii', lowercase=True,
                      min_df=5, token_pattern=r'\w{1,}',ngram_range=(1, 3), stop_words = 'english')

ctv.fit(list(X_train) + list(X_test))
X_train_ctv=ctv.transform(X_train)
X_test_ctv=ctv.transform(X_test)


# Probamos con el TfidfVectorizer
# La diferencia con el CTV es que este vectorizador destaca las palabras más relevantes en detrimento de otras del tipo 'the'
tfv = TfidfVectorizer(analyzer='word',encoding='ascii', lowercase=True,
                      min_df=5, token_pattern=r'\w{1,}',ngram_range=(1, 3), stop_words = 'english')

tfv.fit(list(X_train) + list(X_test))
X_train_tfv=tfv.transform(X_train)
X_test_tfv=tfv.transform(X_test)


# Probamos con el HashingVectorizer
# El problema de las funciones anteriores es que el diccionario de palabras es muy grande.
# Es una función que transforma el texto en unaa cantidad de features que definimos
hv=HashingVectorizer(analyzer='word',encoding='ascii', lowercase=True, n_features=100,
                      token_pattern=r'\w{1,}',ngram_range=(1, 3), stop_words = 'english')

hv.fit(list(X_train) + list(X_test))
X_train_hv=hv.transform(X_train)
X_test_hv=hv.transform(X_test)



########## MODELOS ##########

# Logistic Regresion para CoutVectorizer
lg = LogisticRegression(C=1.0)
lg.fit(X_train_ctv, y_train)
roc_auc_score(y_test, lg.predict(X_test_ctv))

# Logistic Regresion para TfidfVectorizer
lg_tfv = LogisticRegression(C=1.0)
lg_tfv.fit(X_train_tfv, y_train)
roc_auc_score(y_test, lg_tfv.predict(X_test_tfv))

# Logistic Regresion para HashingVectorizer
lg_hv = LogisticRegression(C=1.0)
lg_hv.fit(X_train_hv, y_train)
roc_auc_score(y_test, lg_hv.predict(X_test_hv))




