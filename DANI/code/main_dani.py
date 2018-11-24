
# Importamos funciones
from aux_functions import Stemmer
from aux_functions import Vectorizer

# STEMMING
data['question_text'] = Stemmer(data_=data, name_='Porter', field_='question_text')

# VECTORIZAMOS
#   name= [CoutVectorizer,TfidfVectorizer,HashingVectorizer]
X_train_vect, X_test_vect=Vectorizer(X_train_=X_train, X_test_=X_test, name_='HashingVectorizer',
                                      analyzer_='word', encoding_='ascii',lowercase_=True,
                                      token_pattern_=r'\w{1,}', ngram_range_=(1, 3),
                                      stop_words_='english', n_features_=50)


