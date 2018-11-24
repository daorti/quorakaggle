

def Vectorizer(X_train_, X_test_, name_, analyzer_, encoding_, lowercase_, token_pattern_, ngram_range_, stop_words_, n_features_):
    
    vect=None
    if name_=='CoutVectorizer':
        print("Vectorizando con CountVectorizer...")
        vect = CountVectorizer(analyzer=analyzer_,encoding=encoding_, lowercase=lowercase_, min_df=5,
                               token_pattern=token_pattern_,ngram_range=ngram_range_, stop_words = stop_words_)
        
    elif name_=='TfidfVectorizer':
        print("Vectorizando con TfidfVectorizer...")
        vect = TfidfVectorizer(analyzer=analyzer_,encoding=encoding_, lowercase=lowercase_, min_df=5,
                               token_pattern=token_pattern_,ngram_range=ngram_range_, stop_words = stop_words_)

    elif name_=='HashingVectorizer':
        print("Vectorizando con HashingVectorizer...")
        vect = HashingVectorizer(analyzer=analyzer_,encoding=encoding_, lowercase=lowercase_, n_features=n_features_,
                                 token_pattern=token_pattern_,ngram_range=ngram_range_, stop_words = stop_words_)

    else:
        return "No existe ningun vectorizador con ese nombre"
    
    vect.fit(list(X_train_) + list(X_test_))
    X_train_vect = vect.transform(X_train_)
    X_test_vect = vect.transform(X_test_)
    
    return X_train_vect, X_test_vect



def Stemmer(data_, name_, field_):
    
    def stemming(text):
        text = [stemmer.stem(word) for word in text.split()]
        return " ".join(text)
    
    if name_=='SnowBall':
        print("Lematizando con SnowBall...")
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return data_[field_].apply(stemming)
        
    elif name_=='Porter':
        print("Lematizando con Porter...")
        stemmer = PorterStemmer()
        return data_[field_].apply(stemming)
    
    else:
        return 'No existe este lematizador'