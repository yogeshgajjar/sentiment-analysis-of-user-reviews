import numpy as np
import pandas as pd
import warnings 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
from wordcloud import ImageColorGenerator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import sys
warnings.filterwarnings('ignore')


def read_file(filelist):
    """
    Creates dataframe of the dataset by UCI - Sentiment Analysis 

    :param filelist: list of file directory.
    """
    #UCI dataset Dataframe
    df_uci = pd.concat([pd.read_csv(item, header=None, sep='\t') for item in filelist], axis=0)
    df_uci.columns = ['reviews', 'sentiment']
    
    #IMDB dataset DataFrame
    reviews_train = []
    for line in open('/home/yogesh/fall19/ml660/project/movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())
    
    df_imdb_train = pd.DataFrame(reviews_train, columns=['reviews'])
    
    reviews_test = []
    for line in open('/home/yogesh/fall19/ml660/project/movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())
    
    df_imdb_test = pd.DataFrame(reviews_train, columns=['reviews'])
    
    return df_uci, df_imdb_train, df_imdb_test

def datasetselection(df_uci, df_imdb_train, df_imdb_test):
    # use argv method to call configuration from the call
    """
    
    """
    if sys.argv[1] == '1':
        print("This includes training on 80% of whole dataset and testing on 20%")
        df_train_data = pd.concat([pd.DataFrame(df_uci['reviews']), df_imdb_train, df_imdb_test], axis=0)
        y = pd.concat([pd.DataFrame(df_uci['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment'])])    
        X = tfidfvectorization(df_train_data) 
        y = y['sentiment'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    elif sys.argv[1] == '2':
        print("This includes Train/Test on UCI dataset")
        df_train_data = pd.DataFrame(df_uci['reviews'])
        y = pd.DataFrame(df_uci['sentiment'])
        X = tfidfvectorization(df_train_data) 
        y = y['sentiment'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    elif sys.argv[1] == '3':
        print("This includes train on IMDB and test on UCI dataset")
        df_train_data, df_test_data = pd.concat([df_imdb_train, df_imdb_test]), pd.DataFrame(df_uci['reviews'])
        y_train, y_test = pd.concat([pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment'])]), df_uci['sentiment']
        X_train, X_test = tfidfvectorization(df_train_data), tfidfvectorization(df_test_data)

    else:
        print("done")
    # print("This includes training on 80% of whole dataset and testing on 20%")
    # df_train_data = pd.concat([pd.DataFrame(df_uci['reviews']), df_imdb_train, df_imdb_test], axis=0)
    # y = pd.concat([pd.DataFrame(df_uci['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment'])])    
    
    # if sys.argv[2] == '2':
    #     print("This includes training on 100% UCI dataset + 80% of Imdb dataset and testing on 20% Imdb dataset")
    #     df_train_data = pd.concat([pd.DataFrame(df_uci['reviews']), ])
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_train.shape) 
    return X_train, X_test, y_train, y_test

# df_train_data, y = datasetselection(df_uci, df_imdb_train, df_imdb_test)

def tfidfvectorization(df):
    """

    """
    stemmer = WordNetLemmatizer()

    df_array = df['reviews'].to_numpy()
    word_list = []
    for i in range(len(df_array)):
        tokens_new = word_tokenize(df_array[i])
        words = [word for word in tokens_new if word.isalpha()]
        words = [stemmer.lemmatize(word) for word in words]
        doc = ' '.join(words)
        word_list.append(doc) 
    
#     return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]


#     tfidfconv = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), max_features=2000, min_df=2, max_df=0.7)
    tfidfconv = TfidfVectorizer(stop_words=stopwords.words('english'))
    X = tfidfconv.fit_transform(word_list)
#     print(X.toarray().shape) 
    return X

# X = tfidfvectorization(df_train_data)

def test_train_split(X, y):
    """


    """
    y = y['sentiment'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     print(X_train.shape)
#     print(type(X_test))
#     print(y_train.shape)
#     print(y_test.shape) 
    return X_train, X_test, y_train, y_test


def logisticRegression(X_train, X_test, y_train, y_test):
    
    parameters = {'C': np.logspace(-2,3,6)}
    mod_lr = LogisticRegression()
    
    clf = GridSearchCV(mod_lr, parameters, cv=5)  # gridsearchCV with 5 fold CV
    clf.fit(X_train, y_train)
    lambda_scale = 1/clf.best_params_.get('C')  # calculating the best lambda 
    
    
    score_scale = clf.best_score_   # Average cross validation score to calculate the error 
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_score = 1 - accuracy_score(y_train, y_pred_train)  # Calculating train error 
    test_score = 1 - accuracy_score(y_test, y_pred_test) # Calculating test error 
    print("-----Logistic Regression--------") 
    print(confusion_matrix(y_test,y_pred_test))
    print(classification_report(y_test,y_pred_test))
    print(accuracy_score(y_test, y_pred_test))
    
#     sel_ = SelectFromModel(LogisticRegression(C= (1/clf.best_params_.get('C')), penalty='l1'))
#     sel_.fit(X_train_fit, y_train)
#     print("For l:", i)
#     selected_feat = X_train.columns[(sel_.get_support())]
#     print('selected features: {}'.format(len(selected_feat)))
#     print('c value', clf.best_params_.get('C'))
#     print('The best score:', clf.best_score_)
#     plist.append(len(selected_feat))
#     score.append(clf.best_score_)

def randomforest(X_train, X_test, y_train, y_test):
    """

    """
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train) 
    y_pred = classifier.predict(X_test)
    
#     print(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

def linearSVM(X_train, X_test, y_train, y_test):
    
    parameters = {'C': np.logspace(-2,3,6)}
    # GridsearchCV used to identify the best parameters from the range declared above. 
    gs = GridSearchCV(LinearSVC(), parameters, cv = 5)
    gs.fit(X_train, y_train)

    #Instanciating of the classifier LinearSVC to train SVC with kernel Linear and l1-penalized with loss = squared hinge loss. 
    svc = LinearSVC(penalty='l2', loss='squared_hinge', dual = False, C = gs.best_params_.get('C'))
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
#     loss_f_l1 = hamming_loss(y_test.loc[:,'Family'], y_pred_f_l1)

    print("CLASSIFICATION USING L1 PENALIZED SVM WITH LINEAR KERNEL")
#     print("LABEL - FAMILY")
#     print("")
    print("Test Score is :", svc.score(X_test, y_test, sample_weight=None))
#     print("Hamming loss is :", loss_f_l1)
    print("Best Penalty is:", gs.best_params_.get('C'))
    print(accuracy_score(y_test, y_pred))


filelist = ['/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/amazon_cells_labelled.txt', '/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/imdb_labelled.txt', '/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/yelp_labelled.txt']
df_uci, df_imdb_train, df_imdb_test = read_file(filelist)

X_train, X_test, y_train, y_test = datasetselection(df_uci, df_imdb_train, df_imdb_test)
# X = tfidfvectorization(df_train_data)
# X_train, X_test, y_train, y_test = test_train_split(X, y)
# randomforest(X_train, X_test, y_train, y_test) 
logisticRegression(X_train, X_test, y_train, y_test)
# linearSVM(X_train, X_test, y_train, y_test)