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
import pickle
warnings.filterwarnings('ignore')


def uciData():

    """
    Reads UCI dataset .txt files (Amazon, IMDB, YELP),
    Reference for uci_data collection from https://github.com/hoomanm/Sentiment-Analysis
    :returns : DataFrame consisting of reviews and sentiments of user reviews in UCI dataset
    """

    uci_train_data = []
    uci_train_labels = []

    with open(sys.argv[1]+"/sentiment_labelled_sentences/amazon_cells_labelled.txt", 'r') as f:
        text = f.readlines()
        text = [x.strip() for x in text]

    for review in text:
        uci_train_data.append(review.split("\t")[0])
        uci_train_labels.append(review.split("\t")[1])

    with open(sys.argv[1]+"/sentiment_labelled_sentences/imdb_labelled.txt", 'r') as f:
        text = f.readlines()
        text = [x.strip() for x in text]

    for review in text:
        uci_train_data.append(review.split("\t")[0])
        uci_train_labels.append(review.split("\t")[1])

    with open(sys.argv[1]+"/sentiment_labelled_sentences/yelp_labelled.txt", 'r') as f:
        text = f.readlines()
        text = [x.strip() for x in text]

    for review in text:
        uci_train_data.append(review.split("\t")[0])
        uci_train_labels.append(review.split("\t")[1])

    df_uci_train = pd.DataFrame(uci_train_data, columns=['reviews'])
    df_uci_labels = pd.DataFrame(uci_train_labels, columns=['sentiment'])
    df_uci = pd.concat([df_uci_train, df_uci_labels], axis = 1)
    return df_uci

def imdbData():


    """
    Reads IMDB dataset .txt files,
    Modified dataset downloaded from https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/movie_data.tar.gz
    :returns df_imdb_train: DataFrame consisting train samples of user reviews in IMDB dataset
    :returns df_imdb_test: Dataframe consisting test samples of user reviews in IMDB dataset
    """

    reviews_train = []
    for line in open(sys.argv[1]+'/movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())
    df_imdb_train = pd.DataFrame(reviews_train, columns=['reviews'])

    reviews_test = []
    for line in open(sys.argv[1]+'/movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())
    df_imdb_test = pd.DataFrame(reviews_train, columns=['reviews'])

    return df_imdb_train, df_imdb_test

def datasetSelection(df_uci, df_imdb_train, df_imdb_test, target):

    """
    Different configuration for train/test of the dataset based on the user defined number(param: target).
    Total of 6 different configuration which includes -
        1. TRAIN AND TEST ON OVERALL DATASET INCLUDING IMDB AND UCI DATASET
        2. TRAIN AND TEST ON UCI DATASET
        3. TRAIN ON IMDB DATASET AND TEST ON UCI DATASET
        4. TRAIN AND TEST ON IMDB DATASET
        5. TRAIN ON 100% IMDB + 80% UCI DATASET AND TEST ON 20% UCI DATASET
        6. TRAIN ON 100% UCI + 80% IMDB DATASET AND TEST ON 20% IMDB DATASET

    :param df_uci: Dataframe of UCI dataset with columns "review"(user review) and "sentiment"(label either positive or negative).
    :param df_imdb_train: Dataframe of IMDB train dataset with only column "review"(user review).
    :param df_imdb_test: Dataframe of IMDB test dataset with only column "review"(user review).
    :param target: Target is an interger value represeting the configuration value.

    :returns X_train: Dataframe containing train samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :returns X_test: Dataframe containing test samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :returns y_train: Dataframe containing train labels for a particular configurations specified in terminal.
    :returns y_test: Dataframe containing test labels for a particular configurations specified in terminal.
    """

    if target == 1:
        print("CONFIGURATION 1: TRAIN AND TEST ON OVERALL DATASET INCLUDING IMDB AND UCI DATASET\n")
        df_train_data = pd.concat([pd.DataFrame(df_uci['reviews']), df_imdb_train, df_imdb_test], axis=0)
        y = pd.concat([pd.DataFrame(df_uci['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment'])])
        X = tfidfVectorization(df_train_data, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        y_train['sentiment'], y_test['sentiment'] = y_train.sentiment.astype(float), y_test.sentiment.astype(float)

    elif target == 2:
        print("CONFIGURATION 2: TRAIN AND TEST ON UCI DATASET\n")
        df_train_data = pd.DataFrame(df_uci['reviews'])
        y = pd.DataFrame(df_uci['sentiment'])
        X = tfidfVectorization(df_train_data, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    elif target == 3:
        print("CONFIGURATION 3: TRAIN ON IMDB DATASET AND TEST ON UCI DATASET\n")
        df_train_data, df_test_data = shuffle(pd.concat([df_imdb_train, df_imdb_test]), random_state = 7), shuffle(pd.DataFrame(df_uci['reviews']), random_state = 7)
        y_train, y_test = shuffle(pd.concat([pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment'])]), random_state = 7), shuffle(pd.DataFrame(df_uci['sentiment'], columns=['sentiment']), random_state = 7)
        y_train['sentiment'], y_test['sentiment'] = y_train.sentiment.astype(float), y_test.sentiment.astype(float)
        X_train, X_test = tfidfVectorization(df_train_data, 2), tfidfVectorization(df_test_data, 2)

    elif target == 4:
        print("CONFIGURATION 4: TRAIN AND TEST ON IMDB DATASET\n")
        y_train, y_test = shuffle(pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), random_state = 7), shuffle(pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), random_state = 7)
        df_imdb_train, df_imdb_test = shuffle(df_imdb_train, random_state = 7), shuffle(df_imdb_test, random_state = 7)
        X_train, X_test = tfidfVectorization(df_imdb_train, 1), tfidfVectorization(df_imdb_test, 1)

    elif target == 5:
        print("CONFIGURATION 5: TRAIN ON 100% IMDB + 80% UCI DATASET AND TEST ON 20% UCI DATASET\n")
        df_train_data, df_test_data, y_train, y_test = splitTrain(df_uci, df_imdb_test, df_imdb_train, 1)
        y_train['sentiment'] = y_train.sentiment.astype(float)
        y_test['sentiment'] = y_test.sentiment.astype(float)
        X_train, X_test = tfidfVectorization(df_train_data, 3), tfidfVectorization(df_test_data, 3)

    elif target == 6:
        print("CONFIGURATION 6: TRAIN ON 100% UCI + 80% IMDB DATASET AND TEST ON 20% IMDB DATASET\n")
        df_train_data, df_test_data, y_train, y_test = splitTrain(df_uci, df_imdb_test, df_imdb_train, 2)
        y_train['sentiment'] = y_train.sentiment.astype(float)
        y_test['sentiment'] = y_test.sentiment.astype(float)
        X_train, X_test = tfidfVectorization(df_train_data, 4), tfidfVectorization(df_test_data, 4)

    else:
        print("Press a valid configuration number from 1 to 6")

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    return X_train, X_test, y_train, y_test

def splitTrain(df_uci, df_imdb_test, df_imdb_train, target):


    """
    Splitting the dataset for configuration 5 and 6 into their required respective configurations.
    :param df_uci: Dataframe of UCI dataset with columns "review"(user review) and "sentiment"(label either positive or negative).
    :param df_imdb_test: Dataframe of IMDB test dataset with only column "review"(user review).
    :param df_imdb_train: Dataframe of IMDB train dataset with only column "review"(user review).
    :param target: Target is an interger value which is either 1 or 2
    :returns df_train_data: Dataframe containing train samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :returns df_test_data: Dataframe containing test samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :returns y_train: Dataframe containing train labels for a particular configurations specified in terminal.
    :returns y_test: Dataframe containing test labels for a particular configurations specified in terminal.
    """


    if (target == 1):
        df_train_data = shuffle(pd.concat([pd.DataFrame(df_uci.iloc[0:2401,0]), df_imdb_train, df_imdb_test], axis=0), random_state = 7)
        df_test_data = shuffle(pd.DataFrame(df_uci.iloc[2401:,0]), random_state=7)
        y_train = shuffle(pd.concat([pd.DataFrame(df_uci.iloc[0:2401, 1]), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment'])]), random_state = 7)
        y_test = shuffle(pd.DataFrame(df_uci.iloc[2401:, 1]), random_state = 7)

    if (target == 2):
        df_train_data = pd.concat([pd.DataFrame(df_uci['reviews']), df_imdb_train, df_imdb_test.iloc[0:20000]], axis=0)
        df_test_data = df_imdb_test.iloc[20000:]
        y_train = pd.concat([pd.DataFrame(df_uci['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(25000)], columns=['sentiment']), pd.DataFrame([1 if i < 12500 else 0 for i in range(20000)], columns=['sentiment'])])
        y_test = pd.DataFrame([1 for i in range(5000)], columns=['sentiment'])

    return df_train_data, df_test_data, y_train, y_test

def tfidfVectorization(df, target):


    """
    Function for preprocessing and converting the word list to unique vectors using tfidfvectorization package.
    Preprocessing includes -
        1. Word tokenize, that means dividing the sentences into separate words. It includes special characters and numbers also.
        2. Removing all the special charaters, numbers and not required words
        3. Lemmatizing the word list
        4. Converting into word vectors using tfidfvectorization package.
    :param df: Dataframe for preprocessing.
    :param target:  Target is an interger value which is either 1, 2, 3 or 4.
    :returns X: Feature vector of type
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
    wordcloud(word_list)

    if target == 1:
        tfidfconv = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'))
        X = tfidfconv.fit_transform(word_list)

    elif target == 2:
        tfidfconv = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), max_features= 4000)
        X = tfidfconv.fit_transform(word_list)

    elif target == 3:
        tfidfconv = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), max_features= 1281)
        X = tfidfconv.fit_transform(word_list)

    elif target == 4:
        tfidfconv = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), max_features= 31366)
        X = tfidfconv.fit_transform(word_list)

    else:
        print("done")

    return X

def wordcloud(words):
    words_string = TreebankWordDetokenizer().detokenize(words)
    char_mask = np.array(Image.open("/home/yogesh/Git/sentiment_analysis/image.jpg"))
    image_colors = ImageColorGenerator(char_mask)
    wordcloud = WordCloud(background_color="black", max_words=200, width=400, height=400, mask=char_mask, random_state=1).generate(words_string)

    fig = plt.figure(1, figsize=(12,12))
    plt.imshow(wordcloud.recolor(color_func=image_colors))
    plt.axis("off")
    plt.show()

def pickleModel(target, classifier, base_model):
    if target == 1:
        filename = 'lr_1.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        filename = 'lr_base_1.sav'
        pickle.dump(base_model, open(filename, 'wb'))

    elif target == 2:
        filename = 'lr_2.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        filename = 'lr_base_2.sav'
        pickle.dump(base_model, open(filename, 'wb'))

    elif target == 3:
        filename = 'svm_3.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        filename = 'svm_base_3.sav'
        pickle.dump(base_model, open(filename, 'wb'))

    elif target == 4:
        filename = 'rf_4.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        filename = 'rf_base_4.sav'
        pickle.dump(base_model, open(filename, 'wb'))

    elif target == 5:
        filename = 'svm_5.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        filename = 'svm_base_5.sav'
        pickle.dump(base_model, open(filename, 'wb'))

    elif target == 6:
        filename = 'svm_6.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        filename = 'svm_base_6.sav'
        pickle.dump(base_model, open(filename, 'wb'))

def logisticRegression(X_train, X_test, y_train, y_test, target):

    """
    Logistic Regression model with regularization coefficient calculated using GridSearchCV and K-Fold CV = 5.

    :param X_train: Dataframe containing train samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param X_test: Dataframe containing test samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param y_train: Dataframe containing train labels for a particular configurations specified in terminal.
    :param y_test: Dataframe containing test labels for a particular configurations specified in terminal.
    :param target: Target is an integer value denoting which configuration number.
    """

    parameters = {'C': np.logspace(-2,3,6)}
    mod_lr = LogisticRegression()

    classifier = GridSearchCV(mod_lr, parameters, cv=5)  # gridsearchCV with 5 fold CV
    classifier.fit(X_train, y_train)

    lr = LogisticRegression(penalty = 'l1', C = classifier.best_params_.get('C'))
    lr.fit(X_train, y_train)

    base_model = LogisticRegression()
    base_model.fit(X_train, y_train)

    # pickleModel(target, classifier, base_model)

def randomForest(X_train, X_test, y_train, y_test, target):

    """
    Random Forest model.

    :param X_train: Dataframe containing train samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param X_test: Dataframe containing test samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param y_train: Dataframe containing train labels for a particular configurations specified in terminal.
    :param y_test: Dataframe containing test labels for a particular configurations specified in terminal.
    :param target: Target is an integer value denoting which configuration number.
    """

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    classifier = GridSearchCV(rf, random_grid, cv=5)
    classifier.fit(X_train, y_train)


    base_model = RandomForestClassifier()
    base_model.fit(X_train, y_train)

    print(classifier.best_params_)

    print("------ TRAIN AND TEST ERRORS -------\n")
    print("TRAIN ERROR : ", 1 - accuracy_score(y_train, classifier.predict(X_train)))
    print("TEST ERROR : ", 1 - accuracy_score(y_test, classifier.predict(X_test)))
    print(" ")
    print("-------- ACCURACY SCORE -----------\n")
    print("Model Accuracy: ", accuracy_score(y_test, classifier.predict(X_test)))
    print("Base Model Accuracy: ", accuracy_score(y_test, model_base.predict(X_test)))
    print('Improvement of {:0.2f}%.'.format((accuracy_score(y_test, classifier.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("=======================================================================================")
    # classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    # classifier.fit(X_train, y_train)

    # pickleModel(target, classifier, base_model)

def linearSvm(X_train, X_test, y_train, y_test, target):

    """
    Linear SVM model with regularization coefficient calculated using GridSearchCV and K-Fold CV = 5.

    :param X_train: Dataframe containing train samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param X_test: Dataframe containing test samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param y_train: Dataframe containing train labels for a particular configurations specified in terminal.
    :param y_test: Dataframe containing test labels for a particular configurations specified in terminal.
    :param target: Target is an integer value denoting which configuration number.

    """

    parameters = {'C': np.logspace(-2,3,6)}
    classifier = GridSearchCV(LinearSVC(), parameters, cv = 5)
    classifier.fit(X_train, y_train)

    svc = LinearSVC(penalty='l2', loss='squared_hinge', dual = False, C = classifier.best_params_.get('C'))
    svc.fit(X_train, y_train)

    base_model = LinearSVC()
    base_model.fit(X_train, y_train)

    # y_pred_train = svc.predict(X_train)
    # y_pred_test = svc.predict(X_test)
    # train_score = 1 - accuracy_score(y_train, y_pred_train)  # Calculating train error
    # test_score = 1 - accuracy_score(y_test, y_pred_test)

    # pickleModel(target, svc, base_model)

def rbfSvm(X_train, X_test, y_train, y_test, target):

    """
    Radial Basis Function SVM model with C=1000.

    :param X_train: Dataframe containing train samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param X_test: Dataframe containing test samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param y_train: Dataframe containing train labels for a particular configurations specified in terminal.
    :param y_test: Dataframe containing test labels for a particular configurations specified in terminal.
    :param target: Target is an integer value denoting which configuration number.

    """
    estimator = SVC(C = 1000, kernel = 'rbf')
    estimator.fit(X_train, y_train)

    base_model = SVC(kernel = 'rbf')
    base_model.fit(X_train, y_train)

    # pickleModel(target, estimator, base_model)
