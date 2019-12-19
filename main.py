from module import uciData, imdbData, datasetSelection, logisticRegression,randomForest, linearSvm, rbfSvm
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

def pickleModel(filename1, filename2, X_test, y_test, X_train,y_train):

    """
    Loads the saved model and prints the required results

    :param filename1: The filename for the model. It is a .sav file
    :param filename2: The filename for the base model. It is a .sav file
    :param X_test: Dataframe containing test samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param y_test: Dataframe containing test labels for a particular configurations specified in terminal.
    :param X_train: Dataframe containing train samples with features extracted from TfidfVectorizer for a particular configurations specified in terminal.
    :param y_train: Dataframe containing train labels for a particular configurations specified in terminal.
    """

    model = pickle.load(open(filename1, 'rb'))
    model_base = pickle.load(open(filename2, 'rb'))

    print("------ TRAIN AND TEST ERRORS -------\n")
    print("TRAIN ERROR : ", 1 - accuracy_score(y_train, model.predict(X_train)))
    print("TEST ERROR : ", 1 - accuracy_score(y_test, model.predict(X_test)))
    print(" ")
    print("-------- ACCURACY SCORE -----------\n")
    print("Model Accuracy: ", accuracy_score(y_test, model.predict(X_test)))
    print("Base Model Accuracy: ", accuracy_score(y_test, model_base.predict(X_test)))
    print('Improvement of {:0.2f}%.'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("=======================================================================================")
    return model, model_base

def main():
    df_uci, df_imdb_train, df_imdb_test = uciData(), imdbData()[0], imdbData()[1]
    accuracy, error, base_accuracy, improvement = ([] for i in range(4))
    config = pd.Series(["Configuration 1", "Configuration 2", "Configuration 3", "Configuration 4", "Configuration 5", "Configuration 6"])

    X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,1)
    # logisticRegression(X_train, X_test, y_train, y_test, 1)
    print("-------------------- LOGISTIC REGRESSION ------------------------\n")
    model, model_base = pickleModel('lr_1.sav', 'lr_base_1.sav', X_test, y_test, X_train, y_train)
    accuracy.append(100 * accuracy_score(y_test, model.predict(X_test)))
    error.append(1 - accuracy_score(y_test, model.predict(X_test)))
    base_accuracy.append(100 * accuracy_score(y_test, model_base.predict(X_test)))
    improvement.append('{:0.2f}'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("\n")

    X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,2)
    # logisticRegression(X_train, X_test, y_train, y_test, 2)

    print("-------------------- LOGISTIC REGRESSION ------------------------\n")
    model, model_base = pickleModel('lr_2.sav', 'lr_base_2.sav', X_test, y_test, X_train, y_train)
    accuracy.append(100 * accuracy_score(y_test, model.predict(X_test)))
    error.append(1 - accuracy_score(y_test, model.predict(X_test)))
    base_accuracy.append(100 * accuracy_score(y_test, model_base.predict(X_test)))
    improvement.append('{:0.2f}'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("\n")

    X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,3)
    # linearSvm(X_train, X_test, y_train, y_test, 3)
    # logisticRegression(X_train, X_test, y_train, y_test, 3)
    print("-------------------- LINEAR SVM ------------------------\n")
    model, model_base = pickleModel('svm_3.sav', 'svm_base_3.sav', X_test, y_test, X_train, y_train)
    accuracy.append(100 * accuracy_score(y_test, model.predict(X_test)))
    error.append(1 - accuracy_score(y_test, model.predict(X_test)))
    base_accuracy.append(100 * accuracy_score(y_test, model_base.predict(X_test)))
    improvement.append('{:0.2f}'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("\n")

    X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,4)
    # randomForest(X_train, X_test, y_train, y_test, 4)
    print("-------------------- RANDOM FOREST ------------------------\n")
    model, model_base = pickleModel('rf_4.sav', 'rf_base_4.sav', X_test, y_test, X_train, y_train)
    accuracy.append(100 * accuracy_score(y_test, model.predict(X_test)))
    error.append(1 - accuracy_score(y_test, model.predict(X_test)))
    base_accuracy.append(100 * accuracy_score(y_test, model_base.predict(X_test)))
    improvement.append('{:0.2f}'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("\n")

    X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,5)
    # linearSvm(X_train, X_test, y_train, y_test, 5)
    print("-------------------- LINEAR SVM ------------------------\n")
    model, model_base = pickleModel('svm_5.sav', 'svm_base_5.sav', X_test, y_test, X_train, y_train)
    accuracy.append(100* accuracy_score(y_test, model.predict(X_test)))
    error.append(1 - accuracy_score(y_test, model.predict(X_test)))
    base_accuracy.append(100 * accuracy_score(y_test, model_base.predict(X_test)))
    improvement.append('{:0.2f}'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("\n")

    X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,6)
    # linearSvm(X_train, X_test, y_train, y_test, 6)
    print("-------------------- RANDOM FOREST ------------------------\n")
    model, model_base = pickleModel('rf_6.sav', 'rf_base_6.sav', X_test, y_test, X_train, y_train)
    accuracy.append(100 * accuracy_score(y_test, model.predict(X_test)))
    error.append(1 - accuracy_score(y_test, model.predict(X_test)))
    base_accuracy.append(100 * accuracy_score(y_test, model_base.predict(X_test)))
    improvement.append('{:0.2f}'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("\n")

    df = pd.DataFrame({"Config":config, "Base Model F-1 score":base_accuracy, "Best F-1 score":accuracy, "Best Out-Sample Error":error, "Improvement %":improvement})
    print("-------------------- FINAL MODEL COMPARISION AND RESULT ------------------------\n")
    print(df)

if __name__ == "__main__":
        main()
