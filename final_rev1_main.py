from final_rev1 import uciData, imdbData, datasetSelection, logisticRegression,randomForest, linearSvm, rbfSvm
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

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
    # print("Accuracy is :", model.score(X_test,y_test))
    model_base = pickle.load(open(filename2, 'rb'))
    # print("Base Accuracy is :",model_base.score(X_test, y_test))
    # print('Improvement of {:0.2f}%.'.format((model.score(X_test, y_test)- model_base.score(X_test, y_test)) / model_base.score(X_test, y_test)))
    # print("=========================================\n\n")
    print("------ TRAIN AND TEST ERRORS -------\n")
    print("TRAIN ERROR : ", 1 - accuracy_score(y_train, model.predict(X_train)))
    print("TEST ERROR : ", 1 - accuracy_score(y_test, model.predict(X_test)))
    print(" ")
    print("-------- CONFUSION MATRIX ----------\n") 
    print(confusion_matrix(y_test,model.predict(X_test)))
    print("------ CLASSIFICATION REPORT -------\n") 
    print(classification_report(y_test,model.predict(X_test)))
    print("-------- ACCURACY SCORE -----------\n") 
    print("Model Accuracy: ", accuracy_score(y_test, model.predict(X_test)))
    print("Base Model Accuracy: ", accuracy_score(y_test, model_base.predict(X_test)))
    print('Improvement of {:0.2f}%.'.format((accuracy_score(y_test, model.predict(X_test)) - accuracy_score(y_test, model_base.predict(X_test))) / accuracy_score(y_test, model_base.predict(X_test))))
    print("=======================================================================================")



def main():
    df_uci, df_imdb_train, df_imdb_test = uciData(), imdbData()[0], imdbData()[1]

    if sys.argv[1] == '1':
        X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,1)
        # logisticRegression(X_train, X_test, y_train, y_test, 1)
        print("-------------------- LOGISTIC REGRESSION ------------------------\n") 
        pickleModel('lr_1.sav', 'lr_base_1.sav', X_test, y_test, X_train, y_train)
        # randomForest(X_train, X_test, y_train, y_test, 1)
        print("-------------------- RANDOM FOREST ------------------------\n") 
        pickleModel('rf_1.sav', 'rf_base_1.sav', X_test, y_test, X_train, y_train)
        # linearSvm(X_train, X_test, y_train, y_test, 1)
        print("-------------------- LINEAR SVM ------------------------\n") 
        pickleModel('svm_1.sav', 'svm_base_1.sav', X_test, y_test, X_train, y_train)
        # rbfSvm(X_train, X_test, y_train, y_test, 1)
        print("-------------------- RBF SVM ------------------------\n") 
        pickleModel('rbf_1.sav', 'rbf_base_1.sav', X_test, y_test, X_train, y_train)

    if sys.argv[1] == '2':
        X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,2)
       
         # logisticRegression(X_train, X_test, y_train, y_test, 2)
        print("-------------------- LOGISTIC REGRESSION ------------------------\n") 
        pickleModel('lr_2.sav', 'lr_base_2.sav', X_test, y_test, X_train, y_train)
        # randomForest(X_train, X_test, y_train, y_test, 2)
        print("-------------------- RANDOM FOREST ------------------------\n") 
        pickleModel('rf_2.sav', 'rf_base_2.sav', X_test, y_test, X_train, y_train)
        # linearSvm(X_train, X_test, y_train, y_test, 2)
        print("-------------------- LINEAR SVM ------------------------\n") 
        pickleModel('svm_2.sav', 'svm_base_2.sav', X_test, y_test, X_train, y_train)
        # rbfSvm(X_train, X_test, y_train, y_test, 2)
        print("-------------------- RBF SVM ------------------------\n") 
        pickleModel('rbf_2.sav', 'rbf_base_2.sav', X_test, y_test, X_train, y_train)

    if sys.argv[1] == '3':
        X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,3)
         # logisticRegression(X_train, X_test, y_train, y_test, 3)
        print("-------------------- LOGISTIC REGRESSION ------------------------\n") 
        pickleModel('lr_3.sav', 'lr_base_3.sav', X_test, y_test, X_train, y_train)
        # randomForest(X_train, X_test, y_train, y_test, 3)
        print("-------------------- RANDOM FOREST ------------------------\n") 
        pickleModel('rf_3.sav', 'rf_base_3.sav', X_test, y_test, X_train, y_train)
        # linearSvm(X_train, X_test, y_train, y_test, 3)
        print("-------------------- LINEAR SVM ------------------------\n") 
        pickleModel('svm_3.sav', 'svm_base_3.sav', X_test, y_test, X_train, y_train)
        # rbfSvm(X_train, X_test, y_train, y_test, 3)
        print("-------------------- RBF SVM ------------------------\n") 
        pickleModel('rbf_3.sav', 'rbf_base_3.sav', X_test, y_test, X_train, y_train)

    if sys.argv[1] == '4':
        X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,4)
         # logisticRegression(X_train, X_test, y_train, y_test, 4)
        print("-------------------- LOGISTIC REGRESSION ------------------------\n") 
        pickleModel('lr_4.sav', 'lr_base_4.sav', X_test, y_test, X_train, y_train)
        # randomForest(X_train, X_test, y_train, y_test, 4)
        print("-------------------- RANDOM FOREST ------------------------\n") 
        pickleModel('rf_4.sav', 'rf_base_4.sav', X_test, y_test, X_train, y_train)
        # linearSvm(X_train, X_test, y_train, y_test, 4)
        print("-------------------- LINEAR SVM ------------------------\n") 
        pickleModel('svm_4.sav', 'svm_base_4.sav', X_test, y_test, X_train, y_train)
        # rbfSvm(X_train, X_test, y_train, y_test, 4)
        print("-------------------- RBF SVM ------------------------\n") 
        pickleModel('rbf_4.sav', 'rbf_base_4.sav', X_test, y_test, X_train, y_train)

    if sys.argv[1] == '5':
        X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,5)
         # logisticRegression(X_train, X_test, y_train, y_test, 5)
        print("-------------------- LOGISTIC REGRESSION ------------------------\n") 
        pickleModel('lr_5.sav', 'lr_base_5.sav', X_test, y_test, X_train, y_train)
        # randomForest(X_train, X_test, y_train, y_test, 5)
        print("-------------------- RANDOM FOREST ------------------------\n") 
        pickleModel('rf_5.sav', 'rf_base_5.sav', X_test, y_test, X_train, y_train)
        # linearSvm(X_train, X_test, y_train, y_test, 5)
        print("-------------------- LINEAR SVM ------------------------\n") 
        pickleModel('svm_5.sav', 'svm_base_5.sav', X_test, y_test, X_train, y_train)
        # rbfSvm(X_train, X_test, y_train, y_test, 5)
        print("-------------------- RBF SVM ------------------------\n") 
        pickleModel('rbf_5.sav', 'rbf_base_5.sav', X_test, y_test, X_train, y_train)

    if sys.argv[1] == '6':
        X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test,6)
         # logisticRegression(X_train, X_test, y_train, y_test, 6)
        print("-------------------- LOGISTIC REGRESSION ------------------------\n") 
        pickleModel('lr_6.sav', 'lr_base_6.sav', X_test, y_test, X_train, y_train)
        # randomForest(X_train, X_test, y_train, y_test, 6)
        print("-------------------- RANDOM FOREST ------------------------\n") 
        pickleModel('rf_6.sav', 'rf_base_6.sav', X_test, y_test, X_train, y_train)
        # linearSvm(X_train, X_test, y_train, y_test, 6)
        print("-------------------- LINEAR SVM ------------------------\n") 
        pickleModel('svm_6.sav', 'svm_base_6.sav', X_test, y_test, X_train, y_train)
        # rbfSvm(X_train, X_test, y_train, y_test, 6)
        print("-------------------- RBF SVM ------------------------\n") 
        pickleModel('rbf_6.sav', 'rbf_base_6.sav', X_test, y_test, X_train, y_train)

if __name__ == "__main__":
        main()


# logistic_regression = pickle.load(open('lr_2.sav', 'rb'))

        # print("Accuracy is :",logistic_regression.score(X_test, y_test))
        # logistic_regression_base = pickle.load(open('lr_base_2.sav', 'rb'))
        # print("Base Accuracy is :",logistic_regression_base.score(X_test, y_test))
        # print('Improvement of {:0.2f}%.'.format(( logistic_regression.score(X_test, y_test)- logistic_regression_base.score(X_test, y_test)) / logistic_regression_base.score(X_test, y_test)))
        # print("=========================================\n\n")
        # # print("=======================================================================================")
        # # print("-------------------- LOGISTIC REGRESSION ------------------------\n") 
        # # print("------ TRAIN AND TEST ERRORS -------\n")
        # # print("TRAIN ERROR : ", accuracy_score(y_train, logistic_regression.predict(X_train)))
        # # print("TEST ERROR : ", accuracy_score(y_test, logistic_regression.predict(X_test)))
        # # print(" ")
        # # print("-------- CONFUSION MATRIX ----------\n") 
        # # print(confusion_matrix(y_test,logistic_regression.predict(X_test)))
        # # print("------ CLASSIFICATION REPORT -------\n") 
        # # print(classification_report(y_test,logistic_regression.predict(X_test)))
        # # print("-------- ACCURACY SCORE -----------\n") 
        # # print(accuracy_score(y_test, logistic_regression.predict(X_test)))
        # # print(" ")
        # # print("-------- BASE MODEL ACCURACY--------\n")
        # # print("Base Model Accuracy: ", accuracy_score(y_test, logistic_regression_base.predict(X_test)))
        # # print('Improvement of {:0.2f}%.'.format((accuracy_score(y_test, logistic_regression.predict(X_test)) - accuracy_score(y_test, logistic_regression_base.predict(X_test))) / accuracy_score(y_test, logistic_regression_base.predict(X_test))))
        # # print("=======================================================================================")

        # # randomForest(X_train, X_test, y_train, y_test, 2)
        # random_forest = pickle.load(open('rf_2.sav', 'rb'))
        # print("Accuracy is :",random_forest.score(X_test, y_test))
        # random_forest_base = pickle.load(open('rf_base_2.sav', 'rb'))
        # print("Base Accuracy is :",random_forest_base.score(X_test, y_test))
        # print('Improvement of {:0.2f}%.'.format(( random_forest.score(X_test, y_test)- random_forest_base.score(X_test, y_test)) / random_forest_base.score(X_test, y_test)))
        # print("=========================================\n\n")

        # # linearSvm(X_train, X_test, y_train, y_test, 2)
        # linear_svm = pickle.load(open('svm_2.sav', 'rb'))
        # print("Accuracy is :",linear_svm.score(X_test, y_test))
        # linear_svm_base = pickle.load(open('svm_base_2.sav', 'rb'))
        # print("Base Accuracy is :",linear_svm_base.score(X_test, y_test))
        # print('Improvement of {:0.2f}%.'.format(( linear_svm.score(X_test, y_test)- linear_svm_base.score(X_test, y_test)) / linear_svm_base.score(X_test, y_test)))
        # print("=========================================\n\n")

        # # rbfSvm(X_train, X_test, y_train, y_test, 2)
        # rbf_svm = pickle.load(open('rbf_2.sav', 'rb'))
        # print("Accuracy is :",rbf_svm.score(X_test, y_test))
        # rbf_svm_base = pickle.load(open('rbf_base_2.sav', 'rb'))
        # print("Base Accuracy is :",rbf_svm_base.score(X_test, y_test))
        # print('Improvement of {:0.2f}%.'.format(( rbf_svm.score(X_test, y_test)- rbf_svm_base.score(X_test, y_test)) / rbf_svm_base.score(X_test, y_test)))
        # print("=========================================\n\n")