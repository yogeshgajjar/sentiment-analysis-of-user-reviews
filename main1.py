from sentiment_analysis_rev3 import uciData, imdbData, datasetSelection, logisticRegression,randomForest, linearSvm, rbfSvm
import pickle

# def pickleModel(filename1, filename2, X_test, y_test):
#     model = pickle.load(open(filename1, 'rb'))
#     print("Accuracy is :", model.score(X_test,y_test))
#     model_base = pickle.load(open(filename2, 'rb'))
#     print("Base Accuracy is :",model_base.score(X_test, y_test))
#     print('Improvement of {:0.2f}%.'.format((model.score(X_test, y_test)- model_base.score(X_test, y_test)) / model_base.score(X_test, y_test)))
#     print("=========================================\n\n")


def main():
    df_uci, df_imdb_train, df_imdb_test = uciData(), imdbData()[0], imdbData()[1]
    X_train, X_test, y_train, y_test = datasetSelection(df_uci, df_imdb_train, df_imdb_test)

    logisticRegression(X_train, X_test, y_train, y_test)
    # pickleModel('logistic_regression.sav', 'logistic_regression_base.sav', X_test, y_test)

    # logistic_regression = pickle.load(open('logistic_regression.sav', 'rb'))
    # print("Accuracy is :",logistic_regression.score(X_test, y_test))
    # logistic_regression_base = pickle.load(open('logistic_regression_base.sav', 'rb'))
    # print("Base Accuracy is :",logistic_regression_base.score(X_test, y_test))
    # print('Improvement of {:0.2f}%.'.format(( logistic_regression.score(X_test, y_test)- logistic_regression_base.score(X_test, y_test)) / logistic_regression_base.score(X_test, y_test)))
    # print("=========================================\n\n")

    randomForest(X_train, X_test, y_train, y_test)
    # pickleModel('random_forest.sav', 'random_forest_base.sav', X_test, y_test)

    # random_forest = pickle.load(open('random_forest.sav', 'rb'))
    # print("Accuracy is :",random_forest.score(X_test, y_test))
    # random_forest_base = pickle.load(open('random_forest_base.sav', 'rb'))
    # print("Base Accuracy is :",random_forest_base.score(X_test, y_test))
    # print('Improvement of {:0.2f}%.'.format((random_forest.score(X_test, y_test)- random_forest_base.score(X_test, y_test)) / random_forest_base.score(X_test, y_test)))
    # print("=========================================\n\n")

    linearSvm(X_train, X_test, y_train, y_test)
    # pickleModel('linear_svm.sav', 'linear_svm_base.sav', X_test, y_test)

    # linear_svm = pickle.load(open('linear_svm.sav', 'rb'))
    # print("Accuracy is :",linear_svm.score(X_test, y_test))
    # linear_svm_base = pickle.load(open('linear_svm_base.sav', 'rb'))
    # print("Base Accuracy is :",linear_svm_base.score(X_test, y_test))
    # print('Improvement of {:0.2f}%.'.format((linear_svm.score(X_test, y_test)- linear_svm_base.score(X_test, y_test)) / linear_svm_base.score(X_test, y_test)))
    # print("=========================================\n\n")

    rbfSvm(X_train, X_test, y_train, y_test)
    # pickleModel('rbf_svm.sav', 'rbf_svm_base.sav', X_test, y_test)

    # rbf_svm = pickle.load(open('rbf_svm.sav', 'rb'))
    # print("Accuracy is :",rbf_svm.score(X_test, y_test))
    # rbf_svm_base = pickle.load(open('rbf_svm_base.sav', 'rb'))
    # print("Base Accuracy is :",rbf_svm_base.score(X_test, y_test))
    # print('Improvement of {:0.2f}%.'.format((rbf_svm.score(X_test, y_test)- rbf_svm_base.score(X_test, y_test)) / rbf_svm_base.score(X_test, y_test)))
    # print("=========================================\n\n")
    # PIK = "pickle.dat"
    # data = ["A", "b", "C", "d"]
    # with open(PIK, "wb") as f:
    #     pickle.dump(data, f)

    # with open(PIK, "rb") as f:
    #     pickle.load(f)
    #     print("Accuracy is :",f.score(X_test, y_test))

if __name__ == "__main__":
    main()