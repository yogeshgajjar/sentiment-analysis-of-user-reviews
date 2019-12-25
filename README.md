# sentiment_analysis
Sentiment Analysis of User Reviews

## Abstract 

This project aims to solve a binary classification problem that predicts the sentiment of user reviews into either positive(1) or negative(0) using the best model out of four different models. The data consists of
53,000 reviews from various sites such as IMDB, Yelp and Amazon and is uniformly distributed into two classes i.e. positive(1) and negative(0). The four different models considered are Logistic Regression, Random Forest, Linear SVM and Complex Kernel Gaussian RBF SVM. The project also compares the performance of all the ML models with six different dataset configurations and finds the best model in terms of out-sample error and F-1 score. The results from this project gives a understanding of how differently two dataset can be used and how the results get affected if the source domain and target domain gets mismatched. 

Dataset A = Sentiment Labelled Sentences Dataset by UCI [UCI] (https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

Dataset B = Large Movie Review Dataset by Stanford [Stanford] (https://ai.stanford.edu/~amaas/data/sentiment/)

The configurations are: 
1. Train and Test on overall dataset which includes A + B
2. Train and Test on A
3. Train on B and Test on A
4. Train and Test in B
5. Train on 100% B + 80% A and Test on 20% A
6. Train on 100% A + 80% B and Test on 20% B 