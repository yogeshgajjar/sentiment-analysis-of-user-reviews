# sentiment-analysis
Sentiment Analysis of User Reviews from Yelp, Amazon and IMDB websites 

## Abstract 

This project aims to solve a binary classification problem that predicts the sentiment of user reviews into either positive(1) or negative(0) using the best model out of four different models. The data consists of
53,000 reviews from various sites such as IMDB, Yelp and Amazon and is uniformly distributed into two classes i.e. positive(1) and negative(0). The four different models considered are Logistic Regression, Random Forest, Linear SVM and Complex Kernel Gaussian RBF SVM. The project also compares the performance of all the ML models with six different dataset configurations and finds the best model in terms of out-sample error and F-1 score. The results from this project gives a understanding of how differently two dataset can be used and how the results get affected if the source domain and target domain gets mismatched. 

Dataset A = Sentiment Labelled Sentences Dataset by [UCI](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

Dataset B = Large Movie Review Dataset by [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/)

The configurations are: 
1. Train and Test on overall dataset which includes A + B
2. Train and Test on A
3. Train on B and Test on A
4. Train and Test in B
5. Train on 100% B + 80% A and Test on 20% A
6. Train on 100% A + 80% B and Test on 20% B 

## Instructions to run the code 

1. Download the pre-trained models 
``` 
sh models.sh 
```

2. Execute the `main.py` file 
```
python3 main.py "path_of_dataset" 
For example, 
python3 main.py /home/tars/project/data/
``` 

## Results 

| Configurations     | Model | Base Model F-1 Score | Best Model F-1 Score  | Best Out-Sample Error | Improvement in Model Performance | 
| ------------- |:-------------:| -----:|-----:|-----:|-----:|
| Config 1     | Logistic Regression | 93.79%| 99.28% | 0.0071 | 0.06% |
| Config 2    | Logistic Regression |  90.00% | 95.93% | 0.0416 | 0.06% |
| Config 3     | Linear SVM | 49.76% | 49.80% | 0.5020 | 0.001% |
| Config 4 | Random Forest | 99.27% | 100.00% | 0.000 | 0.01% |
| Config 5 | Linear SVM | 51.58% | 51.58% | 0.4842 | 0.00% | 
| Config 6 | Random Forest | 71.18% | 77.98% | 0.2202 | 0.10% |

