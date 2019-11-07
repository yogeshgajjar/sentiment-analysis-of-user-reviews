import numpy as np
import pandas as pd
import glob
import warnings 
warnings.filterwarnings('ignore')


def read_file(filelist):
    """
    Creates dataframe of the dataset by UCI - Sentiment Analysis 

    :param filelist: list of file directory.
    """
    
    df_uci = pd.concat([pd.read_csv(item, header=None, sep='\t') for item in filelist], axis=0)
    df_uci.columns = ['reviews', 'sentiment']
    return df_uci

def main():
    filelist = ['/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/amazon_cells_labelled.txt', '/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/imdb_labelled.txt', '/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/yelp_labelled.txt']
    print(read_file(filelist))

if __name__ == "__main__":
    main()