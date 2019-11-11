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

warnings.filterwarnings('ignore')


def read_file(filelist):
    """
    Creates dataframe of the dataset by UCI - Sentiment Analysis 

    :param filelist: list of file directory.
    """
    
    df_uci = pd.concat([pd.read_csv(item, header=None, sep='\t') for item in filelist], axis=0)
    df_uci.columns = ['reviews', 'sentiment']
    return df_uci

def preprocessing(df_uci):
    """
    Contains preprocessing methods such as Tokenization and Vectorization

    :param df_uci: Dataframe containing dataset 
    """

    reviews = df_uci.reviews.str.cat(sep=' ')
    tokens = word_tokenize(reviews) 
    stop_words = set(stopwords.words('english'))
    tokens_stop_words = [token for token in tokens if not token in stop_words]
    words = [word for word in tokens_stop_words if word.isalpha()]

    # fdist = FreqDist()
    # for word in words:
    #     fdist[word.lower()] += 1
    return words

def wordcloud(words):
    words_string = TreebankWordDetokenizer().detokenize(words)
    char_mask = np.array(Image.open("/home/yogesh/Git/sentiment_analysis/image.jpg"))    
    image_colors = ImageColorGenerator(char_mask)
    wordcloud = WordCloud(background_color="black", max_words=200, width=400, height=400, mask=char_mask, random_state=1).generate(words_string)

    fig = plt.figure(1, figsize=(12,12))
    plt.imshow(wordcloud.recolor(color_func=image_colors))
    plt.axis("off")
    plt.show()

    # frequency_dist = nltk.FreqDist(words)
    # wordcloud = WordCloud(background_color='black', max_words=200, width=400, height=400).generate_from_frequencies(frequency_dist)
    # plt.imshow(wordcloud)

    # plt.axis("off")
    # plt.show()
    





def main():
    filelist = ['/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/amazon_cells_labelled.txt', '/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/imdb_labelled.txt', '/home/yogesh/fall19/ml660/project/sentiment_labelled_sentences/yelp_labelled.txt']
    # print(read_file(filelist))
    file = read_file(filelist)
    words = preprocessing(file)
    wordcloud(words)

if __name__ == "__main__":
    main()