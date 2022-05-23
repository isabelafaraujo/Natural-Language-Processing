import spacy
from gensim.models import KeyedVectors

def most_similar(phrase, topn):

    #dictionaries (CBOW)
    fastText50 = KeyedVectors.load_word2vec_format('vocab/fastText-s50.txt', binary=False)
    fastText100 = KeyedVectors.load_word2vec_format('vocab/fastText-s100.txt', binary=False)
    glove50 = KeyedVectors.load_word2vec_format('vocab/glove_s50.txt', binary=False)
    glove100 = KeyedVectors.load_word2vec_format('vocab/glove_s100.txt', binary=False)
    wang50 = KeyedVectors.load_word2vec_format('vocab/wang2Vec-s50.txt', binary=False)
    wang100 = KeyedVectors.load_word2vec_format('vocab/wang2Vec-s100.txt', binary=False)
    word50 = KeyedVectors.load_word2vec_format('vocab/word2Vec-s50.txt', binary=False)
    word100 = KeyedVectors.load_word2vec_format('vocab/word2Vec-s100.txt', binary=False)
    
    nlp = spacy.load("pt_core_news_lg")
    tokens = nlp(phrase)

    print('fastText50')
    print(fastText50.most_similar(phrase)[:topn])

    print('fastText100')
    print(fastText100.most_similar(phrase)[:topn])

    print('glove50')
    print(glove50.most_similar(phrase)[:topn])

    print('glove100')
    print(glove100.most_similar(phrase)[:topn])

    print('wang50')
    print(wang50.most_similar(phrase)[:topn])

    print('wang100')
    print(wang100.most_similar(phrase)[:topn])

    print('word50')
    print(word50.most_similar(phrase)[:topn])

    print('word100')
    print(word100.most_similar(phrase)[:topn])
    
word = input("Search a synonym for the word: ")
quantity = input("How many synonyms do you search for?")

most_similar(word, quantity)
