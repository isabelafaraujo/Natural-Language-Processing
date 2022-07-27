from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances


def cosineSimilarity(mainWord, listWord):
    mainList = listWord
    mainList = [mainWord] + mainList

    vectorizer = CountVectorizer()
    vector_list = vectorizer.fit_transform(mainList).todense()

    # to find an specific word
    # print(vectorizer.vocabulary_.get('objecto'))

    for distance in vector_list:
        print(distance, euclidean_distances(vector_list[1], distance))

    return vectorizer.vocabulary_

