import spacy
from spacy.tokens import Doc

def englishFunction(phrase):

    nlp = spacy.load("en_core_web_lg")
    doc = nlp(phrase)

    for token in doc:
        print("Token: ", token.text,
              ", Lemma: ", token.lemma_,
              ", Stopword: ", token.is_stop)

def portugueseFunction(frase):

    nlp = spacy.load("pt_core_news_lg")
    doc = nlp(frase)

    for token in doc:
        print("Token: ", token.text,
              ", Lemma: ", token.lemma_,
              ", Stopword: ", token.is_stop)
