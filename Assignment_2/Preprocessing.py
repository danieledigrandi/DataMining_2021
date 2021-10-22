import os
from Assignment_2.utils import notImplemented
#import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

"""
Implement the function to check if all the text are different.
"""

def preprocess_data(path):

    unigrams_train = {}
    bigrams_train = {}
    unigrams_test = {}
    bigrams_test = {}

    test = False
    counter = 0

    for root, dirs, files in os.walk(path):

        for file in files:

            if file.endswith(".txt"):

                filepath = root + '/' + file
                local_unigrams_test = {}
                local_bigrams_test = {}
                local_unigrams_train = {}
                local_bigrams_train = {}

                if root.endswith("fold5"):
                    test = True
                    local_unigrams_test = get_unigrams(filepath)
                    local_bigrams_test = get_bigrams(filepath)

                else:
                    local_unigrams_train = get_unigrams(filepath)
                    local_bigrams_train = get_bigrams(filepath)

                # updating the dictionaries with the unigrams and bigrams of the new processed file
                if test:
                    for key in local_unigrams_test:
                        if key not in unigrams_test:
                            unigrams_test[key] = local_unigrams_test[key]
                        else:
                            unigrams_test[key] += local_unigrams_test[key]

                    for key in local_bigrams_test:
                        if key not in bigrams_test:
                            bigrams_test[key] = local_bigrams_test[key]
                        else:
                            bigrams_test[key] += local_bigrams_test[key]

                else:
                    for key in local_unigrams_train:
                        if key not in unigrams_train:
                            unigrams_train[key] = local_unigrams_train[key]
                        else:
                            unigrams_train[key] += local_unigrams_train[key]

                    for key in local_bigrams_train:
                        if key not in bigrams_train:
                            bigrams_train[key] = local_bigrams_train[key]
                        else:
                            bigrams_train[key] += local_bigrams_train[key]

                test = False
                counter += 1
                print("Processed files:", counter, "/ 800")

    return unigrams_train, bigrams_train, unigrams_test, bigrams_test


def preprocess_text(text):

    # nltk.download('stopwords')
    stop_words = stopwords.words('english')

    cleaned_text = []

    for line in text:
        all_words = word_tokenize(line)
        # removing punctuation
        cleaned_line = [word for word in all_words if word.isalpha()]

        # removing stopwords
        cleaned_line = [word for word in cleaned_line if word not in stop_words]
        cleaned_text.append(cleaned_line)

    stemmer = PorterStemmer()
    stemmed_text = []

    for line in cleaned_text:
        stemmed = [stemmer.stem(word) for word in line]
        stemmed_text.append(stemmed)

    return stemmed_text


def get_unigrams(filepath):

    unigrams = {}

    with open(filepath, 'r') as f:

        lines = f.readlines()
        text = [line.rstrip().lower() for line in lines if line != '\n']

    stemmed_text = preprocess_text(text)

    for line in stemmed_text:
        for word in line:
            if word not in unigrams:
                unigrams[word] = 1
            else:
                unigrams[word] += 1

    return unigrams


def get_bigrams(filepath):

    bigrams = {}

    with open(filepath, 'r') as f:

        lines = f.readlines()
        text = [line.rstrip().lower() for line in lines if line != '\n']

    stemmed_text = preprocess_text(text)

    for line in stemmed_text:
        for i in range(len(line)-1):
            word = (line[i], line[i + 1])
            if word not in bigrams:
                bigrams[word] = 1
            else:
                bigrams[word] += 1

    return bigrams


def eliminate_sparse_words(unigrams_train, bigrams_train, unigram_threshold, bigram_threshold):

    new_unigrams_train = unigrams_train.copy()
    new_bigrams_train = bigrams_train.copy()

    # eliminate the sparse words only in the training sample
    for word, count in unigrams_train.items():
        if count < unigram_threshold:
            del new_unigrams_train[word]

    for words, count in bigrams_train.items():
        if count < bigram_threshold:
            del new_bigrams_train[words]

    return new_unigrams_train, new_bigrams_train
