import os
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

"""
Implement the function to check if all the texts are different.
"""

def preprocess_data(path):

    unigrams_train = {}
    bigrams_train = {}
    unigrams_test = {}
    bigrams_test = {}

    overall_unigrams_train = {}
    overall_bigrams_train = {}
    overall_unigrams_test = {}
    overall_bigrams_test = {}

    test = False
    counter = 0

    for root, dirs, files in os.walk(path):

        for file in files:

            if file.endswith(".txt") and file != "trainingLabels.txt" and file != "testLabels.txt":

                filepath = root + '/' + file
                local_unigrams_test = {}
                local_bigrams_test = {}
                local_unigrams_train = {}
                local_bigrams_train = {}

                if root.endswith("fold5"):
                    test = True
                    local_unigrams_test = get_unigrams(filepath)
                    local_bigrams_test = get_bigrams(filepath)
                    unigrams_test[file] = local_unigrams_test
                    bigrams_test[file] = local_bigrams_test

                else:
                    local_unigrams_train = get_unigrams(filepath)
                    local_bigrams_train = get_bigrams(filepath)
                    unigrams_train[file] = local_unigrams_train
                    bigrams_train[file] = local_bigrams_train

                # updating the dictionaries with the unigrams and bigrams of the new processed file
                if test:
                    for key in local_unigrams_test:
                        if key not in overall_unigrams_test:
                            overall_unigrams_test[key] = local_unigrams_test[key]
                        else:
                            overall_unigrams_test[key] += local_unigrams_test[key]

                    for key in local_bigrams_test:
                        if key not in overall_bigrams_test:
                            overall_bigrams_test[key] = local_bigrams_test[key]
                        else:
                            overall_bigrams_test[key] += local_bigrams_test[key]

                else:
                    for key in local_unigrams_train:
                        if key not in overall_unigrams_train:
                            overall_unigrams_train[key] = local_unigrams_train[key]
                        else:
                            overall_unigrams_train[key] += local_unigrams_train[key]

                    for key in local_bigrams_train:
                        if key not in overall_bigrams_train:
                            overall_bigrams_train[key] = local_bigrams_train[key]
                        else:
                            overall_bigrams_train[key] += local_bigrams_train[key]

                test = False
                counter += 1
                print("Processed files:", counter, "/ 800")

    return overall_unigrams_train, overall_unigrams_test, overall_bigrams_train, overall_bigrams_test, unigrams_train, bigrams_train, unigrams_test, bigrams_test


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

