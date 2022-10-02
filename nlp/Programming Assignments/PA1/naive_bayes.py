# CS114B Spring 2021 Programming Assignment 1
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict
import re
import time


class Document:

    def __init__(self, name, cls):
        self.name = name
        self.cls = cls
        self.words = set()

    def contains_word(self, word):
        return word in self.words

    def add_word(self, word):
        self.words.add(word)


class NaiveBayes:

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = None
        self.feature_dict = dict()
        self.prior = None
        self.likelihood = None
        self.docs_per_class = dict()
        self.class_word_counts = dict()
        self.vocabulary = 0

    '''
    Given a directory of classes, determines the target classes, the shape of priors, the number of documents per class,
    word counts per class, and the size of the vocabulary. 
    '''

    def __collect_word_counts(self, train_set):
        vocab = set()
        for root, dirs, files in os.walk(train_set):
            # Use the directory names to populate the class labels. Maybe a bad assumption?
            if dirs:
                self.class_dict = {dirs[i]: i for i in range(len(dirs))}
                self.prior = np.zeros(len(dirs))

            # This assumes that the only directory containing any files will be the training/testing data. Otherwise,
            # this will crash.
            cls = root.split(os.sep)[-1]
            if len(files) > 0:
                self.docs_per_class[cls] = len(files)
                self.class_word_counts[cls] = dict()
            for name in files:
                with open(os.path.join(root, name)) as f:
                    word_counts = self.class_word_counts[cls]
                    # collect class counts and feature counts
                    for line in f:
                        for word in line.split():
                            new_word = re.sub('[^A-Za-z\\d]+', '', word)
                            if len(new_word):
                                word_counts.setdefault(new_word, 0)
                                word_counts[new_word] += 1
                                if new_word not in vocab:
                                    vocab.add(new_word)
                                    self.vocabulary += 1

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''

    def train(self, train_set):
        # Calling order matters in this function. self.select_features must be called after both
        # self.collect_word_counts and the calculation of the prior probabilities. This is to reduce extra
        # computations across functions.

        self.__collect_word_counts(train_set)

        # normalize counts to probabilities, and take logs
        den = sum(self.docs_per_class.values())
        for i, cls in enumerate(self.class_dict):
            num = self.docs_per_class[cls]
            self.prior[i] = np.log(num / den)

        # This must be called here
        self.feature_dict = self.select_features(train_set)
        ###

        self.likelihood = np.zeros((len(self.class_dict), len(self.feature_dict)))

        for cls, i in self.class_dict.items():
            word_counts = self.class_word_counts[cls]
            sum_class_counts = sum(word_counts.values())
            for word, j in self.feature_dict.items():
                self.likelihood[i][j] = np.log((word_counts.get(word, 0) + 1) / (self.vocabulary + sum_class_counts))

        return self.prior, self.likelihood

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''

    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            # This assumes that the only directory containing any files will be the training/testing data. Otherwise,
            # this will crash. It also assumes the class name of the documents is the directory name.
            cls = root.split(os.sep)[-1]
            for name in files:
                results[name] = dict()
                results[name]['correct'] = self.class_dict[cls]
                doc_class_preds = {key: 0 for key in self.class_dict}
                for t_cls in self.class_dict:
                    doc_class_preds[t_cls] += self.prior[self.class_dict[t_cls]]
                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    for line in f:
                        for word in line.split():
                            new_word = re.sub('[^A-Za-z\\d]+', '', word)
                            if len(new_word) > 0:
                                word_pos = self.feature_dict.get(new_word)
                                if word_pos is not None:
                                    for t_cls, i in self.class_dict.items():
                                        doc_class_preds[t_cls] += self.likelihood[i][word_pos]

                    # get most likely class
                    arr = list(doc_class_preds.values())
                    pred = np.argmax(arr)
                    results[name]['predicted'] = pred

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''

    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        correct = list()
        pred = list()
        for file_name in results:
            correct.append(results[file_name]['correct'])
            pred.append(results[file_name]['predicted'])
        for i in range(len(correct)):
            confusion_matrix[pred[i]][correct[i]] += 1

        true_pos = np.diag(confusion_matrix)
        precision = true_pos / np.sum(confusion_matrix, axis=1)
        recall = true_pos / np.sum(confusion_matrix, axis=0)

        format_string = f"Confusion Matrix: \n{confusion_matrix}\n\n"
        for i, cls in enumerate(self.class_dict):
            cls_precision = precision[i]
            cls_recall = recall[i]
            cls_f1 = 2 * ((cls_precision * cls_recall) / (cls_precision + cls_recall))
            format_string += f"Metrics for class {cls}:\n\tPrecision: {cls_precision}\n\tRecall: {cls_recall}\n\tF1: {cls_f1}\n\n"

        accuracy = np.sum(true_pos) / confusion_matrix.sum()
        format_string += f"Accuracy: {accuracy}"
        print(format_string)

    '''
    Performs feature selection.
    Returns a dictionary of features.
    '''

    def select_features(self, train_set):
        return self.__select_features_mutual_info(train_set)

    '''
    Selects all words in every document as features.
    Returns a dictionary of features.
    '''

    def __select_features_all_words(self, train_set):
        """ Results of this feature selection on movie_reviews
            Precision: 0.7722772277227723
            Recall: 0.7572815533980582
            F1: 0.7647058823529412
            Accuracy: 0.76
            |Features|: 42841
            Time taken: 1.5900003910064697 seconds"""

        # iterate over training documents
        word_index = 0
        feature_dict = dict()
        for root, dirs, files in os.walk(train_set):
            # This assumes that the only directory containing any files will be the training/testing data. Otherwise,
            # this will crash. It also assumes the class name of the documents is the directory name.
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # collect class counts and feature counts
                    for line in f:
                        for word in line.split():
                            new_word = re.sub('[^A-Za-z\\d]+', '', word)
                            if len(new_word) and new_word not in feature_dict:
                                feature_dict[new_word] = word_index
                                word_index += 1
        return feature_dict

    @staticmethod
    def __count_docs_in_cls_containing_word(all_docs, word, cls):
        num_docs_that_have_word = 0
        docs = all_docs[cls]
        for doc in docs:
            if doc.contains_word(word):
                num_docs_that_have_word += 1
        return num_docs_that_have_word / len(all_docs)

    @staticmethod
    def __count_docs_that_contain_word(all_docs, word):
        num_docs_that_have_word = 0
        for doc_list in all_docs.values():
            for doc in doc_list:
                if doc.contains_word(word):
                    num_docs_that_have_word += 1
        return num_docs_that_have_word

    '''
    Selects features by calculating the mutual information and removing words whose mutal information does not meet a 
    specific threshold.
    Returns a dictionary of features.
    '''

    def __select_features_mutual_info(self, train_set):
        """ Results of this feature selection on movie_reviews
            Precision: 0.801980198019802
            Recall: 0.8617021276595744
            F1: 0.8307692307692307
            Accuracy: 0.835
            |Features|: 4782
            Time taken: 13.178499937057495 seconds"""
        # iterate over training documents
        feature_dict = dict()
        docs = dict()
        all_words = set()
        for root, dirs, files in os.walk(train_set):
            # This assumes that the only directory containing any files will be the training/testing data. Otherwise,
            # this will crash. It also assumes the class name of the documents is the directory name.
            cls = root.split(os.sep)[-1]
            if len(files) > 0:
                docs[cls] = []
            for name in files:
                doc = Document(name, cls)
                with open(os.path.join(root, name)) as f:
                    # collect class counts and feature counts
                    for line in f:
                        for word in line.split():
                            new_word = re.sub('[^A-Za-z\\d]+', '', word)
                            if len(new_word):
                                doc.add_word(word)
                                all_words.add(word)
                docs[cls].append(doc)

        # Calculate mutual info of words
        mutual_info = dict()
        sum = 0
        for word in all_words:
            term_1 = 0
            term_2 = 0
            term_3 = 0
            for cls in self.class_dict:
                prior = np.exp(self.prior[self.class_dict[cls]])
                log_prior = self.prior[self.class_dict[cls]]
                term_1 += prior * log_prior
                val = self.__count_docs_in_cls_containing_word(docs, word, cls)
                term_2 += val * prior
                term_3 += (1 - val) * prior

            num_docs_containing_word = self.__count_docs_that_contain_word(docs, word)
            result = (-1 * term_1) + \
                     num_docs_containing_word * term_2 + \
                     (1 - num_docs_containing_word) * term_3
            mutual_info[word] = result
            sum += result

        # Select the best words given a required normalized information value
        vals = list(mutual_info.values())
        xmin = min(vals)
        xmax = max(vals)
        for i, x in enumerate(vals):
            vals[i] = (x - xmin) / (xmax - xmin)

        index = 0
        for i, word in enumerate(mutual_info):
            # Accuracies: 0.05 == 79.5%, 0.0002 == 83%, 0.00008 == 83.5%
            if vals[i] > 0.00008:
                feature_dict[word] = index
                index += 1

        return feature_dict


if __name__ == '__main__':
    start = time.time()
    nb = NaiveBayes()
    # make sure these point to the right directories
    # nb.train('movie_reviews_small/train')
    nb.train('movie_reviews/train')
    # results = nb.test('movie_reviews_small/test')
    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)
    total = time.time() - start
    print(f"\nTime taken: {total} seconds")
