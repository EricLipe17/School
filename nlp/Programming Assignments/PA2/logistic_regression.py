# CS542 Fall 2021 Programming Assignment 2
# Logistic Regression Classifier

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random
import re

dont_use = {'a', 'as', 'am', 'the', 'to', 'of', 'and', 'it', 'its', 'too', 'is', 'for', 'they', 'them', 'and',
            'but', 'or', 'by', 'who', 'in', 'if', 'this', 'get', 'can', 'that', 'with', 'I', 'i', 'on', 'his',
            'he', 'she', 'her', 'are', 'be', 'an', 'have', 'not', 'at', 'was', 'from', 'you', 'when', 'about',
            'more', 'like', 'so', 'has', 'all', 'their', 'there', 'theyre', 'which', 'just', 'some', 'most',
            'will', 'than', 'we', 'also', 'him', 'into', 'than', 'what', 'do', 'been', 'would', 'much', 'out',
            'up', 'only', 'other', 'the', 'be', 'and', 'a', 'of', 'to', 'in', 'i', 'you', 'it', 'have', 'to',
            'that', 'for', 'do', 'he', 'with',
            'on', 'this', 'nâ€™t', 'we', 'that', 'not', 'but', 'they', 'say', 'at', 'what', 'his', 'from', 'go',
            'or',
            'by', 'get', 'she', 'my', 'can', 'as', 'know', 'if', 'me', 'your', 'all', 'who', 'about', 'their',
            'will',
            'so', 'would', 'make', 'just', 'up', 'think', 'time', 'there', 'see', 'her', 'as', 'out', 'one',
            'come',
            'people', 'take', 'year', 'him', 'them', 'some', 'want', 'how', 'when', 'which', 'now', 'like',
            'other',
            'could', 'our', 'into', 'here', 'then', 'than', 'look', 'way', 'more', 'these', 'no', 'thing',
            'well',
            'because', 'also', 'two', 'use', 'tell', 'first', 'man', 'day', 'find', 'give', 'more', 'new',
            'one', 'us',
            'any', 'those', 'very', 'her', 'need', 'back', 'there', 'should', 'even', 'only', 'many', 'really',
            'work',
            'life', 'why', 'right', 'down', 'on', 'try', 'let', 'something', 'too', 'call', 'woman', 'may',
            'still',
            'through', 'mean', 'after', 'never', 'no', 'world', 'in', 'feel', 'yeah', 'last', 'child', 'oh',
            'over', 'ask', 'when', 'as', 'school', 'state', 'much', 'talk', 'out', 'keep', 'leave', 'put',
            'like', 'help',
            'big', 'where', 'same', 'all', 'own', 'while', 'start', 'three', 'high', 'every', 'another',
            'become', 'most',
            'between', 'happen', 'family', 'over', 'president', 'old', 'yes', 'house', 'show', 'again',
            'student', 'so',
            'seem', 'might', 'part', 'hear', 'its', 'place', 'problem', 'where', 'believe', 'country', 'always',
            'week',
            'point', 'hand', 'off', 'play', 'turn', 'few', 'group', 'such', 'against', 'run', 'guy', 'about',
            'case',
            'question', 'work', 'night', 'live', 'game', 'number', 'write', 'bring', 'without', 'money', 'lot',
            'most',
            'book', 'system', 'government', 'next', 'city', 'company', 'story', 'today', 'job', 'move', 'must',
            'friend',
            'during', 'begin', 'love', 'each', 'hold', 'different', 'american', 'little', 'before', 'ever',
            'word', 'fact',
            'right', 'read', 'anything', 'nothing', 'sure', 'small', 'month', 'program', 'maybe', 'right',
            'under',
            'business', 'home', 'kind', 'stop', 'pay', 'study', 'since', 'issue', 'name', 'idea', 'room',
            'percent', 'far',
            'away', 'law', 'actually', 'large', 'though', 'provide', 'lose', 'power', 'kid', 'war',
            'understand', 'head',
            'mother', 'real', 'team', 'eye', 'long', 'long', 'side', 'water', 'young', 'wait', 'okay', 'both',
            'yet', 'after', 'meet', 'service', 'area', 'important', 'person', 'hey', 'thank', 'much', 'someone',
            'end',
            'change', 'however', 'only', 'around', 'hour', 'everything', 'national', 'four', 'line', 'girl',
            'around',
            'watch', 'until', 'father', 'sit', 'create', 'information', 'car', 'learn', 'least', 'already',
            'kill',
            'minute', 'party', 'include', 'stand', 'together', 'back', 'follow', 'health', 'remember', 'often',
            'reason',
            'speak', 'ago', 'set', 'black', 'member', 'community', 'once', 'social', 'news', 'allow', 'win',
            'body',
            'lead', 'continue', 'whether', 'enough', 'spend', 'level', 'able', 'political', 'almost', 'boy',
            'university',
            'before', 'stay', 'add', 'later', 'change', 'five', 'probably', 'center', 'among', 'face', 'public',
            'die',
            'food', 'else', 'history', 'buy', 'result', 'morning', 'off', 'parent', 'office', 'course', 'send',
            'research',
            'walk', 'door', 'white', 'several', 'court', 'home', 'grow', 'better', 'open', 'moment',
            'including',
            'consider', 'both', 'such', 'little', 'within', 'second', 'late', 'street', 'free', 'better',
            'everyone',
            'policy', 'table', 'sorry', 'care', 'low', 'human', 'please', 'hope', 'TRUE', 'process', 'teacher',
            'data',
            'offer', 'death', 'whole', 'experience', 'plan', 'easy', 'education', 'build', 'expect', 'fall',
            'himself',
            'age', 'hard', 'sense', 'across', 'show', 'early', 'college', 'music', 'appear', 'mind', 'class',
            'police',
            'use', 'effect', 'season', 'tax', 'heart', 'son', 'art', 'possible', 'serve', 'break', 'although',
            'end',
            'market', 'even', 'air', 'force', 'require', 'foot', 'up', 'listen', 'agree', 'according', 'anyone',
            'baby',
            'wrong', 'love', 'cut', 'decide', 'republican', 'full', 'behind', 'pass', 'interest', 'sometimes',
            'security',
            'eat', 'report', 'control', 'rate', 'local', 'suggest', 'report', 'nation', 'sell', 'action',
            'support',
            'wife', 'decision', 'receive', 'value', 'base', 'pick', 'phone', 'thanks', 'event', 'drive',
            'strong', 'reach',
            'remain', 'explain', 'site', 'hit', 'pull', 'church', 'model', 'perhaps', 'relationship', 'six',
            'fine',
            'movie', 'field', 'raise', 'less', 'player', 'couple', 'million', 'themselves', 'record',
            'especially',
            'difference', 'light', 'development', 'federal', 'former', 'role', 'pretty', 'myself', 'view',
            'price',
            'effort', 'nice', 'quite', 'along', 'voice', 'finally', 'department', 'either', 'toward', 'leader',
            'because',
            'photo', 'wear', 'space', 'project', 'return', 'position', 'special', 'million', 'film', 'need',
            'major',
            'type', 'town', 'article', 'road', 'form', 'chance', 'drug', 'economic', 'situation', 'choose',
            'practice',
            'cause', 'happy', 'science', 'join', 'teach', 'early', 'develop', 'share', 'yourself', 'carry',
            'clear',
            'brother', 'matter', 'dead', 'image', 'star', 'cost', 'simply', 'post', 'society', 'picture',
            'piece', 'paper',
            'energy', 'personal', 'building', 'military', 'open', 'doctor', 'activity', 'exactly', 'american',
            'media',
            'miss', 'evidence', 'product', 'realize', 'save', 'arm', 'technology', 'catch', 'comment', 'look',
            'term',
            'color', 'cover', 'describe', 'guess', 'choice', 'source', 'mom', 'soon', 'director',
            'international', 'rule',
            'campaign', 'ground', 'election', 'face', 'uh', 'check', 'page', 'fight', 'itself', 'test',
            'patient',
            'produce', 'certain', 'whatever', 'half', 'video', 'support', 'throw', 'third', 'care', 'rest',
            'recent',
            'available', 'step', 'ready', 'opportunity', 'official', 'oil', 'call', 'organization', 'character',
            'single',
            'current', 'likely', 'county', 'future', 'dad', 'whose', 'less', 'shoot', 'industry', 'second',
            'list',
            'general', 'stuff', 'figure', 'attention', 'forget', 'risk', 'no', 'focus', 'short', 'fire', 'dog',
            'red',
            'hair', 'point', 'condition', 'wall', 'daughter', 'before', 'deal', 'author', 'truth', 'upon',
            'husband',
            'period', 'series', 'order', 'officer', 'close', 'land', 'note', 'computer', 'thought', 'economy',
            'goal',
            'bank', 'behavior', 'sound', 'deal', 'certainly', 'nearly', 'increase', 'act', 'north', 'well',
            'blood',
            'culture', 'medical', 'ok', 'everybody', 'top', 'difficult', 'close', 'language', 'window',
            'response',
            'population', 'lie', 'tree', 'park', 'worker', 'draw', 'plan', 'drop', 'push', 'earth', 'cause',
            'per',
            'private', 'tonight', 'race', 'than', 'letter', 'other', 'gun', 'simple', 'course', 'wonder',
            'involve',
            'hell', 'poor', 'each', 'answer', 'nature', 'administration', 'common', 'no', 'hard', 'message',
            'song',
            'enjoy', 'similar', 'congress', 'attack', 'past', 'hot', 'seek', 'amount', 'analysis', 'store',
            'defense',
            'bill', 'like', 'cell', 'away', 'performance', 'hospital', 'bed', 'board', 'protect', 'century',
            'summer',
            'material', 'individual', 'recently', 'example', 'represent', 'fill', 'state', 'place', 'animal',
            'fail',
            'factor', 'natural', 'sir', 'agency', 'usually', 'significant', 'help', 'ability', 'mile',
            'statement',
            'entire', 'democrat', 'floor', 'serious', 'career', 'dollar', 'vote', 'sex', 'compare', 'south',
            'forward',
            'subject', 'financial', 'identify', 'beautiful', 'decade', 'bit', 'reduce', 'sister', 'quality',
            'quickly',
            'act', 'press', 'worry', 'accept', 'enter', 'mention', 'sound', 'thus', 'plant', 'movement',
            'scene',
            'section', 'treatment', 'wish', 'benefit', 'interesting', 'west', 'candidate', 'approach',
            'determine',
            'resource', 'claim', 'answer', 'prove', 'sort', 'enough', 'size', 'somebody', 'knowledge', 'rather',
            'hang',
            'sport', 'tv', 'loss', 'argue', 'left', 'note', 'meeting', 'skill', 'card', 'feeling', 'despite',
            'degree',
            'crime', 'that', 'sign', 'occur', 'imagine', 'vote', 'near', 'king', 'box', 'present', 'figure',
            'seven',
            'foreign', 'laugh', 'disease', 'lady', 'beyond', 'discuss', 'finish', 'design', 'concern', 'ball',
            'east',
            'recognize', 'apply', 'prepare', 'network', 'huge', 'success', 'district', 'cup', 'name',
            'physical', 'growth',
            'rise', 'hi', 'standard', 'force', 'sign', 'fan', 'theory', 'staff', 'hurt', 'legal', 'september',
            'set',
            'outside', 'et', 'strategy', 'clearly', 'property', 'lay', 'final', 'authority', 'perfect',
            'method', 'region',
            'since', 'impact', 'indicate', 'safe', 'committee', 'supposed', 'dream', 'training', 'shit',
            'central',
            'option', 'eight', 'particularly', 'completely', 'opinion', 'main', 'ten', 'interview', 'exist',
            'remove',
            'dark', 'play', 'union', 'professor', 'pressure', 'purpose', 'stage', 'blue', 'herself', 'sun',
            'pain',
            'artist', 'employee', 'avoid', 'account', 'release', 'fund', 'environment', 'treat', 'specific',
            'version',
            'shot', 'hate', 'reality', 'visit', 'club', 'justice', 'river', 'brain', 'memory', 'rock', 'talk',
            'camera',
            'global', 'various', 'arrive', 'notice', 'bit', 'detail', 'challenge', 'argument', 'lot', 'nobody',
            'weapon',
            'station', 'island', 'absolutely', 'instead', 'discussion', 'instead', 'affect', 'design', 'little',
            'anyway', 'respond', 'control', 'trouble', 'conversation', 'manage', 'close', 'date', 'public',
            'army', 'top',
            'post', 'charge', 'seat', 'does', 'had', 'were', 'dont', 'theres', 'being', 'made', 'hes', 'shes',
            'doesnt', 'things'}


class Document:

    def __init__(self):
        self.words = set()

    def contains_word(self, word):
        return word in self.words

    def add_word(self, word):
        self.words.add(word)


'''
Computes the logistic function.
'''


def sigma(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:

    def __init__(self, n_features=4):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        # self.class_dict = {'action': 0, 'comedy': 1}
        self.feature_dict = dict()
        self.n_features = n_features
        self.theta = np.zeros(n_features + 1)  # weights (and bias)
        self.prior = None
        self.cls_word_counts = dict()

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''

    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()
        files_words = dict()
        for root, dirs, files in os.walk(data_set):
            # This assumes that the only directory containing any files will be the training/testing data. Otherwise,
            # this will crash.
            cls = root.split(os.sep)[-1]
            word_counts = dict()
            for name in files:
                with open(os.path.join(root, name)) as f:
                    filenames.append(name)
                    classes.setdefault(name, cls)
                    words = list()
                    for line in f:
                        for word in line.split():
                            new_word = re.sub('[^A-Za-z\\d]+', '', word)
                            if len(new_word):
                                words.append(new_word)
                                if new_word not in dont_use:
                                    word_counts.setdefault(new_word, 0)
                                    word_counts[new_word] += 1
                    files_words[name] = words

            if files and cls not in self.cls_word_counts:
                self.cls_word_counts[cls] = word_counts

        temps = []
        for key in self.cls_word_counts:
            self.cls_word_counts[key] = dict(
                sorted(self.cls_word_counts[key].items(), key=lambda item: item[1], reverse=True))
            d = self.cls_word_counts[key]
            if len(d) > 5:
                temp = dict()
                for i, key in enumerate(d):
                    temp[key] = d[key]
                    if i == 4:
                        temps.append(temp)
                        break
            else:
                temps.append(d)

        for i, key in enumerate(self.cls_word_counts):
            self.cls_word_counts[key] = temps[i]

        for name, words in files_words.items():
            documents.setdefault(name, self.featurize(words))

        return filenames, classes, documents

    def __is_top_five_word(self, word):
        contains = []
        for word_dict in self.cls_word_counts.values():
            contains.append(word in word_dict)
        return contains

    '''
    Given a document (as a list of words) and a vector, fills the vector of size self.n_features with
    the self.n_features that occur the most in the document.
    '''

    def __top_n_feature_word_counts(self, document, vector):
        counts = dict()
        for word in document:
            counts.setdefault(word, 0)
            counts[word] += 1

        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        for i in range(self.n_features):
            if i > len(sorted_counts) - 1:
                vector[i] = sorted_counts[0][1]
            else:
                vector[i] = sorted_counts[i][1]

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''

    def __get_num_occurrences_of_top_five(self, document):
        counts = {}
        for cls, _dict in self.cls_word_counts.items():
            temp = dict()
            for key in _dict:
                temp[key] = 0
            counts[cls] = temp

        for cls, _dict in counts.items():
            d = self.cls_word_counts[cls]
            for word in document:
                if word in _dict:
                    _dict[word] += 1
        return counts

    @staticmethod
    def __contains_negative_words(document):
        negatives = ['bad', 'awful', 'terrible', 'gross', 'abysmal', 'annoy', 'annoying']
        res = []
        for word in negatives:
            if word in document:
                res.append(1)
            else:
                res.append(0)
        return res

    @staticmethod
    def __contains_positive_words(document):
        negatives = ['good', 'great', 'awesome', 'amazing', 'brilliant', 'beautiful', 'delightful', 'delight']
        res = []
        for word in negatives:
            if word in document:
                res.append(1)
            else:
                res.append(0)
        return res

    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        counts = self.__get_num_occurrences_of_top_five(document)
        i = 0
        for cls, dict in counts.items():
            for word_count in dict.values():
                vector[i] = word_count
                i += 1

        pos = self.__contains_positive_words(document)
        negs = self.__contains_negative_words(document)
        for val in pos:
            vector[i] = val
            i += 1
        for val in negs:
            vector[i] = val
            i += 1

        vector[-2] = len(document)

        vector[-1] = 1
        return vector

    def __create_matrices(self, documents, classes, curr_batch, batch_len):
        rows = batch_len
        cols = self.n_features + 1
        X = np.zeros((rows, cols))
        for i in range(batch_len):
            X[i] = documents[curr_batch[i]]

        Y = np.zeros(batch_len)
        for j in range(batch_len):
            Y[j] = self.class_dict[classes[curr_batch[j]]]

        return X, Y

    # LCE(ˆy, y) = −[y log ˆy + (1 − y) log(1 − yˆ)]
    @staticmethod
    def cross_entropy_loss(Y, y_hat):
        eps = np.finfo(float).eps
        return -1 * (Y @ np.log(y_hat + eps) + (1 - Y) @ np.log(1 - y_hat + eps))

    '''
    Trains a logistic regression classifier on a training set.
    '''

    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]

                # create and fill in matrix x and vector y
                X, Y = self.__create_matrices(documents, classes, minibatch, len(minibatch))

                # compute y_hat
                y_hat = sigma(np.dot(X, self.theta))

                # update loss
                loss += np.sum(self.cross_entropy_loss(Y, y_hat))

                # compute gradient
                grad_batch = np.dot(X.T, (y_hat - Y)) / len(minibatch)

                # update weights (and bias)
                self.theta -= eta * grad_batch

            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    def __determine_class(self, y_hat):
        vals = list(self.class_dict.values())
        if y_hat <= 0.5:
            return vals[0]
        return vals[1]

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''

    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            # create and fill in matrix x and vector y
            X, Y = self.__create_matrices(documents, classes, [name], 1)

            # compute y_hat
            y_hat = sigma(np.dot(X, self.theta))

            results[name]['correct'] = self.class_dict[classes[name]]
            results[name]['predicted'] = self.__determine_class(y_hat)

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


if __name__ == '__main__':
    # lr = LogisticRegression(n_features=23)
    lr = LogisticRegression(n_features=26)
    # make sure these point to the right directories
    lr.train('movie_reviews/train', batch_size=30, n_epochs=20, eta=0.01)
    # lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    results = lr.test('movie_reviews/dev')
    # results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)
