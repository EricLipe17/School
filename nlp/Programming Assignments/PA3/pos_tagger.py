# CS542 Fall 2021 Homework 3
# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
import random
from random import Random


class POSTagger:

    def __init__(self):
        # for testing with the toy corpus from worked example
        self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          'dwarf': 4, 'cheered': 5}
        # initial tag weights [shape = (len(tag_dict),)]
        self.initial = np.array([-0.3, -0.7, 0.3])
        # tag-to-tag transition weights [shape = (len(tag_dict),len(tag_dict))]
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        # tag emission weights [shape = (len(word_dict),len(tag_dict))]
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        self.unk_index = -1

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''

    def make_dicts(self, train_set):
        # Changing these to dictionaries because I want order preserved
        tag_vocabulary = set()
        word_vocabulary = set()
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            files.sort()
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # BEGIN STUDENT CODE
                    # create vocabularies of every tag and word
                    #  that exists in the training data
                    for line in f:
                        for word in line.split():
                            word_and_pos = word.rsplit('/', 1)
                            tag_vocabulary.add(word_and_pos[1])
                            word_vocabulary.add(word_and_pos[0])
                    # END STUDENT CODE
        # create tag_dict and word_dict
        # if you implemented the rest of this
        #  function correctly, these should be formatted
        #  as they are above in __init__
        self.tag_dict = {v: k for k, v in enumerate(tag_vocabulary)}
        self.word_dict = {v: k for k, v in enumerate(word_vocabulary)}

    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''

    def load_data(self, data_set):
        sentence_ids = []  # doc name + ordinal number of sentence (e.g., ca010)
        sentences = dict()
        tag_lists = dict()
        word_lists = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    # BEGIN STUDENT CODE
                    # for each sentence in the document
                    #  1) create a list of tags and list of words that
                    #     appear in this sentence
                    #  2) create the sentence ID, add it to sentence_ids
                    #  3) add this sentence's tag list to tag_lists and word
                    #     list to word_lists
                    words = []
                    words_and_pos = ""
                    tags = []
                    index = 0
                    for line in f:
                        for word in line.split():
                            word_and_pos = word.rsplit('/', 1)
                            word_index = self.unk_index
                            if word_and_pos[0] in self.word_dict:
                                word_index = self.word_dict[word_and_pos[0]]
                            words.append(word_index)

                            tag = self.unk_index
                            if word_and_pos[1] in self.tag_dict:
                                tag = self.tag_dict[word_and_pos[1]]

                            tags.append(tag)
                            words_and_pos += word + " "

                            if word_and_pos[1] == ".":
                                sentence_id = name + str(index)
                                sentence_ids.append(sentence_id)

                                sentences[sentence_id] = words_and_pos
                                word_lists[sentence_id] = words
                                tag_lists[sentence_id] = tags

                                words = []
                                words_and_pos = ""
                                tags = []

                    # If we don't ever see the sentence termination we assume all words collected to that point are a
                    # sentence.
                    if len(words) + len(tags) + len(words_and_pos) > 0:
                        sentence_id = name + str(index)
                        sentence_ids.append(sentence_id)

                        sentences[sentence_id] = words_and_pos
                        word_lists[sentence_id] = words
                        tag_lists[sentence_id] = tags
                    # END STUDENT CODE
        return sentence_ids, sentences, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''

    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        best_path = [None] * T
        # BEGIN STUDENT CODE
        # initialization step
        #  fill out first column of viterbi trellis
        #  with initial + emission weights of the first observation
        v[:, 0] = self.initial + self.emission[sentence[0]]

        # recursion step
        max_vector = None
        for i in range(1, T):
            #  1) fill out the t-th column of viterbi trellis
            #  with the max of the t-1-th column of trellis
            #  + transition weights to each state
            #  + emission weights of t-th observation
            curr_word = sentence[i]
            viterbi_prev_word = np.reshape(v[:, i - 1], (len(v[:, i - 1]), 1))
            word_emission_probs = self.emission[curr_word, :]

            mat = viterbi_prev_word + self.transition + word_emission_probs
            max_vector = np.amax(mat, axis=0)
            v[:, i] = max_vector

            #  2) fill out the t-th column of the backpointer trellis
            #  with the associated argmax values
            # termination step
            vec = np.argmax(mat, axis=0)
            backpointer[:, i] = vec

        #  1) get the most likely ending state, insert it into best_path
        final_state_max = np.where(max_vector == np.amax(max_vector))[0][0]
        best_path[-1] = final_state_max

        #  2) fill out best_path from backpointer trellis`
        for i in range(T - 1, 0, -1):
            col = backpointer[:, i]
            bp = col[best_path[i]]
            best_path[i - 1] = bp

        # END STUDENT CODE
        return best_path

    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''

    def train(self, train_set, dummy_data=None):
        self.make_dicts(train_set)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(train_set)
        if dummy_data is None:  # for automated testing: DO NOT CHANGE!!
            Random(0).shuffle(sentence_ids)
            self.initial = np.zeros(len(self.tag_dict))
            self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
            self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        else:
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            # get the word sequence for this sentence and the correct tag sequence
            # use viterbi to predict
            correct_tags = tag_lists[sentence_id]
            pred_tags = self.viterbi(word_lists[sentence_id])

            # if mistake
            if correct_tags != pred_tags:
                self.initial[correct_tags[0]] += 1
                self.initial[pred_tags[0]] -= 1
                words = word_lists[sentence_id]
                for j in range(len(correct_tags)):
                    corr_tag, pred_tag, word = correct_tags[j], pred_tags[j], words[j]

                    # promote weights that appear in correct sequence
                    self.emission[word][corr_tag] += 1
                    if j + 1 < len(correct_tags):
                        self.transition[corr_tag][correct_tags[j + 1]] += 1

                    #  demote weights that appear in (incorrect) predicted sequence
                    self.emission[word][pred_tag] -= 1
                    if j + 1 < len(pred_tags):
                        self.transition[pred_tag][pred_tags[j + 1]] -= 1

            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''

    def test(self, dev_set, dummy_data=None):
        results = defaultdict(dict)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(dev_set)
        if dummy_data is not None:  # for automated testing: DO NOT CHANGE!!
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            # should be very similar to train function before mistake check
            correct_tags = tag_lists[sentence_id]
            best_path = self.viterbi(word_lists[sentence_id])
            results[sentence_id]['correct'] = correct_tags
            results[sentence_id]['predicted'] = best_path
            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return sentences, results

    '''
    Given results, calculates overall accuracy.
    This evaluate function calculates accuracy ONLY,
    no precision or recall calculations are required.
    '''

    def evaluate(self, sentences, results, dummy_data=False):
        if not dummy_data:
            self.sample_results(sentences, results)
        accuracy = 0.0
        # BEGIN STUDENT CODE
        # for each sentence, how many words were correctly tagged out of the total words in that sentence?
        total_correct = 0
        total_words = 0
        for sentence_id in results:
            for correct, predicted in zip(results[sentence_id]['correct'], results[sentence_id]['predicted']):
                total_correct += correct == predicted
                total_words += 1
                # total_correct += sum(x == y for x, y in zip(correct, predicted))
                # total_words += len(correct)
        accuracy = total_correct / float(total_words)
        # END STUDENT CODE
        return accuracy

    '''
    Prints out some sample results, with original sentence,
    correct tag sequence, and predicted tag sequence.
    This is just to view some results in an interpretable format.
    You do not need to do anything in this function.
    '''

    def sample_results(self, sentences, results, size=2):
        print('\nSample results')
        results_sample = [random.choice(list(results)) for i in range(size)]
        inv_tag_dict = {v: k for k, v in self.tag_dict.items()}
        for sentence_id in results_sample:
            length = len(results[sentence_id]['correct'])
            correct_tags = [inv_tag_dict[results[sentence_id]['correct'][i]] for i in range(length)]
            predicted_tags = [inv_tag_dict[results[sentence_id]['predicted'][i]] for i in range(length)]
            print(sentence_id, sentences[sentence_id], 'Correct:\t', correct_tags, '\n Predicted:\t', predicted_tags,
                  '\n')


if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    # pos.train('data_small/train')  # train: toy data
    # pos.train('brown_news/train') # train: news data only
    pos.train('brown/train') # train: full data
    # sentences, results = pos.test('data_small/test')  # test: toy data
    # sentences, results = pos.test('brown_news/dev') # test: news data only
    sentences, results = pos.test('brown/dev') # test: full data
    print('\nAccuracy:', pos.evaluate(sentences, results))
