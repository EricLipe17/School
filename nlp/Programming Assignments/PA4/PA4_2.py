import os
import numpy as np
import scipy


class SATAnalogyQuestion:
    def __init__(self, analogy_split):
        self.analogy = analogy_split[0:2]
        self.possibilities = list()
        self.pos = analogy_split[-1]
        self.solution = None
        self.predicted_solution = None
        self.soln_converter = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

    def add_possibility(self, possibility):
        self.possibilities.append(possibility)

    def set_solution(self, soln):
        self.solution = self.soln_converter[soln]

    def set_predicted_solution(self, soln):
        self.predicted_solution = soln


class TrainedDistributionalSemanticModel:
    def __init__(self, vector_data_path, data_path):
        self.vector_data_path = vector_data_path
        self.data_path = data_path
        self.word_vec_map = dict()
        self.sat_questions = list()
        self.unk_vec = None
        self.__load_data()
        self.__load_vectors()
        self.__create_unk_vector()

    def __load_vectors(self):
        for root, dirs, files in os.walk(self.vector_data_path):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        split = line.split()
                        split[0] = split[0].replace("\x1b[?1034h", "")
                        self.word_vec_map[split[0]] = np.array(split[1:], dtype=np.float16)

    def __load_data(self):
        prev_ln_was_header = False
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    analogy = None
                    for line in f:
                        if line[0] != "#" and line[0] != "\n":
                            if len(line) == 2 and line[0].isalpha():
                                analogy.set_solution(line[0])
                                continue
                            if prev_ln_was_header:
                                prev_ln_was_header = False
                                splits = line.split()
                                analogy = SATAnalogyQuestion(splits)
                                self.sat_questions.append(analogy)
                            elif line[0:3] == "190":
                                prev_ln_was_header = True
                            else:
                                analogy.add_possibility(line)

    def __create_unk_vector(self):
        values = list(self.word_vec_map.values())
        matrix = np.zeros((len(self.word_vec_map), len(values[0])))
        for i, vec in enumerate(values):
            matrix[i] = vec
        self.unk_vec = np.average(matrix, axis=0)

    @staticmethod
    def cos_sim(a, b):
        return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def euclidean_distance(a, b):
        return abs(scipy.linalg.norm(a) - scipy.linalg.norm(b))

    def test(self):
        for question in self.sat_questions:
            an_v1 = self.word_vec_map.get(question.analogy[0], self.unk_vec)
            an_v2 = self.word_vec_map.get(question.analogy[1], self.unk_vec)
            an_vec = np.concatenate((an_v1, an_v2), axis=None)
            predicted_solution = None
            for i, possibility in enumerate(question.possibilities):
                split = possibility.split()
                p_v1 = self.word_vec_map.get(split[0], self.unk_vec)
                p_v2 = self.word_vec_map.get(split[1], self.unk_vec)
                p_vec = np.concatenate((p_v1, p_v2), axis=None)
                similarity = self.cos_sim(p_vec, an_vec)

                if predicted_solution is None:
                    predicted_solution = (i, similarity)
                elif similarity > predicted_solution[1]:
                    predicted_solution = (i, similarity)
            question.set_predicted_solution(predicted_solution[0])

    def evaluate(self):
        correct = 0
        for question in self.sat_questions:
            correct += (question.solution == question.predicted_solution)
        accuracy = float(correct) / len(self.sat_questions)
        print(f"\tAccuracy: {accuracy * 100}%")


print("Evaluating the Google vectors:")
google = TrainedDistributionalSemanticModel("GoogleNews-vectors-rcv_vocab.txt.tar", "sat_data")
google.test()
google.evaluate()

print("\nEvaluating the COMPOSE vectors:")
compose = TrainedDistributionalSemanticModel("EN-wform.w.2.ppmi.svd.500.rcv_vocab-1.txt.tar", "sat_data")
compose.test()
compose.evaluate()
