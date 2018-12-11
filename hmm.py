import numpy as np
import pickle
import os
import json
import datetime
import util
from collections import deque


class HMM:
    def __init__(self, ngram, delta=0.1):
        self.ngram = ngram
        """
        tag maps to normalize some tags in data set
        """
        self.tags_map = {
            "Ab": "A",
            "B": "FW",
            "Fw": "FW",
            "Nb": "FW",
            "Ne": "Nc",
            "Ni": "Np",
            "NNP": "Np",
            "Ns": "Nc",
            "S": "Z",
            "Vb": "V",
            "Y": "Np"
        }
        self.delta = delta
        self.tags = {'Q0': 0}
        self.vocab = {}
        transition_shape = tuple([1 for i in range(ngram)])
        self.Q = np.zeros(transition_shape)
        self.E = np.zeros(tuple([1, 0]))
        self.export_file_name = 'result/{}{}{}'
        self.lamda = 0.1

    def normalizeTag(self, tag):
        if tag in self.tags_map:
            return self.tags_map[tag]
        else:
            return tag

    def initialize_matrix(self, corpus):
        for idx, sentence in enumerate(corpus):
            pre_states = deque([self.tags['Q0']
                                for i in range(self.ngram - 1)])
            for token in sentence.tokens:
                word = token.getWord()
                tag = token.getTag()
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                    self.E = util.expand_one_dimesion(self.E, 1, 1)
                if tag not in self.tags:
                    self.tags[tag] = len(self.tags)
                    self.Q = util.expand_matrix(self.Q, 1)
                    self.E = util.expand_one_dimesion(self.E, 1, 0)

                self.E[self.tags[tag], self.vocab[word]] += 1
                pre_states.append(self.tags[tag])
                self.Q[tuple(pre_states)] += 1
                pre_states.popleft()
            if (idx+1) % 500 == 0:
                print('> Read sentence {}'.format(idx))
        transition_sum = np.sum(self.Q, axis=-1, keepdims=True)
        transition_sum[transition_sum == 0] = -1
        emission_sum = np.sum(self.E, axis=-1, keepdims=True)
        self.Q = (self.Q + self.delta) / \
            (transition_sum + self.delta*len(self.vocab))
        self.E = (self.E + self.delta) / \
            (emission_sum + self.delta*len(self.vocab))
        print('> shape Q: {}, shape E: {}'.format(self.Q.shape, self.E.shape))

    def saveModel(self, name='model'):
        # time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        time = ''
        basePath = os.path.dirname(os.path.abspath(__file__))
        resultPath = basePath + '/result'
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
            print('> Create new folder {}'.format(resultPath))
        with open(self.export_file_name.format('vocab', time, '.json'), 'w') as fp:
            json.dump(self.vocab, fp)
        with open(self.export_file_name.format('tags', time, '.json'), 'w') as fp:
            json.dump(self.tags, fp)

        np.save(self.export_file_name.format('transition-matrix', time, '.npy'),
                self.Q)
        np.savetxt(self.export_file_name.format('emission-matrix', time, '.txt'),
                   self.E, delimiter=',')

    def loadModel(self):
        time = ''
        with open(self.export_file_name.format('vocab', time, '.json'), 'r') as fp:
            print('> Load vocabulary')
            self.vocab = json.load(fp)
        with open(self.export_file_name.format('tags', time, '.json'), 'r') as fp:
            self.tags = json.load(fp)
            print('> Load tags')

        self.Q = np.load(self.export_file_name.format(
            'transition-matrix', time, '.npy'))
        print('> Load transition matrix. Shape: {}'.format(self.Q.shape))
        self.E = np.loadtxt(self.export_file_name.format(
            'emission-matrix', time, '.txt'), delimiter=',')
        print('> Load emission matrix. Shape: {}'.format(self.E.shape))

    def get_tag(self, search_index):
        for tag in self.tags:    # for name, age in list.items():  (for Python 3.x)
            if self.tags[tag] == search_index:
                return tag
        return None

    def loadCorpus(self, fileName):
        print('> Start read file: {}'.format(fileName))
        corpus = []
        with open(fileName) as fp:
            newSetence = Sentence()
            for line in fp:
                line = line.strip()
                line = line
                parts = line.rpartition('\t')
                word = parts[0].lower()
                tag = self.normalizeTag(parts[-1])
                if(word == ''):
                    corpus.append(newSetence)
                    newSetence = Sentence()
                else:
                    newSetence.addToken(Token(word, tag))
            corpus.append(newSetence)
        print('> End read file: {}'.format(fileName))
        return corpus

    def decode(self, sentence):
        if len(sentence) > 0:
            # V(tags,sentence length) denote for viterbi matrix
            V = np.zeros((len(self.tags), len(sentence)), dtype=np.float64)
            # backtrack(tags,sentence length)
            backtrack = np.zeros(
                (len(self.tags), len(sentence)), dtype=np.int16)

            if(sentence[0] not in self.vocab):
                # Add new word into vocabulary and assign new word emission = 1/len(vocab)
                self.vocab[sentence[0]] = len(self.vocab)
                self.E = util.expand_one_dimesion(
                    self.E, 1, 1, 1/(len(self.vocab)))
            # Initialize vitebi 0
            # V[i,0] = Q[Si|Q0..Q0] * E(W|Si)
            V[:, 0] = self.Q[tuple([self.tags['Q0'] for i in range(self.ngram - 1)])
                             ] * self.E[:, self.vocab[sentence[0]]]
            # Given P(Si|Sj...Sk) in transition matrix. Denote Sj...Sk as previous state
            pre_states = np.zeros(
                (len(self.tags), self.ngram-1), dtype=np.int16)
            pre_states[:, -1] = range(len(self.tags))
            # Start from word at position 2.
            # Value index indicate previous word's viterbi value
            for index, word in enumerate(sentence[1:]):
                if word not in self.vocab:
                    # Add new word into vocabulary and assign new word emission = 1/len(vocab)
                    self.vocab[word] = len(self.vocab)
                    self.E = util.expand_one_dimesion(
                        self.E, 1, 1, 1/(len(self.vocab)))
                # V_candidate(states, states) denote for all likelihood probability for all state at current word
                # V_candidate[i,j] = V_before[j] * P[Si|Sj...Sk] * P[W|Si]
                V_candidate = V[:, index] * \
                    np.array([self.Q[tuple(prev)] for prev in pre_states]).T * \
                    self.E[:, self.vocab[word]].reshape((-1, 1))
                # Find max each row to update viterbi matrix for current word
                # Update back track matrix
                V[:, index+1] = np.amax(V_candidate, axis=1)
                backtrack[:, index+1] = np.argmax(V_candidate, axis=1)
                # Update previous state for next word
                temp_states = np.copy(pre_states)
                for i, prev in enumerate(backtrack[:, index+1]):
                    pre_states[i, :-1] = temp_states[prev, 1:]
                    pre_states[i, -1] = i
            tag_indexs = []
            tag_indexs.append(np.argmax(V[:, -1]))
            for i, col in enumerate(backtrack[:, ::-1].T):
                tag_indexs.append(col[tag_indexs[-1]])
            rs = [self.get_tag(i) for i in tag_indexs]
            rs.reverse()
            return rs[1:]

    def evaluate(self, corpus):
        correct_num = 0
        for index, sentence in enumerate(corpus):
            predict_tag = self.decode(sentence.getWords())
            # print('> Sentence {}:'.format(index))
            # print('> predict_tag: {}'.format(predict_tag))
            label_tag = sentence.getTags()
            count = 0
            for idx, tag in enumerate(predict_tag):
                if (tag == label_tag[idx]):
                    count += 1
            correct_num+= count/len(predict_tag)
        return correct_num/len(corpus)


class Sentence():
    def __init__(self):
        self.tokens = []

    def addToken(self, token):
        self.tokens.append(token)

    def size(self):
        return len(self.tokens)

    def getWords(self):
        return [token.getWord() for token in self.tokens]

    def getTags(self):
        return [token.getTag() for token in self.tokens]


class Token():
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

    def getWord(self):
        return self.word

    def getTag(self):
        return self.tag


# fix_bug_file = 'experiment/test_data.txt'
# trainFile = 'corpus/train/train.txt'
# devFile = 'corpus/train/dev.txt'
# testFile = 'corpus/vlsp2016/test.txt'
# model = HMM(3)
# corpus = model.loadCorpus(trainFile)
# test_corpus = model.loadCorpus(testFile)
# model.initialize_matrix(corpus)
# model.saveModel()
# # model.loadModel()
# print(model.evaluate(test_corpus))


# index = 68
# print('Sentence: {}'.format(corpus[index].getWords()))
# print('Label tag: {}'.format(corpus[index].getTags()))
# print('Predict tag: {}'.format(model.decode(corpus[index].getWords())))

