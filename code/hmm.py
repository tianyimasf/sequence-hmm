# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.


import os
import argparse
import numpy as np
from nlputil import *   # utility methods for working with text
import random
from matplotlib import pyplot as plt
import pickle
import random


# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size
#
# Note: You may want to add fields for expectations.


class HMM:
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size', 'word_vocab')

    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.
    def __init__(self, num_states, vocab_size, word_vocab):
        # Num_states is number of hidden states
        self.num_states = num_states
        # Vocab size is the number of unique words
        self.vocab_size = vocab_size
        # Transitions are a KxK matrix where K is the number of hidden states. Initialized randomly between 0 and 1
        self.transitions = self.normalize(
            np.random.rand(self.num_states, self.num_states))
        # pi is vector of size K, also initialized between 0 and 1
        self.pi = self.normalize_row(np.random.rand(self.num_states))
        # TEMPORARY: Intializing emissions to uniform distribution
        self.emissions = self.normalize(
            np.random.rand(self.num_states, self.vocab_size))
        # save the word vocab for future testing and predicting
        self.word_vocab = word_vocab

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset):
        mean_loglikelihood = 0
        count = 0
        for sample in dataset:
            count += 1
            loglikelihood = self.loglikelihood_helper(sample)
            mean_loglikelihood += loglikelihood
            # print(count/len(dataset)*100, "%")
        mean_loglikelihood /= len(dataset)
        return mean_loglikelihood

    # return the loglikelihood for a single sequence (numpy matrix)
    def loglikelihood_helper(self, sample):
        alpha, c = self.forward(sample)
        loglikelihood = 0
        for c_t in c:
            loglikelihood += np.log(c_t)
            # print("c_t", c_t)
            # print("log(c_t)", np.log(c_t))
        # print("LL",loglikelihood)
        loglikelihood = -loglikelihood
        # print("LL",loglikelihood)
        return loglikelihood

    def normalize(self, matrix):
        for i in range(0, matrix.shape[0]):
            matrix[i] = self.normalize_row(matrix[i])
        return matrix

    # Normalizes a row in the matrix
    def normalize_row(self, row):
        return np.true_divide(row, np.sum(row))

    # return a prediction of next n words of a sequence by evaluating likelihoods
    # possible bug: because we are calculating *log* likelihoods we may want to minimize it instead of maximize it
    def predict_simple(self, sample, vocab_size, num_words_into_future):
        pred = np.copy(sample)
        for i in range(num_words_into_future):
            pred_step = self.predict_next_word(pred, vocab_size)
            pred = np.append(pred, pred_step)
            print("simple: {} out of {} words complete".format(i+1, num_words_into_future))
        return pred

    # return a prediction of the next word of a sequence
    def predict_next_word(self, sample, vocab_size):
        pred = 0
        pred_prob = float('-inf')
        for i in range(0, vocab_size):
            prob = self.loglikelihood_helper(np.append(sample, i))
            if prob > pred_prob:
                pred_prob = prob
                pred = i
        return pred

    # given a sequence of observations (single array of numbers) with blanks at the end (represented by -99),
    # find the most probable path of hidden states for given observations including blanks
    # and then predict best words that can go into the blanks
    def predict_with_viterbi(self, sample, num_words_into_future):
        sample_with_blanks = np.zeros(
            len(sample)+num_words_into_future, np.intc)
        for i in range(0, len(sample)):
            sample_with_blanks[i] = sample[i]
        for i in range(len(sample), len(sample)+num_words_into_future):
            sample_with_blanks[i] = 0  # blank added at end
        max_value, path_trace = self.viterbi(sample_with_blanks)
        most_recent_state = path_trace[len(path_trace)-1]
        # predict this many times into the future
        for i in range(0, num_words_into_future):
            random_num = np.random.rand(1)[0]
            # find most probable transition state
            best_state_to_go_into = np.argmax(
                self.transitions[most_recent_state])
            best_word_to_fill_blank = np.argmax(
                self.emissions[best_state_to_go_into])
            most_recent_state = best_state_to_go_into
            sample_with_blanks[len(sample)+i] = best_word_to_fill_blank
            print("viterbi: {} out of {} words complete".format(i+1, num_words_into_future))
        return sample_with_blanks

    # given a sequence of observations (single array of numbers), find the most probable path of hidden states it could have followed

    def viterbi(self, sample):
        # Dimenstions of v are TxN where T=number of observations, N=number of hidden states
        v = np.zeros((len(sample), self.num_states))
        # Dimenstions of backpointer are TxN where T=number of observations, N=number of hidden states
        backpointer = np.zeros((len(sample), self.num_states))
        for t in range(0, len(sample)):
            for j in range(0, self.num_states):
                if(sample[0] == -99):  # blank then emission is random b/w 0 and 1
                    v[t][j] = self.pi[j] * \
                        np.random.rand(1)[0]  # pi_j * bj(o1)
                else:
                    v[t][j] = self.pi[j] * \
                        self.emissions[j][sample[0]]  # pi_j * bj(o1)
        max_v_prev = -99
        prev_state_selected = 0

        # Recursive Step
        for t in range(1, len(sample)):
            for j in range(0, self.num_states):
                # go through all states again to analyze states that pass through time t-1
                for i in range(0, self.num_states):
                    if(v[t-1][i]*self.transitions[i][j] > max_v_prev):  # we need max v_t-1 (i) *aij
                        max_v_prev = v[t-1][i]*self.transitions[i][j]
                        prev_state_selected = i

                # find best value of v for each hidden state in this time step
                if(sample[0] == -99):  # blank then emission israndom b/w 0 and 1
                    v[t][j] = max_v_prev * np.random.rand(1)[0]
                else:
                    #print(self.emissions.shape, sample[t])
                    v[t][j] = max_v_prev * self.emissions[j][sample[t]]

                backpointer[t][j] = prev_state_selected

        # Find value and indices of best v value for time T (final time)
        max_val = -99
        time = len(sample)-1  # final time
        best_state = 0
        for j in range(0, self.num_states):
            if (v[time][j] > max_val):
                max_val = v[time][j]
                best_state = j

        # intialize path trace array
        # preparing array to build path of hidden states to output
        path_trace = np.zeros(len(sample), np.intc)
        # start back trace by adding to the end, the best_state for for time T in previous state
        path_trace[len(path_trace)-1] = best_state
        # run backtrace
        index = len(path_trace)-2
        while(index >= 0):
            # backpointer[current_time][state of the next node in path]
            path_trace[index] = backpointer[time][path_trace[index+1]]
            time -= 1
            index -= 1
        return max_val, path_trace

    # given the integer representation of a single sequence
    # return a T x num_states matrix of alpha where T is the total number of tokens in a single sequence
    # and also return a T x 1 array of c for normalizing alpha, beta and calculating the log likelihood
    def forward(self, sample):
        alpha = np.zeros((len(sample), self.num_states))
        c = np.zeros((len(sample),))
        # initialization
        for j in range(0, self.num_states):
            alpha[0][j] = np.longdouble(
                self.pi[j] * self.emissions[j][sample[0]])
            # print("c[0]", c[0], "alpha[0][j]", alpha[0][j])
            c[0] += alpha[0][j]
        # print("c[0] before", c[0])
        c[0] = 1/c[0]
        # print("c[0] after", c[0])
        alpha[0] *= c[0]
        # print("alpha[0]", alpha[0])
        # recursion
        for t in range(1, len(sample)):
            for j in range(0, self.num_states):
                for i in range(0, self.num_states):
                    # print(t, j, sample[t])
                    alpha[t][j] += np.longdouble(alpha[t-1][i] *
                                                 self.transitions[i][j] * self.emissions[j][sample[t]])
                # print("c[t]", c[t], "alpha[t][j]", alpha[t][j])
                c[t] += alpha[t][j]
            # print("c[t] before", c[t])
            c[t] = 1/c[t]
            # print("c[t] after", c[t])
            alpha[t] *= c[t]
            # print("alpha["+str(t)+"]",alpha[t])

        return alpha, c

    # given the integer representation of a single sequence
    # return a T x num_state matrix of beta where T is the total number of tokens in a single sequence
    def backward(self, sample, c):
        beta = np.zeros((len(sample), self.num_states))
        # initialization
        beta[len(sample)-1] = c[len(sample)-1]

        # recursion
        for t in range(1, len(sample)):
            for i in range(0, self.num_states):
                for j in range(0, self.num_states):
                    beta[len(sample)-1-t][i] += np.longdouble(self.transitions[i][j] *
                                                              self.emissions[j][sample[len(sample)-t]] * beta[len(sample)-t][j])
            beta[len(sample)-1-t] *= c[len(sample)-1-t]
            # print("beta["+str(len(sample)-1-t)+"]", beta[len(sample)-1-t])
        # for alph in beta:
            # print("Beta rom sum ", np.sum(alph))
        return beta

    # Uses alpha and beta values to calculate
    # e[t][i][j] = Probability of being in state i at time t and state j at time t+1
    # y[t][j] = Probability of being in state j at time t
    def e_step(self, sample):
        # print("Sample is ", sample)
        alpha, c = self.forward(sample)
        beta = self.backward(sample, c)
        # print("Alpha Below ")
        # print(alpha)
        # print("Beta Below ")
        # print(beta)
        # print("C Below ")
        # print(c)

        y = np.zeros((len(sample), self.num_states))
        e = np.zeros((len(sample), self.num_states, self.num_states))
        # print(beta)
        for t in range(0, len(sample)-1):
            for j in range(0, self.num_states):
                y[t][j] = 0
                for i in range(0, self.num_states):
                    #print(len(sample)-1, t, j, i)
                    e[t][j][i] = (alpha[t][j] * self.transitions[j]
                                  [i] * self.emissions[i][sample[t+1]] * beta[t+1][i])
                    y[t][j] += e[t][j][i]

        for i in range(0, self.num_states):
            y[len(sample)-1][i] = alpha[len(sample)-1][i]
        # for yrow in y:
        #     print("Sum of row in y", np.sum(yrow))
        # for t in range(0, len(sample)-1):
        #     for erow in e[t]:
        #         print("Sum of row in e", np.sum(erow))
        # print("Y Below ")
        # print(y)
        # print("E Below ")
        # print(e)
        return y, e

    # Tunes transitions
    def tune_transitions(self, sample, y, e):
        for i in range(0, self.num_states):
            for j in range(0, self.num_states):
                num = 0
                den = 0
                for t in range(0, len(sample) - 1):
                    num += e[t][i][j]
                    for k in range(0, self.num_states):
                        den += e[t][i][k]
                self.transitions[i][j] = num/den

    def tune_transitions_new(self, sample, y, e):
        for i in range(0, self.num_states):
            den = 0
            for t in range(0, len(sample)-1):
                den += y[t][i]
            for j in range(0, self.num_states):
                num = 0
                for t in range(0, len(sample) - 1):
                    num += e[t][i][j]
                self.transitions[i][j] = num/den
        # self.normalize(self.transitions)

    # Tunes emissions
    def tune_emissions(self, sample, y, e):
        for j in range(0, self.num_states):
            den = 0
            for t in range(0, len(sample)):
                den += y[t][j]
            for vk in range(0, self.vocab_size):
                num = 0.00000000000001
                for t in range(0, len(sample)):
                    if vk == sample[t]:
                        num += y[t][j]
                self.emissions[j][vk] = num/den
        # self.normalize(self.emissions)

    def tune_emissions_new(self, sample, y, e):
        pseudocount = 0.0
        dist_inertia = 0.1
        for j in range(0, self.num_states):
            den = 0
            for t in range(0, len(sample)):
                den += y[t][j]
            for vk in range(0, self.vocab_size):
                num = pseudocount
                for t in range(0, len(sample)):
                    if vk == sample[t]:
                        num += y[t][j]
                self.emissions[j][vk] = num/den * \
                    (1 - dist_inertia) + self.emissions[j][vk] * dist_inertia

    # Uses the e and y matrices from the e_step to tune transition and emission probabilities
    def m_step(self, sample, y, e):
        for i in range(0, len(self.pi)):
            self.pi[i] = y[0][i]

        self.tune_transitions_new(sample, y, e)
        self.tune_emissions_new(sample, y, e)

    def compare_theta(self, prev, after):
        pos_count = 0
        neg_count = 0
        for i in range(0, len(prev)):
            for j in range(0, len(prev[i])):
                # if np.log(after[i][j]) - np.log(prev[i][j]) == 0:
                #     # print("NO CHANGE.")
                # else:
                #     print("YES CHANGE.")

                if np.log(after[i][j]) - np.log(prev[i][j]) > 0:
                    pos_count += 1
                else:
                    neg_count += 1
        return pos_count, neg_count

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, sample):
        # Takes out a sample from the dataset and does e_step and m_step
        # print("Before EM. Below are transitions followed by emissions")
        # print(self.transitions)
        # # print(self.emissions)
        # for transition in self.emissions:
        #     print("Sum of row in emissions", np.sum(transition))
        # mean_loglikelihood = 0.0
        # for i in range(0, sample_size):
        #     # rnd = random.randint(0, len(dataset)-1)  # Pick a random sample
        #     sample = dataset_flattened[i]

        transitions = np.copy(self.transitions)
        emissions = np.copy(self.emissions)
        # print("Started ", i+1, " out of ",
        #       sample_size, " samples in this iteration")
        y, e = self.e_step(sample)
        self.m_step(sample, y, e)
        # print("Completed ", i+1, " out of ",
        #       sample_size, " samples in this iteration")
        # print(
        #     "Transitions After Em on this sample during this iteration", self.transitions)
        # print(
        #     "Emissions After Em on this sample during this iteration", self.emissions)
        # mean_loglikelihood += self.loglikelihood_helper(dataset[i])

        # mean_loglikelihood = mean_loglikelihood/sample_size
        # return mean_loglikelihood
        # print("After EM. Below are transitions followed by emissions")
        # print(self.transitions)
        # print(self.emissions)

    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        pass

    def train(self, iterations, sample_size, dataset):
        loglikes = np.zeros((iterations,))
        epsilon = 0.01
        sample = []
        print(sample_size)
        for x in dataset:
            for string in x:
                sample.append(string)
        for i in range(0, iterations):
            print("Started ", i+1, " out of ", iterations, " iterations")
            self.em_step(sample)
            print("Completed ", i+1, " out of ", iterations, " iterations")
            loglike = self.loglikelihood_helper(sample)/len(dataset)
            if i > 1:
                if loglike - loglikes[i-1] < epsilon:
                    self.save(os.path.join(
                        "../modelFile3/", "model" + str(int(i/5))))
                    self.get_figure(
                        range(1, i+1), loglikes[0:i], 'Iteration', 'Log Likelihood', "train_plot_sample_size_"+str(sample_size)+"_hidden_states_"+str(self.num_states)+"_v2")
                    break
            loglikes[i] = loglike
            if i % 5 == 0:
                self.save(os.path.join("../modelFile3/", "model" + str(int(i/5))))
                self.get_figure(range(1, i+1), loglikes[0:i], 'Iteration', 'Log Likelihood', "train_plot_sample_size_"+str(sample_size)+"_hidden_states_"+str(self.num_states)+"_v2")
            print("Log Likelihoods:", loglike)

    def pred_accuracy(self, sample, prediction, num_words_into_future):
        correct = 0
        incorrect = 0
        for i in range(1, num_words_into_future+1):
            if sample[len(sample)-i] == prediction[len(sample)-i]:
                correct += 1
            else:
                incorrect += 1
        return correct/(correct+incorrect)

    def int_to_words(self, sample_int, int_to_word_map):
        sample = ""
        for i in sample_int:
            sample += int_to_word_map[i] + " "
        return sample

    def predict(self, test_data, vocab_size, int_to_word_map, num_words_into_future):
        acc_viterbi = 0
        acc_simple = 0
        for i, sample in enumerate(test_data):
            predicted_viterbi = self.predict_with_viterbi(sample[:len(sample)-num_words_into_future], num_words_into_future)
            # translated = translate_int_to_words(predicted, int_to_word_map)
            # print(translate_int_to_words(predicted[-5:], int_to_word_map))
            acc_viterbi += self.pred_accuracy(sample, predicted_viterbi, num_words_into_future)
            # predicted_simple = self.predict_simple(sample[:len(sample)-num_words_into_future], vocab_size, num_words_into_future)
            # acc_simple += self.pred_accuracy(sample, predicted_simple, num_words_into_future)
            print("{} out of {} samples complete".format(i+1, len(test_data)))
        return acc_viterbi/len(test_data), acc_simple/len(test_data)

    def get_figure(self, xvalues, yvalues, xaxisname, yaxisname, filename):
        fig = plt.figure()
        plt.plot(xvalues, yvalues)
        plt.xlabel(xaxisname)
        plt.ylabel(yaxisname)
        plt.savefig(os.path.join("../plots/", filename))

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)


def main():
    parser = argparse.ArgumentParser(
        description='Program to build and train a neural network.')
    parser.add_argument('--train_path', default=None,
                        help='Path to the training data directory.')
    parser.add_argument('--dev_path', default=None,
                        help='Path to the development data directory.')
    parser.add_argument('--model_path', default=None,
                        help='Path to model directory')
    parser.add_argument('--max_iters', type=int, default=1000,
                        help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--hidden_states', type=int, default=15,
                        help='The number of hidden states to use. (default 10)')
    parser.add_argument('--train_sample_size', type=int, default=100,
                        help='The max number of samples. (default 100)')
    parser.add_argument('--test_sample_size', type=int, default=50,
                        help='The max number of samples. (default 100)')
    parser.add_argument('--mode', type=int, default=1,
                        help='Modes. Testing is 2, training is 1, prediction is 0. (default 1)')
    args = parser.parse_args()

    # OVERALL PROJECT ALGORITHM:
    # 1. load training and testing data into memory
    #
    # 2. build vocabulary using training data ONLY
    #
    # 3. instantiate an HMM with given number of states -- initial parameters can
    #    be random or uniform for transitions and inital state distributions,
    #    initial emission parameters could bea uniform OR based on vocabulary
    #    frequency (you'll have to count the words/characters as they occur in
    #    the training data.)
    #
    # 4. output initial loglikelihood on training data and on testing data
    #
    # 5+. use EM to train the HMM on the training data,
    #     output loglikelihood on train and test after each iteration
    #     if it converges early, stop the loop and print a message

    postrain = os.path.join(args.train_path, 'pos')
    negtrain = os.path.join(args.train_path, 'neg')

    # Combine into list
    train_paths = [postrain, negtrain]

    word_vocab, int_to_word_map = build_vocab_words(
        train_paths, args.train_sample_size)

    vocab_size = len(word_vocab)

    if args.mode == 1:
        # Paths for positive and negative training data

        # Create vocab and get its size. word_vocab is a dictionary from words to integers. Ex: 'painful':2070
        # word_vocab, int_to_word_map = build_vocab_words(
        #     train_paths, args.sample_size)
        dataset_complete = load_and_convert_data_words_to_ints(
            train_paths, word_vocab, args.train_sample_size)
        dataset = dataset_complete
        # dataset = np.random.choice(dataset_complete, size=args.sample_size)
        # dataset = dataset_complete

        # Create model
        model = HMM(args.hidden_states, vocab_size, word_vocab)
        # sample_with_predictions_added = model.predict_with_viterbi(dataset[0], 5)
        # print(model.translate_int_to_words(
        # sample_with_predictions_added, int_to_word_map))
        # loglikelihood = model.loglikelihood(dataset)
        # print(loglikelihood)
        model.train(args.max_iters, len(dataset), dataset)
        # loglikelihood = model.loglikelihood(dataset)
        # print(loglikelihood)
        # print(int_to_word_map.get(0))
        # give it sample and a number. It will return a new sample with predicted words appended to the end.
        # model.predict_with_viterbi(sample, 5)
        # prediction_with_v = model.predict_with_viterbi(
        # dataset[2][0:len(dataset[2])-8], 5)
        # print(model.translate_int_to_words(prediction_with_v, int_to_word_map))
        #model.predict_with_viterbi(sample, 5)
        model.save(os.path.join("../modelFile3/", "model"))
    elif args.mode == 0:
        # Paths for positive and negative training data
        postest = os.path.join(args.dev_path, 'pos')
        negtest = os.path.join(args.dev_path, 'neg')
        test_paths = [postest, negtest]
        num_words_into_future = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Sort list of models
        model_list = os.listdir(args.model_path)
        final_model_exists = False
        if "model" in model_list:
            model_list.remove("model")
            final_model_exists = True
        model_list.sort(key=lambda x: int(x[5:]))
        if final_model_exists:
            model_list.append("model")  # "model" is the final model created
        filename = model_list[len(model_list)-1]
        model = HMM.load(os.path.join(args.model_path, filename))

        pred_accuracies_viterbi = [None] * len(num_words_into_future)
        pred_accuracies_simple = [None] * len(num_words_into_future)
        test_data = build_test_samples(test_paths, args.test_sample_size, word_vocab)
        # print(filename, model.emissions.shape)

        for idx, i in enumerate(num_words_into_future):
            pred_accuracies_viterbi[idx], pred_accuracies_simple[idx] = model.predict(test_data, vocab_size, int_to_word_map, i)
            print("Tested: {} pred_accuracy_viterbi: {} pred_accuracy_simple: {}".format(i, pred_accuracies_viterbi[idx], pred_accuracies_simple[idx]))

        model.get_figure(range(1, len(pred_accuracies_viterbi)+1),
                         pred_accuracies_viterbi, '5 x Nth Iteration', 'Viterbi Accuracy', "PredViterbi_sample_size_"+str(args.train_sample_size)+"_hidden_states_"+str(args.hidden_states))
        model.get_figure(range(1, len(pred_accuracies_simple)+1),
                         pred_accuracies_simple, '5 x Nth Iteration', 'Simple Prediction Accuracy', "PredSimple_sample_size_"+str(args.train_sample_size)+"_hidden_states_"+str(args.hidden_states))
        print(pred_accuracies_viterbi)
        # print(pred_accuracies_simple)
    elif args.mode == 2:
        # Paths for positive and negative training data
        postest = os.path.join(args.dev_path, 'pos')
        negtest = os.path.join(args.dev_path, 'neg')
        test_paths = [postest, negtest]

        # Sort list of models
        model_list = os.listdir(args.model_path)
        model_list.remove("model")
        model_list.sort(key=lambda x: int(x[5:]))
        model_list.append("model")  # "model" is the final model created
        dataset = build_test_samples(test_paths, args.test_sample_size, word_vocab)
        dataset_flattened = []
        test_LL = [None] * len(model_list)
        for x in dataset:
            for string in x:
                dataset_flattened.append(string)
        for i in range(0, len(model_list)):
            # Just testing on final model initially --> will actually be = model_list[i]
            filename = model_list[i]
            model = HMM.load(os.path.join(args.model_path, filename))
            test_LL[i] = model.loglikelihood_helper(dataset_flattened)/args.test_sample_size
            print("Tested: {} LL: {}".format(i, test_LL[i]))
        model.get_figure(range(1, len(test_LL)+1), test_LL, 'iterations', 'Log Likelihood per model', "test_LL_sample_size_"+str(args.train_sample_size)+"_hidden_states_"+str(args.hidden_states))


if __name__ == '__main__':
    main()

# CMD arg: python hmm.py --dev_path ../imdbFor246/test --model_path ../modelFile3--train_path ../imdbFor246/train --mode 0
