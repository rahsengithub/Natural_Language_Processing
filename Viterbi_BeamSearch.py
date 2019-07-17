# UID: 180128022


# =================================================================================================================
# THE BELOW CODE IS AN AMALGAMATION OF THE CODE PROVIDED FROM ASSIGNMENT 3 AND NEW CODE FOR ASSIGNMENT 4
# THE IMPLEMENTATION OF THE "Perceptron" CLASS HAS BEEN AVOIDED, INSTEAD, SEPARATE FUNCTIONS HAVE BEEN CONSIDERED
# AS A RESULT, ANY CHANGES THAT WERE REQUIRED (LIKE, REMOVING <self.x>), HAVE BEEN MADE
# =================================================================================================================

###imports

from collections import Counter
import sys
import numpy as np
import time, random
from sklearn.metrics import f1_score
import heapq

# ************ CODE PROVIDED FROM THE ASSIGNMENT 3 ************
random.seed(11242)

depochs = 5
feat_red = 0

print("\nDefault no. of epochs: ", depochs)
print("\nDefault feature reduction threshold: ", feat_red)

print("\nLoading the data \n")

"""Loading the data"""


# ************ CODE PROVIDED FROM ASSIGNMENT 3 ************
### Load the dataset
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)


train_data = load_dataset_sents(sys.argv[2])
test_data = load_dataset_sents(sys.argv[3])

## unique tags
all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

""" Defining our feature space """

print("\nDefining the feature space \n")


# ************ CODE PROVIDED FROM ASSIGNMENT 3 ************
# feature space of cw_ct
def cw_ct_counts(data, freq_thresh=5):  # data inputted as (cur_word, cur_tag)

    cw_c1_c = Counter()

    for doc in data:
        cw_c1_c.update(Counter(doc))

    return Counter({k: v for k, v in cw_c1_c.items() if v > freq_thresh})


# ************ CODE PROVIDED FROM ASSIGNMENT 3 ************
# feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_count):  # sent as (cur_word, cur_tag)
    for each in sent:
        if each not in cw_ct_count.keys():
            count = 0
        else:
            count = 1
    return count


# ************ CODE PROVIDED FROM ASSIGNMENT 3 ************
def train_perceptron(data, cw_ct_count, epochs, shuffle=True):
    # variables used as metrics for performance and accuracy
    iterations = range(len(data) * epochs)
    false_prediction = 0
    false_predictions = []

    # initialising our weights dictionary as a counter
    # counter.update allows addition of relevant values for keys
    # a normal dictionary replaces the key-value pair
    weights = Counter()

    start = time.time()

    # multiple passes
    for epoch in range(epochs):
        false = 0
        now = time.time()

        # going through each sentence-tag_seq pair in training_data

        # shuffling if necessary
        if shuffle == True:
            random.shuffle(data)

        for doc in data:

            # retrieve the highest scoring sequence
            max_scoring_seq = scoring(doc, weights, cw_ct_count)

            # if the prediction is wrong
            if max_scoring_seq != doc:
                correct = Counter(doc)

                # negate the sign of predicted wrong
                predicted = Counter({k: -v for k, v in Counter(max_scoring_seq).items()})

                # add correct
                weights.update(correct)

                # negate false
                weights.update(predicted)

                """Recording false predictions"""
                false += 1
                false_prediction += 1
            false_predictions.append(false_prediction)

        print("Epoch: ", epoch + 1,
              " / Time for epoch: ", round(time.time() - now, 2),
              " / No. of false predictions: ", false)

    return weights, false_predictions, iterations


# ************ CODE PROVIDED FROM ASSIGNMENT 3 ************
# testing the learned weights
def test_perceptron(data, weights, cw_ct_count):
    correct_tags = []
    predicted_tags = []

    i = 0

    for doc in data:
        _, tags = list(zip(*doc))

        correct_tags.extend(tags)

        max_scoring_seq = scoring(doc, weights, cw_ct_count)

        _, pred_tags = list(zip(*max_scoring_seq))

        predicted_tags.extend(pred_tags)

    return correct_tags, predicted_tags


# ************ NEW CODE FOR ASSIGNMENT 4 ************
# function to get the maximum scoring sequence
def scoring(doc, weights, cw_ct_count):
    sentence, tags = list(zip(*doc))  # unzipping the document
    if sys.argv[1] == "-v":  # condition based on the commandline argument
        max_scoring_seq = viterbi_func(sentence, cw_ct_count, weights)  # calling the function for Viterbi
        max_scoring_seq = [(words, tags) for words, tags in zip(sentence, max_scoring_seq)]
    else:  # else calling the function for Beam Search
        max_scoring_seq = beam_func(sentence, cw_ct_count, weights, k=1)  # passing the k value as 1 by default
        max_scoring_seq = [(words, tags) for words, tags in zip(sentence, max_scoring_seq)]
    return max_scoring_seq  # returning the maximum scoring sequence


# ************ NEW CODE FOR ASSIGNMENT 4 ************
# function for Viterbi
def viterbi_func(sent, cw_ct_count, weights):
    rows = len(all_tags)  # calculating the number of the tags
    columns = len(sent)  # calculating the number of sentences
    viterbi_score_matrix = np.zeros((rows, columns),
                                    dtype=float)  # creating a Viterbi matrix of 0s with the size specified
    backpointer_index_matrix = np.zeros((rows, columns), dtype=float)  # creating a backpointer matrix
    for i in range(0, columns):  # iterating over each element in the matrix
        for j in range(0, rows):
            list_of_tuples = [(sent[i], all_tags[j])]  # making a list of tuples with the values
            score = phi_1(list_of_tuples, cw_ct_count)  # calling the phi_1 function to get the count
            if i == 0:  # for the first column of the matrix
                viterbi_score_matrix[j][i] = score * weights[
                    (sent[i], all_tags[j])]  # storing the value after calculation
            else:  # for all other columns
                prev_col_max = np.amax(
                    viterbi_score_matrix[:, i - 1])  # calculating the maximum value of the preceding column
                viterbi_score_matrix[j][i] = prev_col_max + score * weights[
                    (sent[i], all_tags[j])]  # storing the value after calculation
                backpointer_index_matrix[j][i] = np.argmax(
                    viterbi_score_matrix[i - 1])  # storing the index of the maximum value
    index_of_max_col = np.argmax(viterbi_score_matrix, axis=0)  # getting the maximum value index list
    seq_of_tags = [all_tags[a] for a in index_of_max_col]  # getting the sequence of tags for the indices
    return seq_of_tags  # returning the tag sequences


# ************ NEW CODE FOR ASSIGNMENT 4 ************
# function for Beam Search
def beam_func(sent, cw_ct_count, weights, k):  # taking any value of beam through the beam size, k
    rows = len(all_tags)  # calculating the number of the tags
    columns = len(sent)  # calculating the number of sentences
    beam_score_matrix = np.zeros((rows, columns), dtype=float)  # creating a Beam search matrix
    for i in range(0, columns):  # iterating over each element in the matrix
        for j in range(0, rows):
            list_of_tuples = [(sent[i], all_tags[j])]  # making a list of tuples with the values
            score = phi_1(list_of_tuples, cw_ct_count)  # calling the phi_1 function to get the count
            if i == 0:  # for the first column of the matrix
                beam_score_matrix[j][i] = score * weights[(sent[i], all_tags[j])]  # storing the value after calculation
            else:  # for all other columns
                # calculating the top k maximum score for previous column and storing the indices
                previous_top_values_index_list = heapq.nlargest(k, range(len(beam_score_matrix[:, i - 1])),
                                                                beam_score_matrix[:, i - 1].__getitem__)
                prev_column_max = beam_score_matrix[:, i - 1][
                    previous_top_values_index_list[0]]  # making the first value as the maximum
                for m in range(0, len(previous_top_values_index_list)):  # iterating through the index list
                    if beam_score_matrix[:, i - 1][
                        previous_top_values_index_list[m]] > prev_column_max:  # implementing a flag
                        prev_column_max = beam_score_matrix[:, i - 1][
                            previous_top_values_index_list[m]]  # equalising the values
                beam_score_matrix[j][i] = prev_column_max + score * weights[
                    (sent[i], all_tags[j])]  # storing the value after calculation
    index_of_max_col = np.argmax(beam_score_matrix, axis=0)  # getting the maximum value index list
    seq_of_tags = [all_tags[a] for a in index_of_max_col]  # getting the sequence of tags for the indices
    return seq_of_tags  # returning the tag sequences


# ************ CODE PROVIDED FROM ASSIGNMENT 3 ************
def evaluate(correct_tags, predicted_tags):
    f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=all_tags[1:])

    print("F1 Score: ", round(f1, 5))

    return f1


# ************ CODE PROVIDED FROM ASSIGNMENT 3 ************
cw_ct_count = cw_ct_counts(train_data, freq_thresh=feat_red)

print("\nTraining the perceptron with (cur_word, cur_tag) \n")

weights, false_predictions, iterations = train_perceptron(train_data, cw_ct_count, epochs=depochs, shuffle=True)

print("\nEvaluating the perceptron with (cur_word, cur_tag) \n")

correct_tags, predicted_tags = test_perceptron(test_data, weights, cw_ct_count)

f1 = evaluate(correct_tags, predicted_tags)

print("\nTraining the perceptron with (cur_word, cur_tag) & (prev_tag, current_tag) \n")
