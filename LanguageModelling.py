# UID : 180128022
# importing packages
import time
import re
from collections import Counter
import sys


# defining the global variables used
file_train = sys.argv[1]  # training data
file_test = sys.argv[2]  # testing data
dictionary = {}  # dictionary for storing frequency of words
compare_word_list = []  # list for storing the words returned after probability conditions for Unigram
# pre-defined list of correct words for Unigram
correct_list = ["whether", "through", "piece", "court", "allowed", "check", "hear", "cereal", "chews", "sell"]
# pre-defined list of correct words for Bigram
correct_bigram_list = ["know whether", "went through", "a piece", "to court", "only allowed", "to check", "you hear",
                       "eat cereal", "normally chews", "to sell"]
words_before_line = []  # list to store the words before '____' for Bigram
words_after_line = [] # list to store the words after '____' for Bigram
merged_first_list_before = []  # list to store words_before_list + first suggestion word for Bigram
merged_second_list_before = []  # list to store words_before_list + second suggestion word for Bigram
compare_bigram_word_list = []  # list for storing the words returned after probability conditions for Bigram
merged_first_list_after = []
merged_second_list_after = []
# declaring like above for smoothing function
words_before_line_smooth = []
words_after_line_smooth = []
merged_first_list_before_smooth = []
merged_second_list_before_smooth = []
compare_bigram_word_list_smooth = []
merged_first_list_after_smooth = []
merged_second_list_after_smooth = []



# function for pre-processing activity
def ngrams_generation(text, n_gram):
    # replacing all non-alphanumeric characters with spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = [token for token in text.split(" ") if token != ""]  # tokenising
    ngrams = zip(*[tokens[i:] for i in range(n_gram)])  # generating n-grams
    return [" ".join(ngram) for ngram in ngrams]  # concatenating tokens


# function for operating on the training data
def ngram_func(train_file, n):
    ngram_training_set = {}
    with open(train_file, "r") as f:
        line = f.read()
        line = line.lower()
        dictionary = Counter(ngrams_generation(line, n))  # calling the above function to find the frequency of tokens
        for keyindict, valueindict in sorted(dictionary.items()):
            probability = valueindict  # updating the probability of each value
            ngram_training_set[keyindict] = probability  # assigning each value of probability to the respective word
    return ngram_training_set  # returning the dictionary of words and their probabilities


# function for operating on the test data for Unigram
def unigram_test_read_func(test_file, uni_train_set):
    with open(test_file, "r") as f:
        line = f.read()
        line = line.lower()
        list_of_words = re.sub(r".*: ", "", line)  # extracting two suggestion words
        list_of_words = list_of_words.replace("/", " ")  # separating the two extracted sets
        # defining the list for first and second groups of suggestion words
        first_list = re.sub(r" .*", "", list_of_words).split("\n")[:-1]
        second_list = re.sub(r".* ", "", list_of_words).split("\n")[:-1]
        for i in range(0, len(first_list)):  # iterating through the list of words
            # assigning the probability of the words the list and the same words in the training set
            p1 = uni_train_set[first_list[i]]
            p2 = uni_train_set[second_list[i]]
            if p1 == p2:
                compare_word_list.append("No error")
            elif p1 > p2:
                compare_word_list.append(first_list[i])  # appending as required
            else:
                compare_word_list.append(second_list[i])
        words = set(correct_list) & set(compare_word_list)  # matching the result against the pre-defined correct words
        value = Counter(words)  # counting the occurrences
        accuracy = sum(value.values()) / 10  # calculating accuracy
        print("The accuracy for Unigram is", accuracy)
        return accuracy  # returning Unigam accuracy


# function for operating on the test data for Bigram
def bigram_test_read_func(test_file, bi_train_set, uni_train_set):
    with open(test_file, "r") as f:
        line = f.read()
        line = line.lower()
        list_of_words = re.sub(r".*: ", "", line)
        list_of_words = list_of_words.replace("/", " ")
        first_list = re.sub(r" .*", "", list_of_words).split("\n")[:-1]
        second_list = re.sub(r".* ", "", list_of_words).split("\n")[:-1]
        line = line.split("\n")
        for l in range(0, len(line)):
            if line[l] == "":  # skipping if the line is empty
                continue
            text = line[l].split('____')[0].split()[-1]  # extracting the words before "____"
            words_before_line.append(text)  # making a list of the above extracted words
        for l in range(0, len(line)):
            if line[l] == "":  # skipping if the line is empty
                continue
            text_new = line[l].split('____')[1].split()[0]  # extracting the words after "____"
            words_after_line.append(text_new)  # making a list of the above extracted words

        for i in range(0, len(first_list)):
            merged_first_list_before.append(words_before_line[i] + " " + first_list[i])  # appending the pair of words
        for i in range(0, len(second_list)):
            merged_second_list_before.append(words_before_line[i] + " " + second_list[i])

        for i in range(0, len(first_list)):
            merged_first_list_after.append(first_list[i] + " " + words_after_line[i])  # appending the pair of words
        for i in range(0, len(second_list)):
            merged_second_list_after.append(second_list[i] + " " + words_after_line[i])
        # initialising blank lists to be used later
        p1_list_before = []
        p2_list_before = []
        for i in range(0, len(words_before_line)):
            # condition setting
            if merged_first_list_before[i] not in bi_train_set:
                p1 = 0  # assigning 0 if word is not found
                p1_list_before.append(p1)
            else:
                # setting the first probability as the probability of the above group of words in the test file
                # divided by the probability of the word appearing before "____" in the training file for each iteration
                p1 = bi_train_set[merged_first_list_before[i]] / uni_train_set[words_before_line[i]]
                p1_list_before.append(p1)  # saving the result in the above declared list
        # same as above for the second list
        for i in range(0, len(merged_second_list_before)):
            if merged_second_list_before[i] not in bi_train_set:
                p2 = 0
                p2_list_before.append(p2)
            else:
                p2 = bi_train_set[merged_second_list_before[i]] / uni_train_set[words_before_line[i]]
                p2_list_before.append(p2)

        p1_list_after = []
        p2_list_after = []
        for i in range(0, len(words_after_line)):
            # condition setting
            if merged_first_list_after[i] not in bi_train_set:
                p1 = 0  # assigning 0 if word is not found
                p1_list_after.append(p1)
            else:
                # setting the first probability as the probability of the above group of words in the test file
                # divided by the probability of the word appearing after "____" in the training file for each iteration
                p1 = bi_train_set[merged_first_list_after[i]] / uni_train_set[words_after_line[i]]
                p1_list_after.append(p1)  # saving the result in the above declared list
        # same as above for the second list
        for i in range(0, len(merged_second_list_after)):
            if merged_second_list_after[i] not in bi_train_set:
                p2 = 0
                p2_list_after.append(p2)
            else:
                p2 = bi_train_set[merged_second_list_after[i]] / uni_train_set[words_after_line[i]]
                p2_list_after.append(p2)

        for i in range(0, len(merged_first_list_before)):
            # multiplying the probabilities of both the possible biagrams
            if p1_list_before[i] * p1_list_after[i] == p2_list_before[i] * p2_list_after[i]:
                compare_bigram_word_list.append("No error")  # for equality condition, append nothing
            elif p1_list_before[i] * p1_list_after[i] > p2_list_before[i] * p2_list_after[i]:
                compare_bigram_word_list.append(merged_first_list_before[i])
            else:
                compare_bigram_word_list.append(merged_second_list_before[i])
        words = set(correct_bigram_list) & set(compare_bigram_word_list)
        value = Counter(words)
        accuracy = sum(value.values()) / 10
        print("The accuracy for Bigram is", accuracy)
        return accuracy  # returning Bigram accuracy


# function for operating on the test data for Bigram with Add - 1 Smoothing
def smooth_bigram_test_read_func(test_file, bi_train_set, uni_train_set):
    with open(test_file, "r") as f:
        line = f.read()
        line = line.lower()
        list_of_words = re.sub(r".*: ", "", line)
        list_of_words = list_of_words.replace("/", " ")
        first_list = re.sub(r" .*", "", list_of_words).split("\n")[:-1]
        second_list = re.sub(r".* ", "", list_of_words).split("\n")[:-1]
        line = line.split("\n")
        for l in range(0, len(line)):
            if line[l] == "":  # skipping if the line is empty
                continue
            text = line[l].split('____')[0].split()[-1]  # extracting the words before "____"
            words_before_line_smooth.append(text)  # making a list of the above extracted words
        for l in range(0, len(line)):
            if line[l] == "":  # skipping if the line is empty
                continue
            text_new = line[l].split('____')[1].split()[0]  # extracting the words after "____"
            words_after_line_smooth.append(text_new)  # making a list of the above extracted words

        for i in range(0, len(first_list)):
            merged_first_list_before_smooth.append(
                words_before_line_smooth[i] + " " + first_list[i])  # appending the pair of words
        for i in range(0, len(second_list)):
            merged_second_list_before_smooth.append(words_before_line_smooth[i] + " " + second_list[i])

        for i in range(0, len(first_list)):
            merged_first_list_after_smooth.append(
                first_list[i] + " " + words_after_line_smooth[i])  # appending the pair of words
        for i in range(0, len(second_list)):
            merged_second_list_after_smooth.append(second_list[i] + " " + words_after_line_smooth[i])
        # initialising blank lists to be used later
        p1_list_before_smooth = []
        p2_list_before_smooth = []
        for i in range(0, len(words_before_line_smooth)):
            # condition setting
            if merged_first_list_before_smooth[i] not in bi_train_set:
                p1 = 1 / (uni_train_set[words_before_line_smooth[i]] + sum(
                    uni_train_set.values()))  # assigning 0 if word is not found
                p1_list_before_smooth.append(p1)
            else:
                # setting the first probability as the probability of the above group of words in the test file
                # divided by the probability of the word appearing before "____" in the training file for each iteration
                p1 = bi_train_set[merged_first_list_before_smooth[i]] + 1 / (
                            uni_train_set[words_before_line_smooth[i]] + sum(uni_train_set.values()))
                p1_list_before_smooth.append(p1)  # saving the result in the above declared list
        # same as above for the second list
        for i in range(0, len(merged_second_list_before_smooth)):
            if merged_second_list_before_smooth[i] not in bi_train_set:
                p2 = 1 / (uni_train_set[words_before_line_smooth[i]] + sum(uni_train_set.values()))
                p2_list_before_smooth.append(p2)
            else:
                p2 = bi_train_set[merged_second_list_before_smooth[i]] + 1 / (
                            uni_train_set[words_before_line_smooth[i]] + sum(uni_train_set.values()))
                p2_list_before_smooth.append(p2)

        p1_list_after_smooth = []
        p2_list_after_smooth = []
        for i in range(0, len(words_after_line_smooth)):
            # condition setting
            if merged_first_list_after_smooth[i] not in bi_train_set:
                p1 = 1 / (uni_train_set[words_after_line_smooth[i]] + sum(
                    uni_train_set.values()))  # assigning 0 if word is not found
                p1_list_after_smooth.append(p1)
            else:
                # setting the first probability as the probability of the above group of words in the test file
                # divided by the probability of the word appearing after "____" in the training file for each iteration
                p1 = bi_train_set[merged_first_list_after_smooth[i]] + 1 / (
                            uni_train_set[words_after_line_smooth[i]] + sum(uni_train_set.values()))
                p1_list_after_smooth.append(p1)  # saving the result in the above declared list
        # same as above for the second list
        for i in range(0, len(merged_second_list_after_smooth)):
            if merged_second_list_after_smooth[i] not in bi_train_set:
                p2 = 1 / (uni_train_set[words_after_line_smooth[i]] + sum(uni_train_set.values()))
                p2_list_after_smooth.append(p2)
            else:
                p2 = bi_train_set[merged_second_list_after_smooth[i]] + 1 / (
                            uni_train_set[words_after_line_smooth[i]] + sum(uni_train_set.values()))
                p2_list_after_smooth.append(p2)
        for i in range(0, len(merged_first_list_before_smooth)):
            # multiplying the probabilities of both the possible biagrams
            if p1_list_before_smooth[i] * p1_list_after_smooth[i] > p2_list_before_smooth[i] * p2_list_after_smooth[i]:
                compare_bigram_word_list_smooth.append(merged_first_list_before_smooth[i])
            else:
                compare_bigram_word_list_smooth.append(merged_second_list_before_smooth[i])
        words = set(correct_bigram_list) & set(compare_bigram_word_list_smooth)
        value = Counter(words)
        accuracy = sum(value.values()) / 10
        print("The accuracy for Bigram after smoothing is", accuracy)
        return accuracy  # returning Bigram with Add - 1 smoothing


# creating objects and passing the appropriate arguments to the above functions

unigram_trainer = ngram_func(file_train, 1)
unitester = unigram_test_read_func(file_test, unigram_trainer)
bigram_trainer = ngram_func(file_train, 2)
bitester = bigram_test_read_func(file_test, bigram_trainer, unigram_trainer)
smooth_trainer = ngram_func(file_train, 2)
smooth = smooth_bigram_test_read_func(file_test, smooth_trainer, unigram_trainer)

