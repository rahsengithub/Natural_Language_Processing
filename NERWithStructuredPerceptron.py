# UID:180128022
# importing all the libraries
import copy
from collections import Counter
import itertools
from sklearn.metrics import f1_score
from random import shuffle
from random import seed
import sys

seed(180128022)
# function to obtain word and tag sequences for each sentence
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

# function to return the counts of current word-current label
def word_label_phi_1(trainset):
    word_tag = list(itertools.chain.from_iterable(trainset)) # list of tuples of the trainset
    c = Counter(word_tag) # counting the frequency of the word_label
    cw_cl_counted = {} # empty dictionary to store the values of the frequency
    for word, tag in c.items(): #iterating over all the keys and values of the counter items
        #if tag >= 3:
        cw_cl_counted.update({word[0] + "_" + word[1]: tag}) # updation condition
    # print(cw_cl_counted)
    return cw_cl_counted # returning the values

# function to break the training data into lists of sentences and labels
def sentence_label(trainset):
    sentence = [item[0] for item in trainset] # extracting the sentences
    label = [item[1] for item in trainset] # extracting the labels
    return sentence, label # returning the values

# function to return the dictionary with counts of 'cw_cl_counts' keys in the given sentence
def phi_1_func(sent_label, cw_cl_counts):
    newlist = [] # list to store the values
    for i in range(len(sent_label)): # iterating over the length of sent_label (x, y)
        if (i % 2 == 0): # checking if the length ov the list is divisible by 2
            list1 = sent_label[i] # extract the first list
            list2 = sent_label[i + 1] # extract the second list
            for n in range(len(list1)): # iterating over the length of the first list
                newlist.append([list1[n], list2[n]]) # appending the declared list with the values of both the lists
    word_tag_list = [] # list to store the values
    for every in newlist: # every element in the newlist
        word_tag = every[0] + "_" + every[1] # combining two words with an underscore
        word_tag_list.append(word_tag) # appending the values
    phi_1 = dict(Counter(word_tag_list))
    for x in phi_1: # checking all the values in the dictionary
        if x not in cw_cl_counts: # if they are not present in the cw_cl_counts
            phi_1[x] = 0 # assigning 0
    return phi_1 # returning the values




# function to train and return the weights
def train(trainset, cw):
    weight_dict = {} # weight dictionary declaration
    for key in cw: # checking the keys of cw_cl_counts
        weight_dict[key] = 0 # updating the weight dictionary keys with the same values of keys of cw_cl_counts and setting the values as 0
    possible_tags = ["O", "PER", "LOC", "ORG", "MISC"] # list of all the possible tags
    average_list = []
    for iter in range(0, 5): # number of epochs
        print("Iteration number", iter)
        wrong_answers = 0
        combination_possible_tags = [] # possible combination tags list declaration
        shuffle(trainset) # shuffling the training data

        for l in range(1, 6): # taking the range to be 6 as the maximum length of a sentence is 6
            tags = itertools.combinations_with_replacement(possible_tags, l) # combination of all the possible tags
            combination_possible_tags.append(tags) # appending the values to the list
        for sentence in trainset: # for every sentence in the training data
            sent = len(sentence) # checking for the length of each sentence
            combinations = [m for m in itertools.product(possible_tags, repeat=sent)] # storing every combination
            all_scores = [] # list to store all scores
            mysent = [sen[0] for sen in sentence] # list of words in a sentence
            actual_y = [sen[1] for sen in sentence] # list of actual tags in a sentence
            for i in combinations: # iterating over every combination
                i = list(i)
                score = 0
                phi_caller = phi_1_func([mysent, i], cw) # calling the phi_1_function

                for key, val in phi_caller.items(): # iterating over the returned dictionary
                    if key not in weight_dict: # condition
                        continue
                    else:
                        score += val*weight_dict[key] # else, update the score
                all_scores.append(score) # appending all the score values in to the list
            max_index = all_scores.index(max(all_scores)) # index of the maximum value of the score
            predicted_y = list(combinations[max_index]) # y_hat
            if (actual_y != predicted_y): # structured perceptron
                wrong_answers += 1
                phi_actual = phi_1_func([mysent, actual_y], cw)
                phi_predict = phi_1_func([mysent, predicted_y], cw)

                for ac in phi_actual:
                    if ac in weight_dict.keys():
                        weight_dict[ac] += phi_actual[ac]  # w + Φ(x, y)

                for pr in phi_predict:
                    if pr in weight_dict.keys():
                        weight_dict[pr] -= phi_predict[pr]  # w - Φ(x, ŷ)

        average_list.append(copy.deepcopy(weight_dict)) # appending weights to a list
        add = Counter()  # taking Counter items
        count = Counter()
        for each in average_list:  # iterating over the entire list
            add.update(each)  # updating the Counter items
            count.update(each.keys())

        average = {y: float(add[y]) / count[y] for y in add.keys()} # calculate the average

    return average # return the values


# function to return a predicted tag sequence
def predict(trained_weights, cw, test):
    possible_tags = ["O", "PER", "LOC", "ORG", "MISC"]
    combination_possible_tags = []
    for l in range(1, 6):
        tags = itertools.combinations_with_replacement(possible_tags, l)
        combination_possible_tags.append(tags)
    pred_y = [] # predicted_y list
    corr_y = [] # corrected_y list
    for j in test: # for every test data
        sent = len(j) # length of the test data
        combinations = [m for m in itertools.product(possible_tags, repeat=sent)] # find out the combiinations
        all_scores = []
        mysent = [sen[0] for sen in j] # get the sentences
        for i in combinations:
            i = list(i)
            score = 0
            phi_caller = phi_1_func([mysent, i], cw)
            for key, val in phi_caller.items(): # iterating over the dictionary
                if key not in trained_weights: # do nothing
                    continue
                else:
                    score += val * trained_weights[key] # else, update the score
            all_scores.append(score) # append all the scores
        max_index = all_scores.index(max(all_scores))  # find out the maximum score value index
        test_y = list(combinations[max_index]) # list all the values of the maximum combinations
        pred_y.append(test_y) # append to pred_y
        word, tag = sentence_label(j)
        corr_y.append(tag) # append to corr_y
    pred_y_list = [item for sublist in pred_y for item in sublist] # flatten the lists for pred_y
    corr_y_list = [item for sublist in corr_y for item in sublist] # flatten the lists for corr_y
    return(corr_y_list, pred_y_list) # return the values

# function to get the f1 measure
def test(correct, predicted):
    f1_micro = f1_score(correct, predicted,  average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    print("F1 Measure", f1_micro) # print the score

# function to get the top 10 for each tag
def top_10(trainer):
    possible_tags = ["O", "PER", "LOC", "ORG", "MISC"]  # list of all the possible tags
    list_dict = []  # empty list
    for each in possible_tags:
        empty_dict = {} # empty dictionary
        for key, val in trainer.items():
            if key.endswith("_" + each): # condition matching
                empty_dict.update({key: val}) # append the elements to the declared list
        list_dict.append(empty_dict)
    for d in list_dict:
        sorted_key_val = sorted(d.items(), key = lambda key: key[1], reverse = True)[:10] #sorted
        top10 = list(d.keys())[0].split("_")[1] # top 10 values
        print("The top 10 features of " + top10 + " are\n", sorted_key_val)



#function to generate n-grams
def ngrams_generation(tokens, n_gram):
    ngrams = zip(*[tokens[i:] for i in range(n_gram)]) # generating n-grams
    return [" ".join(ngram) for ngram in ngrams] # concatenating tokens

# function to get the counts of previous label-current label
def word_label_phi_2(trainset):
    list_of_tuples = list(itertools.chain.from_iterable(trainset)) # getting the list of tuples
    new_list = [x[1] for x in list_of_tuples] # extracting every tag
    tag_list = ['_'.join(x) for x in zip(new_list[0::2], new_list[1::2])] # join two alternate values using an underscore
    cs_cl_counted = dict(Counter(tag_list)) # Dictionary of the tag_ist with the required frequency
    return cs_cl_counted # return the values

# function to calculate the phi_2 value
def phi_2_func(sent_label, phi_1_dict, cs_cl_counts): # takes in phi_1 dict as an argument to merge it with the returned dictionary
    # print(cs_cl_counts)
    new_sent = []
    for each in sent_label:
        n = ["NONE"]
        new_sent.append(n + each) # adding 'NONE' in front of words and tags
    bigram = ngrams_generation(new_sent[1], 2) # generating bigrams by passing ngram = 2
    c = Counter(bigram) # counting the frequencies of bigram
    bi_dict = {}
    for key, value in c.items():
        key = key.replace(" ", "_") # adding underscore between keys of the bigram counter object
        if key not in cs_cl_counts: # condition
            bi_dict.update({key: 0}) # appropriate updation
        else:
            bi_dict.update({key: value})
    bi_dict = merge_dictionaries(bi_dict, phi_1_dict) # merging phi_1 and phi_2 dictionaries
    return bi_dict # return the value


# function to merge dictionaries
def merge_dictionaries(*dict_args):
    merged_dict = {}
    for d in dict_args:
        merged_dict.update(d)
    return merged_dict # return the merged dict

# function to train and return the weights
def train_phi_2(trainset, cw, ww):
    weight_dict = {} # weight dictionary declaration
    for key in cw: # checking the keys of cw_cl_counts
        weight_dict[key] = 0  # updating the weight dictionary keys with the same values of keys of cs_cl_counts and setting the values as 0
    for key in ww:  # checking the keys of cw_cl_counts
        weight_dict[key] = 0   # updating the weight dictionary keys with the same values of keys of cw_cl_counts and setting the values as 0
    possible_tags = ["O", "PER", "LOC", "ORG", "MISC"] # list of all the possible tags
    average_list = []
    for iter in range(0, 5): # number of epochs
        print("Iteration number", iter)
        wrong_answers = 0
        combination_possible_tags = [] # possible combination tags list declaration
        shuffle(trainset) # shuffling the training data

        for l in range(1, 6): # taking the range to be 6 as the maximum length of a sentence is 6
            tags = itertools.combinations_with_replacement(possible_tags, l) # combination of all the possible tags
            combination_possible_tags.append(tags) # appending the values to the list
        for sentence in trainset: # for every sentence in the training data
            sent = len(sentence) # checking for the length of each sentence
            combinations = [m for m in itertools.product(possible_tags, repeat = sent)]
            all_scores = [] # list to store all scores
            mysent = [sen[0] for sen in sentence] # list of words in a sentence
            actual_y = [sen[1] for sen in sentence] # list of actual tags in a sentence
            for i in combinations: # iterating over every combination
                i = list(i)
                score = 0
                dict_phi_1= phi_1_func([mysent, i], ww) # calling the phi_1_function to merge the dictionaries
                phi_caller = phi_2_func([mysent, i], dict_phi_1, cw) # calling the phi_2_function

                for key, val in phi_caller.items(): # iterating over the returned dictionary
                    if key not in weight_dict: # condition
                        continue
                    else:
                        score += val*weight_dict[key] # else, update the score
                all_scores.append(score) # appending all the score values in to the list
            max_index = all_scores.index(max(all_scores)) # index of the maximum value of the score
            predicted_y = list(combinations[max_index]) # y_hat
            if (actual_y != predicted_y): # structured perceptron
                wrong_answers += 1
                dict_phi_1_ac = phi_1_func([mysent, actual_y], ww) # calling the phi_1 function with actual_y to get the merged values of both phi_1 and phi_2
                dict_phi_1_pr = phi_1_func([mysent, predicted_y], ww) # calling the phi_1 function with predicted_y to get the merged values of both phi_1 and phi_2
                phi_actual = phi_2_func([mysent, actual_y], dict_phi_1_ac, cw) # calling the phi_2 function with actual_y
                phi_predict = phi_2_func([mysent, predicted_y], dict_phi_1_pr, cw)  # calling the phi_2 function with predicted_y
                for ac in phi_actual:
                    if ac in weight_dict.keys():
                        weight_dict[ac] += phi_actual[ac] # w + Φ(x, y)

                for pr in phi_predict:
                    if pr in weight_dict.keys():
                        weight_dict[pr] -= phi_predict[pr] # w - Φ(x, ŷ)

        average_list.append(copy.deepcopy(weight_dict)) # as described in case of phi_1

        add = Counter()
        count = Counter()
        for each in average_list:
            add.update(each)
            count.update(each.keys())

        average = {y: float(add[y]) / count[y] for y in add.keys()}
        # print(average)
    return average # return the values

# function to return a predicted tag sequence
def predict_phi_2(trained_weights, cw, ww, test):
    possible_tags = ["O", "PER", "LOC", "ORG", "MISC"]
    combination_possible_tags = []
    for l in range(1, 6):
        tags = itertools.combinations_with_replacement(possible_tags, l)
        combination_possible_tags.append(tags)
    pred_y = [] # predicted_y list
    corr_y = [] # corrected_y list
    for j in test: # for every test data
        sent = len(j) # length of the test data
        combinations = [m for m in itertools.product(possible_tags, repeat=sent)]
        all_scores = []
        mysent = [sen[0] for sen in j]
        for i in combinations:
            i = list(i)
            score = 0
            phi_1_d = phi_1_func([mysent, i], ww)
            phi_caller = phi_2_func([mysent, i], phi_1_d, cw)
            for key, val in phi_caller.items(): # iterating over the dictionary
                if key not in trained_weights: # do nothing
                    continue
                else:
                    score += val * trained_weights[key] # else, update the score
            all_scores.append(score) # append all the scores
        max_index = all_scores.index(max(all_scores))  # find out the maximum score value index
        test_y = list(combinations[max_index]) # list all the values of the maximum combinations
        pred_y.append(test_y) # append to pred_y
        word, tag = sentence_label(j)
        corr_y.append(tag) # append to corr_y
    pred_y_list = [item for sublist in pred_y for item in sublist] # flatten the lists for pred_y
    corr_y_list = [item for sublist in corr_y for item in sublist] # flatten the lists for corr_y
    # print(corr_y_list, pred_y_list)
    return(corr_y_list, pred_y_list)



# function to get the f1 measure
def test_phi_2(correct, predicted):
    f1_micro = f1_score(correct, predicted,  average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    print("F1 Measure", f1_micro) # print the score

# function to get the top 10 for each tag
def top_10_phi_2(trainer):
    possible_tags = ["O", "PER", "LOC", "ORG", "MISC"]  # list of all the possible tags
    list_dict = []
    for each in possible_tags:  # as described in case of phi_1
        empty_dict = {}
        for key, val in trainer.items():
            if key.endswith("_" + each):
                empty_dict.update({key: val})
        list_dict.append(empty_dict)
    for d in list_dict:
        sorted_key_val = sorted(d.items(), key = lambda key: key[1], reverse = True)[:10]
        top10 = list(d.keys())[0].split("_")[1]
        print("The top 10 features of " + top10 + " are\n", sorted_key_val)



train_data = load_dataset_sents(sys.argv[1])
test_data = load_dataset_sents(sys.argv[2])


print("For Phi_1, below are the values for 5 iterations...")
cw_cl = word_label_phi_1(train_data)
training_phi_1 = train(train_data, cw_cl)
corrected, predicted = predict(training_phi_1, cw_cl, test_data)
testing = test(corrected, predicted)
top10 = top_10(training_phi_1)

print("For (Phi_1+Phi_2), below are the values for 5 iterations...")
# cw_cl = word_label_phi_1(train_data)
cs_cl = word_label_phi_2(train_data)
training_phi_2 = train_phi_2(train_data, cs_cl, cw_cl)
corrected_phi_2, predicted_phi_2 = predict_phi_2(training_phi_2, cs_cl,cw_cl, test_data)
testing_phi_2 = test_phi_2(corrected_phi_2, predicted_phi_2)
top10_phi_2 = top_10_phi_2(training_phi_2)
