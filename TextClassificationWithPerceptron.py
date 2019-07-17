# UID : 180128022
# importing packages
import os, re, random, operator, copy, sys
from random import shuffle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time

# defining the paths
path =  sys.argv[1]
cmd_path_pos = path+'/txt_sentoken/pos'
cmd_path_neg = path+'/txt_sentoken/neg'

seed_val = 0 # initialising seed value

# fixing the random seed
def set_seed(seed_value):
    seed_val = random.seed(seed_value)
    return seed_val
    

# setting the paths
pos_files = os.listdir(cmd_path_pos)
neg_files = os.listdir(cmd_path_neg)
path_pos = cmd_path_pos
path_neg= cmd_path_neg

# appending all the positive text files to the positive path
for i in range(0, len(pos_files)):
    pos_files[i] = os.path.join(path_pos, pos_files[i])
# appending all the negative text files to the negative path
for i in range(0, len(neg_files)):
    neg_files[i] = os.path.join(path_neg, neg_files[i])

# assigning training and testing set for positive and negative instances
train_pos = pos_files[0:800] # positive training set
test_pos = pos_files[800:1000] # positive testing set
train_neg = neg_files[0:800] # negative training set
test_neg = neg_files[800:1000] # negative testing set

# declaring global variables
accuracy_arr = [] # for storing accuracy  
error_arr = [] # for storing error


############ functions ############

# function for pre-processing of texts
def ngrams_generation(text, n_gram):    
    text = text.lower() # converting to lowercases
    # replacing all non-alphanumeric characters with spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = [token for token in text.split(" ") if token != ""] # tokenising
    ngrams = zip(*[tokens[i:] for i in range(n_gram)]) # generating n-grams
    return [" ".join(ngram) for ngram in ngrams] # concateting tokens


# function for opening the text files in the 'pos' folder of both the training
# and testing instances, reading them, adding bias before passing them on to 
# ngrams_generation, getting return values and returning a list of dictionaries
def ngram_traintest_pos(traintestpos, n):
    traintest_pos_list = [] # for storing positive instances
    for filename in traintestpos:
        dict_pos = {}
        with open(filename, "r") as f: # opening file
            line = f.read() # reading file
            #defining and adding bias terms for more accurate results
            bias_term = "__biasedrahsen"
            line = line + " "+ bias_term 
            dictionary = Counter(ngrams_generation(line, n)) # calling the above function
            label = 1 # setting the label to 1
            # adding dictionary as a value in dict_pos
            dict_pos["wordfreq"] = dictionary 
            dict_pos["label"] = label
            traintest_pos_list.append(dict_pos)
    return traintest_pos_list # returning the list of dictionaries

# function, similar as above, but for negative instances
def ngram_traintest_neg(traintestneg, n):
    traintest_neg_list = []
    for filename in traintestneg:
        dict_neg = {}
        with open(filename, "r") as f:
            line = f.read()
            bias_term = "__biasedrahsen"
            line = line + " "+ bias_term
            dictionary = Counter(ngrams_generation(line, n))
            label = -1
            dict_neg["wordfreq"] = dictionary
            dict_neg["label"] = label
            traintest_neg_list.append(dict_neg)
    return traintest_neg_list

# function to add the training and testing instances
def trainertester_adder(pos, neg):
    D = pos + neg
    return D

# function for creating a weight dictionary
def weight_set(traindoc):
    w = {}
    # iterating through the training instances
    for x in traindoc:
        dict_word = x["wordfreq"] # for accessing n-grams
        for y in dict_word.keys():
            if(y in w.keys()):
                continue
            # adding the words that are not present, giving them a weight of 0
            else:
                w[y] = 0 
    return w # returning updated dictionary

# function for standard perceptron
def standard_trainer(testdoc, traindoc, w):
    #iterating through all the training instances
    for x in traindoc: 
        words = x["wordfreq"]
        label = x["label"] # for accessing labels
        y_predict_sum = 0
        # iterating through the dictionary to multiply the frequency and weight
        for word,frequency in words.items():
            y_predict = frequency * w[word]
            y_predict_sum += y_predict # adding all the values
        sign_y_predict = np.sign(y_predict_sum)
        # setting the conditions
        if(sign_y_predict!= label): 
            if (label == 1):
                for word, frequency in words.items():
                    w[word] += frequency
            else:
                for word, frequency in words.items():
                    w[word] -= frequency
    accuracy = prediction(testdoc, w) # getting the accuracy
    print("The standard binary perceptron accuracy is", accuracy * 100, "%")
    return w

# function for perceptron after randomising
def shuffled_standard_trainer(testdoc, traindoc, w):
    shuffle(traindoc) # randomising the training instances
    for x in traindoc: 
        words = x["wordfreq"]
        label = x["label"]
        y_predict_sum = 0
        for word,frequency in words.items():
            y_predict = frequency * w[word]
            y_predict_sum += y_predict
        sign_y_predict = np.sign(y_predict_sum)
        if(sign_y_predict!= label):
            if (label == 1):
                for word, frequency in words.items():
                    w[word] += frequency
            else:
                for word, frequency in words.items():
                    w[word] -= frequency
    accuracy = prediction(testdoc, w)
    print("The accuracy after randomising the training data is", accuracy * 100, "%")
    return w

# function for perceptron with shuffling and multiple passes
def update_trainer(testdoc, traindocument, w_updt):
    w_ret = []
    for loop in range(15): # iterating multiple times
        shuffle(traindocument)
        for x in traindocument: 
            words = x["wordfreq"]
            label = x["label"]
            y_predict_sum = 0
            for word,frequency in words.items():
                y_predict = frequency * w_updt[word]
                y_predict_sum += y_predict
            sign_y_predict = np.sign(y_predict_sum)
            if(sign_y_predict!= label):
                if (label == 1):
                    for word, frequency in words.items():
                        w_updt[word] += frequency
                else:
                    for word, frequency in words.items():
                        w_updt[word] -= frequency               
        w_ret.append(w_updt.copy())
        accuracy_at_itr = prediction(testdoc, w_updt)
        error = 1 - accuracy_at_itr # calculating the error given the accuracy
        accuracy_arr.append(accuracy_at_itr) # appending accuracy
        error_arr.append(error) # appending error
        print("Accuracy at iteration", loop, "after multiple passes and shuffling is", accuracy_at_itr * 100, "%")
    return w_ret

# function for predicting the test data and returning the accuracy
def prediction(testdoc, w):
    correct = 0 # setting a counter
    # iterating over the testing instances
    for x in testdoc:             
        words = x["wordfreq"] # for accessing n-grams
        label = x["label"] # for accessing labels
        y_predict_sum = 0 # for getting a weighted sum
        for word,frequency in words.items():
            if word not in w:
                continue
            else:    
                y_predict = frequency * w[word]
                y_predict_sum += y_predict
        sign_y_predict = np.sign(y_predict_sum)
        # setting the signs
        if(sign_y_predict >= 0):
            sign_y_predict = 1
        else:
            sign_y_predict = -1
        # updating the counter by 1 if the sign and the label match
        if(sign_y_predict == label):
            correct += 1
    acc = correct/len(testdoc) # calculating the accuracy
    return acc

# function for plotting the error-graph
def plotter(input_n):
    # assigning the title based on the n-gram
    if(input_n == 1):
        plt.title("Unigram Error Plot")
    elif(input_n == 2):
        plt.title("Bigram Error Plot")
    else:
        plt.title("Trigram Error Plot")
    plt.plot(error_arr)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()

# function to update the dictionary and returning the average accuracy
def average_weights(weights, pos, neg):
    # setting counters
    add = Counter()
    counts = Counter()
    for item in weights:
        add.update(item)
        counts.update(item.keys())
    # calculating the average accuracy
    avg = {a: float(add[a])/counts[a] for a in add.keys()}
    print("The average accuracy is ", prediction(trainertester_adder(pos, neg), avg) * 100, "%")
    return avg

# function to print the top 10 positive and negative
# n-grams with their respective weights
def top_ten_values(x):
    a1_sorted_keys = sorted(x.items(), key=operator.itemgetter(1))
    a2_sorted_keys = a1_sorted_keys[::-1]
    print("Top ten most positively-weighted n-grams are: ", a2_sorted_keys[:10])
    print("Top ten most negatively-weighted n-grams are: ", a1_sorted_keys[:10])


############ function calls ############

# taking user input
input_n = input("Enter 1 for Unigram, 2 for Bigram and 3 for Trigram : ")
seed_val = input("Enter random seed value : ")
seed_val = int(seed_val)
set_seed(seed_val)
input_n = int(input_n)
# printing according to the input provided
if(input_n == 1):
    print("Unigram Generation with seed value", seed_val, "...")
elif(input_n == 2):
    print("Bigram Generation with seed value", seed_val, "...")
else:
    print("Trigram Generation with seed value", seed_val, "...")
    
# creating objects and passing the appropriate arguments to the above functions
# using deepcopy to address the issue of referencing in Python, but the 
# trade-off is the program becomes a little slow while running
start = time.time()
ngram_pos = ngram_traintest_pos(copy.deepcopy(train_pos), input_n)
ngram_neg = ngram_traintest_neg(copy.deepcopy(train_neg), input_n)
D_train = trainertester_adder(copy.deepcopy(ngram_pos), copy.deepcopy(ngram_neg))
ngram_pos = ngram_traintest_pos(copy.deepcopy(test_pos), input_n)
ngram_neg = ngram_traintest_neg(copy.deepcopy(test_neg), input_n)
D_test = trainertester_adder(copy.deepcopy(ngram_pos), copy.deepcopy(ngram_neg))
weight = weight_set(copy.deepcopy(D_train))
zero_train = standard_trainer(copy.deepcopy(D_test), copy.deepcopy(D_train), copy.deepcopy(weight))
predict = prediction(copy.deepcopy(D_test), copy.deepcopy(zero_train))
train = shuffled_standard_trainer(copy.deepcopy(D_test), copy.deepcopy(D_train), copy.deepcopy(weight))
predict = prediction(copy.deepcopy(D_test), copy.deepcopy(train))
update_train = update_trainer(copy.deepcopy(D_test), copy.deepcopy(D_train), copy.deepcopy(weight))
updatedpredict = prediction(D_test, update_train[-1])
avg_weight = average_weights(update_train, ngram_pos, ngram_neg)
topten = top_ten_values(avg_weight)
end = time.time() - start
print("The time taken is", round(end, 2), "seconds")
plot = plotter(input_n)




