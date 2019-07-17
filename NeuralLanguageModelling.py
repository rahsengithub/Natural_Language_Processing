# UID : 180128022

# imports
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import operator

torch.manual_seed(2)

######################################################################


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# test data
test_sentence = "The mathematician ran . , " \
                "The mathematician ran to the store . , " \
                "The physicist ran to the store . , " \
                "The philosopher thought about it . , " \
                "The mathematician solved the open problem .".split(
    " , ")  # splitting them based on the 'comma' delimitor

print("\nThe test sentences are: \n", test_sentence)

processed_sentence = []  # declaring an empty array to store the processed sentences
string_processed_sentence = ""  # declaration of an empty string
for markers in test_sentence:  # ietrating through the list of the sentences
    sentence = "START" + " " + markers + " " + "STOP" + " "  # adding the START/STOP tags to each sentence
    sentences = sentence.split()  # splitting the sentences into tokens
    processed_sentence.append(sentences)  # appending the tokens to form list of tokens
    string_processed_sentence = string_processed_sentence + sentence  # changing them in to string

print("\nThe list of sentences where each sentence is represented "
      "as a list of tokens: \n", processed_sentence)

print("\nComplete sentences are: \n ", string_processed_sentence)



vocab = sorted(set(string_processed_sentence.split()))  # finding out the unique tokens
print("\nVocab is", vocab, "\n")
word_to_ix = {word: i for i, word in enumerate(vocab)}  # adding a counter to the iterable and forming a dictionary


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size,
                                       embedding_dim)  # storing word embeddings and retrieving them using indices
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)  # applying a linear transformation to the data
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        # print("The input-layer dimension: ", embeds.shape)
        out = F.relu(self.linear1(embeds))  # using the Relu Activation function
        # print("The hidden-layer dimension: ", out.shape)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)  # applying logarithm after softmax
        # print("The output-layer dimension: ", log_probs.shape)
        return log_probs, self.embeddings  # returning the embeddings along with log_probs


losses = []  # declaring an empty list to store the losses
loss_function = nn.NLLLoss()  # function to calculate the loss
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM,
                             CONTEXT_SIZE)  # calling the function by passing the required arguments
# implementing Stochastic Gradient Descent to optimise the model.parameters with the provided learning rate

optimizer = optim.SGD(model.parameters(), lr=0.01)  # learning rate (lr) is a hyper-parameter

for epoch in range(20):  # epoch is 20, which is again a hyper-parameter
    total_loss = torch.Tensor([0])  # constructing a multi-dimensional matrix
    for sent in processed_sentence:
        # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
        trigrams = [([sent[i], sent[i + 1]], sent[i + 2])
                    for i in range(len(sent) - 2)]
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            model.zero_grad()
            log_probs, embeddings = model(context_var)

            # Step 4. Compute your loss function. (Again, Torch wants the target word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!

############### Run a Sanity check ###############

sanity_sent = processed_sentence[1]  # "The mathematician ran to the store . "

sanity_trigrams = [([sanity_sent[i], sanity_sent[i + 1]], sanity_sent[i + 2])
                   for i in range(len(sanity_sent) - 2)]

print(sanity_trigrams)

# swapping the keys and values of word_to_ix dictionary and storing them to a new dictionary
inv_word_to_ix = {value: key for key, value in word_to_ix.items()}
print("\n")

updater = 0
for i in range(5):  # taking range as 5 for 5 consecutive runs
    print("Run", i + 1, "of 5 consecutive runs")  # printing the run values
    updater = 0  # flushing the value of the counter
    for context, target in sanity_trigrams:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        model.zero_grad()
        log_probs, embeddings = model(context_var)
        val_key = list(log_probs[0]).index(max(log_probs[0]))  # finding the index of maximum log_probs
        word_val = inv_word_to_ix[val_key]  # finding the token at the value of the key for the maximum log_probs index
        if target == word_val:  # checking if the target is equal to the word_val
            updater = updater + 1  # updating the counter by 1 upon successful match
    num_trigrams = len(sanity_trigrams)  # calculating the number of trigrams
    percentage = round((updater / num_trigrams), 2)
    print("Percentage of correct predictions is given as", (percentage * 100),
          "%")  # rounding it to two decimal places
############### Test ###############

predict_sent = " The ______ solved the open problem. "  # the sentence to be considered
possible_words = ["physicist", "philosopher", "mathematician"]  # the target words
result_dict = {}  # declaring a dictionary to store the results
emb_dict = {}  # declaring a dictionary to store the embedding values
for word in possible_words:
    possible_sentence = predict_sent.replace("______",
                                             word).split()  # replacing the underline with each of the target words

    test_trigrams = [([possible_sentence[i], possible_sentence[i + 1]], possible_sentence[i + 2])
                     for i in range(len(possible_sentence) - 2)]

    checker = 0  # initialising a counter to 0
    for context, target in test_trigrams:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        model.zero_grad()
        log_probs, embeddings = model(context_var)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[word]])))

        total_loss += loss.data  # updating the total_loss value by the previous loss value
    result_dict[word] = total_loss
    finder = torch.LongTensor([word_to_ix[word]])  # finding out the values in the dictionary for the target words
    emb = embeddings(autograd.Variable(finder))  # wrapping the tensor and recording the operations applied to it
    emb_dict[word] = emb  # updating the dictionary

correct_word = min(result_dict, key=result_dict.get)  # retrieving the key with the maximum value
print("\nThe predicted sentence is: The", correct_word, "solved the open problem. \n")

############### Cosine Similarity ###############

# Returning cosine similarity between two target words computed along dim.
cos = nn.CosineSimilarity(dim=1, eps=1e-8)
out_phy_mat = cos(emb_dict["physicist"], emb_dict["mathematician"])
out_phi_mat = cos(emb_dict["philosopher"], emb_dict["mathematician"])
print("The cosine similarity between physicist and mathematician is", round(float(out_phy_mat), 2))
print("The cosine similarity between philosopher and mathematician is", round(float(out_phi_mat), 2))
