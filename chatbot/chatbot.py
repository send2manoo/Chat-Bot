
################# IMPORTANT ##############
# conda create -n chatbot python=3.5
# pip install tensorflow==1.0.0
# conda install tensorboard


# Building a ChatBot with Deep NLP
# dataset - cornell movie dialog corpus    

# Importing the Libraries
import numpy as np
import tensorflow as tf
import re
import time


######## DATA PREPROCESSING #########

'''
 step 1) Importing the Dataset using right encoding format, ignore errors and split by newline
         1)  lines = [[....]]
         2) conversations = [[........]]
 
 step 2) Creating a Dictionary from step-1 1) lines List to map each line ID's with Line Sentences
         1) id2Line = { 'L154' : 'How Are You?.', ...}
         
step 3)  Creating a Converstations ID's List from step-1 2) conversations list  and remove empty spaces.
         1) conversationsIds = [['L123', 'L478'], [.....]]

step 4)  Creating two lists for Separately Questions And Answers
         1) questions = []
         2) answers = []

step 5)  Define a Function for cleaning of Texts

step 6)  Make a Separate Two list for Cleaned Questions and Answers

step 7)  Now Creating the Second Dictionary that maps word to its number of Occurances of each Cleaned Questions
         Answers.
         1) word2Count = {'how' : 25, ...}

step 8)  Creating 3rd & 4th Dictionaries that maps the Question words and Answers words to a Unique Integer 
         
         questionswords2Int = {}
         answerswords2Int = {}
         
         So that i'm getting Unique Integers for all the words in word2Count Dict and also eliminate least Significant word Which is lesser than 20          


step 9)  Adding the last Tokens to these two Dictionaries questionswords2Int = {}, answerswords2Int = {}
         1) tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']  

step 10) Creating the Inverse dictionary of the answerswords2Int dictionary

step 11) # Adding the End Of String Token to the end of Cleaned Answer (Which is needed for Decoder part for SEQ2SEQ)

step 12) Translating all the Questions and Answers into Integers
         And Replacing all the Words that were not Found in cleanQuestions and cleanAnswers Dictionary to Filtered Out by <OUT> Token
         Tutorial 14.\ ChatBot\ -\ Step\ 16.mp4

         1) cleanQuestions = [['how are you'], [..]] => questionsIntoInt = [[451, 5788], [..]]

step 13)  ############## IMPORTANT ############### CLEAR
          Sorting Questions and Answers by the length of Questions by 25 maximum words
          Adding test.py file for References
'''



# Importing the Dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n');
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n');


# Creating a Dictionary that maps each lineIds with its Line Sentences
id2Line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
#    print(_line)
    if len(_line) == 5:
        id2Line[_line[0]] = _line[4]


# Creating a List of conversations LineIds
conversationsIds = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "") # taking last value and ignore ['A', 'B'] => 'A', 'B' => A, B => A,B => str(A,B)                                            
#    print(_conversation) # type(_conversation) => str
    conversationsIds.append(_conversation.split(",")) # A,B => ['A', 'B']
    

# Getting Separately Questions And Answers
questions = []
answers = []
for conversation in conversationsIds:
     for index in range(len(conversation) - 1):
         questions.append(id2Line[conversation[index]]) # Questions
         answers.append(id2Line[conversation[index + 1]]) # Answers
         
# Doing first cleaning of Texts
def cleanText(text):
    text = text.lower();
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
#    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)        
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't'", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text
    
# Cleaning the Questions
cleanQuestions = []
for question in questions:
    cleanQuestions.append(cleanText(question))
# Cleaning the Answers
cleanAnswers = []
for answer in answers:
    cleanAnswers.append(cleanText(answer))

    
# Creating the Dictionary that maps word to its number of Occurances
word2Count = {}
for question in cleanQuestions:  # Taking the first list list[0] = 'how are you'
    for word in question.split(): # Then split() by space , 'how are you' => str(how, are, you)
        if word not in word2Count: # Initially given word is not in the dictionary, word2Count['how'] = 1, word2Count['are'] = 1, word2Count['you'] = 1 
            word2Count[word] = 1
        else:
            word2Count[word] += 1  # Second time if word appears then it will incremented to 1, str(how, do, know) => word2Count['how'] = word2Count['how'] + 1
for answer in cleanAnswers:
    for word in answer.split(): # it is a Sentence to split by single whitspace
        if word not in word2Count:
            word2Count[word] = 1
        else:
            word2Count[word] += 1

# Creating two Dictionaries that maps the Question words and Answers words to a Unique Integer 
threshold = 20 # Remove 5 percent of least significant words
questionswords2Int = {}
wordNumber = 0
for word, count in word2Count.items():
    if count >= threshold:
        questionswords2Int[word] = wordNumber # So that i'm getting Unique Integers for all the words in word2Count Dict and also eliminate least Significant word Which is lesser than 20 
        wordNumber = wordNumber + 1
answerswords2Int = {}
wordNumber = 0
for word, count in word2Count.items():
    if count >= threshold:
        answerswords2Int[word] = wordNumber
        wordNumber = wordNumber + 1

# Adding the last Tokens to these two Dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']  
#'<PAD>' => Giving same input size (input sentence might vary we should <PAD> tokens to same size),
#           What does the <PAD>token do? PAD is the padding that replaces the empty cells by using the token <PAD>
#'<EOS>' => End of String Should given to the start of Decoder 
#'<OUT>' => Filtered out Dictionaries(questionswords2Int, answerswords2Int)
#'<SOS>' => Start of String
for token in tokens:
    questionswords2Int[token] = len(questionswords2Int) + 1
for token in tokens:
    answerswords2Int[token] = len(answerswords2Int) + 1
    
# Creating the Inverse dictionary of the answerswords2Int dictionary
answersInts2Words = {w_i:w for w, w_i in answerswords2Int.items()}

# Why don’t we attach <EOS>and <SOS>tokens for the questions? 
# We don’t attach them because we only need them for the answers 
# since the first element required by the decoder is the <SOS>token and the last one the <EOS>token
# Adding the End Of String Token to the end of Cleaned Answer (Which is needed for Decoder part for SEQ2SEQ)
for index in range(len(cleanAnswers)):
    cleanAnswers[index] = cleanAnswers[index] + ' <EOS>'
    
# Translating all the Questions and Answers into Integers
# And Replacing all the Words that were not Found in cleanQuestions and cleanAnswers Dictionary to Filtered Out by <OUT> Token
# Tutorial 14.\ ChatBot\ -\ Step\ 16.mp4
questionsIntoInt = []
for question in cleanQuestions:
    listOfIntsForEachQuestion = [] # Create a list for each question which contains Integers
    for word in question.split(): # ['That is why i am using split function buz It's not a string to split by comma. it is a Sentence to split by single whitspace']
        if word not in questionswords2Int:
            listOfIntsForEachQuestion.append(questionswords2Int['<OUT>']) # if the word not found in the questionswords2Int dictionary then takeout the <OUT>Key value append it into listOfIntsForEachQuestion 
        else:
            listOfIntsForEachQuestion.append(questionswords2Int[word]) # else take out the value for given words to append listOfIntsForEachQuestion
    questionsIntoInt.append(listOfIntsForEachQuestion)

answersIntoInt = []
for answer in cleanAnswers:
    listOfIntsForEachAnswer = [] # Create a list for each answer which contains Integers
    for word in answer.split(): # ['That is why i am using split function buz It's not a string to split by comma. it is a Sentence to split by single whitspace']
        if word not in answerswords2Int:
            listOfIntsForEachAnswer.append(answerswords2Int['<OUT>']) # if the word not found in the answerswords2Int dictionary then takeout the <OUT>Key value append it into listOfIntsForEachAnswer
        else:
            listOfIntsForEachAnswer.append(answerswords2Int[word]) # else take out the value for given words to append listOfIntsForEachAnswer
    answersIntoInt.append(listOfIntsForEachAnswer)

'''How can sorting by length of the question help? It’s attributed to reduce loss through reducing the use of padding to help speed up training'''
# Sorting Questions and Answers by the length of Questions by 25 maximum words why bcuz speedup the training help to reduce loss
# It will reduce the amount of padding during training
# Adding test.py file for References
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25+1):  # miminum range of values should be 1 bcuz (Y or Yes , N or No) Where Y usually 1 Characters, So Minimum value of Sorted List is One(1). and Maximum value is 25.
    # reduced to below 25 So that you acheive faster Training Time. Why added (25+1) bcuz index 25 is excluded in Python index
    for i in enumerate(questionsIntoInt):   # i usually as Tuples (0, [....]) 
        if len(i[1]) == length:      # i[1] is a tuple is contains list questionsIntoInt
            sorted_clean_questions.append(questionsIntoInt[i[0]])  ############ IMPORTANT ########### OPEN questionsIntoInt, answersIntoInt list map with enumerated index value
            sorted_clean_answers.append(answersIntoInt[i[0]])  # answersIntoInt[i[0]] Contains exactly the Answer of i[0] WHICH IS THE ANSWER TO THE QUESTION OF INDEX I[0] IN THE QUESTIONSINTOINT[I[0]] LIST
            # if you sort the Questions based on the tuple(i[0], i[1]) 
            # for eg: i = (0, [75, 5545, 78784])(1, [45, 87]),.. , (81, [752]), length = 1
            # i[1] == length, when i[0] = 81. This index value to be placed in Sorted list.
            # using i[0] = 81 get the list value, questionsIntoInt[i[0]] => questionsIntoInt[81] = [752] => sorted_clean_questions.append([752]) => [[752], [4], [5887], [7, 45], [8555, 998], ..]
          
            # Using 81 to get the Answers of questionsIntoInt[81] , How? 
            # Here to open questionsIntoInt, answersIntoInt list check 81st index , we have list of Integers corresponing to the questionsIntoInt and answersIntoInt List
            # questionsIntoInt[81] = 5475 based on length = 1, we have to appended in sorted_clean_questions. 
            # but the corresponding answersIntoInt[81] contains list of Integer values based on questionsIntoInt[81] , so we have to appended in sorted_clean_answers list in this case Answers is not Sorted because of Variable list of Integers.
            # We are going to only Sort questionsIntoInt
            # CLEAR
            
            
########### PART 2 - BUILDING THE SEQ2SEQ MODEL #######################
            
# creating the Placeholder for the Inputs and the Targets
def modelInputs():
    # Create a Node in a Graph
    inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Input')
    '''Why is the input 2 dimensional? The input is 2 dimensional because the neural networks can only accept inputs that are in a batch, as opposed to single inputs. We must add 1 dimension corresponding to the batch.'''
    # tf.int32 - type bcuz we did data preprocessing steps Everything is converted into Integers
    # shape(None, None) - sorted_clean_questions consists of list of Integers with padding will get 2D-Matrix
    # name = 'Input' - Node or Vertex name is 'Input' for Tensorboard Visualizations
    targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Target')
    # tf.int32 - type bcuz we did data preprocessing steps Everything is converted into Integers
    # shape(None, None) - sorted_clean_answers consists of list of Integers with padding will get 2D-Matrix and we will compare target with sorted_clean_answers
    # name = 'Target' - Node or Vertex name is 'Input' for Tensorboard Visualizations
    lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
    # tf.float32 - shouldn't be Integer bcuz learning rate usually contains some decimal digits
    keep_prob = tf.placeholder(dtype=tf.float32, name='dropout')
    # keep_prob - is the parameter that controls dropout rate
    return inputs, targets, lr, keep_prob 


# Preprocessing the Targets
'''Why do we delete the last column of the answer before adding <SOS>in the
process targets function? We have to delete the last column to preserve the max sequence
length since after that we make a concatenation to add the <SOS>token at the beginning of
the sequence, we must remove the last token before so that the sequence length doesn’t go
over the max sequence length'''
# Encoder => Inputs => Questions
# Decoder => Outputs or (Targets) => Answers 
# 1) RNN or LSTM will not allow singel Target that is SINGLE ANSWER. Creating the batches for sorted_clean_answers. why sorted_clean_answers bcuz we need it for decoder. decoder accept the sorted_clean_answers
# 2) Append <SOS> token to each of sorted_clean_answers beginning. 
def preprocessTargets(targets, word2Int, batchSize):
    leftSide = tf.fill(dims = [batchSize, 1], value = word2Int['<SOS>'], name='fill_SOS') 
    # prettyprint # Output tensor has shape [2, 3]. fill([2, 3], 9) ==> [[9, 9, 9][9, 9, 9]]
    rightSide = tf.strided_slice(input_ = targets, begin = [0, 0], end=[batchSize, -1], strides=[1,1], name='strided_Slice')
    # https://www.digitalocean.com/community/tutorials/how-to-index-and-slice-strings-in-python-3
    # start with [0, 0] cell end with [10, -1] 10 row except last column with stride [1, 1] move One by one cell
    preprocessedTargets = tf.concat([leftSide, rightSide], axis = 1, name='concat') # axis = 1 => means horizontal concat
    return preprocessedTargets


# Creating the Encoder RNN layer
'''LSTM stands for Long Short Term Memory and it’s a very popular type of RNN that is prominent for many AI implementations due to the architecture and benefits'''
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    # rnn_inputs = modelInputs() functions
    # rnn_size = no of input Tensors
    # num_layers = no of layers
    # keep_prob = dropout rate
    # sequence_length = length of the Questions is 25 from the sorted_clean_questions in each Batch
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    '''Dropout is a technique used for regularization in neural nets and it can help in learning and preventing overfitting with the data.'''
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob) # it wraps lstm Object that we created lstm above
    # dropout rate = 20% of neurons weights is not Updated to avoid Overfitting
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) # Usually it is a Stacked lstm_dropout that we created above
    # eg. num_layers = 10 => lstm_dropout * 10 => we have created 10 lstm_dropout layer and stacked Implicitly
    '''What is the difference between an encoder cell and encoder state? An encoder cell
    is the cell inside the encoder RNN that contains the stacked LSTM layers. An encoder state
    is the output returned by the encoder RNN, right after the last fully connected layer or simply Fully Connected Layer.'''
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell,
                                                       inputs = rnn_inputs,
                                                       sequence_length = sequence_length,
                                                       dtype = tf.float32)
    '''
    Takes input and builds independent forward and backward RNNs. 
    The input_size of forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not
    given
    '''
    # A tuple (_, encoder_state) 
    # Not Required for This Module: where: _: A tuple (output_fw, output_bw)
    # we are going to use encoder_state: A tuple (output_state_fw, output_state_bw) containing the forward and the backward final states of bidirectional rnn
    # The input_size of forward and backward cell must match
    return encoder_state


# Decoding the Training set
# 1) https://blog.floydhub.com/attention-mechanism/
# 2) https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05
def decodingTrainingSet(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size) :
# encoder_state - we getting from encoder_rnn_layer
# decoder_cell - The cell in the RNN of the Decoder
# decoder_embedded_input - An embedding is a mapping of discrete Objects. Such as WORDS to Vector of REAL numbers.
# sequence_length = length of the Answers is 25 from the sorted_clean_answers
# decoding_scope = tf.variable_scope => An Advanced Data Structure that will WRAP your TENSORFLOW Variables.
# output_function - Is the Function to return Decoder outputs
# keep_prob - dropout rate
# batch_size - (10, 1) 
    attention_states = tf.zeros(shape=(batch_size, 1, decoder_cell.output_size), dtype=tf.float32, name='attn_states_zeros')
    #  _ = tf.zeros(shape=(10, 3,3), dtype=tf.float32)
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states=attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    #    attention_keys: to be compared with target states.
    #    attention_values: to be used to construct context vectors. Context(Returned by the Encoder) Vectors used by the Decoder as the first element of the decoding.
    #    attention_score_fn: to compute similarity between key and target states.
    #    attention_construct_fn: to build attention states
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state = encoder_state[0],
                                                                              attention_keys = attention_keys,
                                                                              attention_values = attention_values,
                                                                              attention_score_fn = attention_score_function,
                                                                              attention_construct_fn = attention_construct_function,
                                                                              name = 'attn_dec_train')
    # Attentional decoder function for dynamic_rnn_decoder during training
    decoder_output, decoder_final_state, decoder_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(cell = decoder_cell,
                                                                                                        decoder_fn = training_decoder_function,
                                                                                                        inputs = decoder_embedded_input,
                                                                                                        sequence_length = sequence_length,
                                                                                                        scope = decoding_scope)
    # Dynamic RNN decoder for a sequence-to-sequence model specified by RNNCell and decoder function
    decoder_output_dropout = tf.nn.dropout(x = decoder_output, keep_prob = keep_prob) # Computes dropout
    return output_function(decoder_output_dropout)
    
    
# Decoding the Test/Validation set
def decodingTestValidationSet(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size) :

    ''' What is the difference between the Decoder embeddings matrix and Decoder embeddings input?
    Also what is the purpose of using multiple columns in the decoder embeddings matrix?
    The purpose of the embeddings matrix is to compute more efficiently the embedding input.
    Basically, you multiply your vector of inputs by the embeddings matrix to get your embed-
    ded inputs. For some further information please take a look at the following resources:
    https://www.tensorflow.org/versions/master/programmers_guide/embeddingandhttp:
    //web.stanford.edu/class/cs20si/lectures/notes_04.pdf '''
# sos_id - start of string, eos_id - end of string, maximum_length - the length of the longest entry you can find in the batch, num_words - total no of words in all the answers dict. e.g answerswords2Int
# sos_id, eos_id, maximum_length, num_words needs for function tf.contrib.seq2seq.attention_decoder_fn_inference() [for Testing, Validation]
# encoder_state - we getting from encoder_rnn_layer
# decoder_cell - The cell in the RNN of the Decoder

# maximum_length = length of the Answers is 25 from the sorted_clean_answers
# decoding_scope = tf.variable_scope => An Advanced Data Structure that will WRAP your TENSORFLOW Variables.
# output_function - Is the Function to return Decoder outputs
# keep_prob - dropout rate
# batch_size - (10, 1) 
    attention_states = tf.zeros(shape=(batch_size, 1, decoder_cell.output_size), dtype=tf.float32, name='attn_states_zeros')
    #  _ = tf.zeros(shape=(10, 3,3), dtype=tf.float32)
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states=attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    #    attention_keys: to be compared with target states.
    #    attention_values: to be used to construct context vectors. Context(Returned by the Encoder) Vectors used by the Decoder as the first element of the decoding.
    #    attention_score_fn: to compute similarity between key and target states.
    #    attention_construct_fn: to build attention states
    # validate 10% of data for testing which is not used while Training
    test_validation_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn = output_function, 
                                                                              encoder_state = encoder_state[0],
                                                                              attention_keys = attention_keys,
                                                                              attention_values = attention_values,
                                                                              attention_score_fn = attention_score_function,
                                                                              attention_construct_fn = attention_construct_function,
                                                                              embeddings = decoder_embeddings_matrix,
                                                                              start_of_sequence_id = sos_id,
                                                                              end_of_sequence_id = eos_id,
                                                                              maximum_length = maximum_length,
                                                                              num_decoder_symbols = num_words,
                                                                              name = 'attn_dec_test_validation')
    # The attention_decoder_fn_inference is a simple inference function for a sequence-to-sequence model. It should be used when dynamic_rnn_decoder is in the inference mode
    test_predictions, decoder_final_state, decoder_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(cell = decoder_cell,
                                                                                                        decoder_fn = test_validation_decoder_function,
                                                                                                        scope = decoding_scope)
    # Dropout for only Training to improves its Performance
    return test_predictions
    

# Creating the Decoder RNN
def decoderRNN(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2Int, keep_prob, batch_size):
#    encoder_state - output of the encoder but that becomes the input to the Decoder
#    num_words - total no of words in Answer corpus
#    word2Int - answerswords2Int dict 
#    num_layers - no of RNN of Decoder.
#    keep_prob - dropout layers
#    batch_size 
    with  tf.variable_scope("decoding") as decoding_scope:
        # rnn_size = no of input Tensors
        # num_layers = no of layers
        # keep_prob = dropout rate
        # sequence_length = length of the Answers is 25 from the sorted_clean_answers
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        '''Dropout is a technique used for regularization in neural nets and it can help in learning and preventing overfitting with the data.'''
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob) # it wraps lstm Object that we created lstm above
        # dropout rate = 20% of neurons weights is not Updated to avoid Overfitting
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) # Usually it is a Stacked lstm_dropout that we created above
        # eg. num_layers = 10 => lstm_dropout * 10 => we have created 10 lstm_dropout layer and stacked Implicitly
        '''What is the decoder cell? An decoder cell
        is the cell inside the decoder RNN that contains the stacked LSTM layers.'''
        weights = tf.truncated_normal_initializer(mean = 0.0, stddev = 1.0)
        biases = tf.zeros_initializer(dtype = tf.float32)
        output_function = lambda x : tf.contrib.layers.fully_connected(x,
                                                                       num_outputs = num_words,                                                                  
                                                                       normalizer_fn = None,
                                                                       scope = decoding_scope,
                                                                       weights_initializer = weights,
                                                                       biases_initializer = biases
                                                                       )
        ''' fully_connected creates a variable called weights, representing a fully connected weight matrix, which is multiplied by the inputs to produce a Tensor of hidden units. If a normalizer_fn is provided (such as batch_norm), it is then applied. 
        Otherwise, if normalizer_fn is None and a biases_initializer is provided then a biases variable would be created and added the hidden units. 
        Finally, if activation_fn is not None, 
        it is applied to the hidden units as well.
        
        num_outputs: Integer or long, the number of output units in the layer. activation_fn: activation function, set to None to skip it and maintain a linear activation.
        '''
        training_predictions = decodingTrainingSet(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope, # with  tf.VariableScope("decoding") as decoding_scope:
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        ''' What is decoding scope.reuse variables() used for? For our approach we get our test
            predictions with cross validations (that keeps 10% of the training data), and we also use this
            to predict our answers of the chatbot after it’s training so we set our following variables and
            function with: 
        '''
        decoding_scope.reuse_variables()
        test_predictions = decodingTestValidationSet(encoder_state,
                                                     decoder_cell,
                                                     decoder_embeddings_matrix,
                                                     word2Int['<SOS>'],
                                                     word2Int['<EOS>'],
                                                     sequence_length - 1, # Not include the Last Token
                                                     num_words, 
                                                     decoding_scope,
                                                     output_function,
                                                     keep_prob,
                                                     batch_size)
    return training_predictions, test_predictions 


# Building the Seq2Seq Model
# 9.\ ChatBot\ -\ Step\ 24.mp4
def seq2seqModel(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2Int):
#    inputs - questions
#    targets - Answers  
#    answers_num_words - total no of words in answers
#    questions_num_words - total no of words in questions
#    encoder_embedding_size - no of dim of embedded Matrix for the encoder
#    decoder_embedding_size - no of dim of embedded Matrix for the decoder
#    questionswords2Int - dict to preprocess the Targets preprocessTargets(targets, word2Int, batchSize)
#    num_layers = no of layers with stacked LSTM with dropout Applied
     ''' Maps a sequence of symbols to a sequence of embeddings.
     Typical use case would be reusing embeddings between an encoder and decoder'''
     encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, 
                                                       answers_num_words + 1,
                                                       encoder_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0, 1))
     encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
     preprocessed_targets = preprocessTargets(targets, questionswords2Int, batch_size)
     decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
     decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
     ''' What is the difference between the Decoder embeddings matrix and Decoder embeddings input?
     Also what is the purpose of using multiple columns in the decoder embeddings matrix?
     The purpose of the embeddings matrix is to compute more efficiently the embedding input.
     Basically, you multiply your vector of inputs by the embeddings matrix to get your embed-ded inputs. 
     For some further information please take a look at the following resources:'''
     training_predictions, test_predictions = decoderRNN(decoder_embedded_input, 
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2Int,
                                                         keep_prob,
                                                         batch_size)
     return training_predictions, test_predictions     
     
 ########## PART 3 - TRAINING THE SEQ2SEQ MODEL ###########
 
 # Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3 # 3 layer LSTM
encoding_embedding_size = 512 # 512 column. no of column in the embedding matrix with each of the column corresponding to the embedded values
decoding_embedding_size = 512 # 512 column. no of column in the embedding matrix with each of the column corresponding to the embedded values
learning_rate = 0.01
learning_rate_decay = 0.9 # which percentage the learning rate is reduced over the iterations of training
min_learning_rate = 0.0001 # Threshold value for learning_rate_decay. Early stopping before learning_rate_decay is reduced too much learning_rate, to Avoid use minimum threshold value min_learning_rate  
keep_probability = 0.5 # 50% from jeffery hinton paper https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

# Defining a Session
tf.reset_default_graph() # remove or release previous Graph # To clear the defined variables and operations of the previous Executions
session = tf.InteractiveSession() 

# Loading the Model Inputs
inputs, targets, lr, keep_prob = modelInputs()

# Setting the Sequence Length which is to be 25
sequence_length = tf.placeholder_with_default(25, shape=None, name='sequence_length')

# Getting the Shape of the Input Tensors
input_shape = tf.shape(inputs, name = 'input_shape')

# Getting the Training and Test Predictions
# tf.reverse() changes the dim of the given Tensors like np.reshape() functions
training_predictions, test_predictions = seq2seqModel(tf.reverse(inputs, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      batch_size,
                                                      sequence_length,
                                                      len(answerswords2Int),
                                                      len(questionswords2Int),
                                                      encoding_embedding_size,
                                                      decoding_embedding_size,
                                                      rnn_size,
                                                      num_layers,
                                                      questionswords2Int)

# Setting up the Loss Error, the Optimizer and Gradient Clipping 
'''What is gradient clipping and why is it necessary?
Gradient clipping is most common in recurrent neural networks. When gradients are being propagated back in time, 
they can vanish because they they are continuously multiplied by numbers less than one. 
This is called the vanishing gradient problem. 
This is solved by LSTMs and GRUs, and if you’re using a deep feedforward network, 
this is solved by residual connections. On the other hand, you can have exploding gradients too. 
This is when they get exponentially large from being multiplied by numbers larger than 1. 
Gradient clipping will clip the gradients between two numbers to prevent them from getting too large.

Gradient clipping is one of the two ways to tackle exploding gradients. The other method is gradient scaling.
In gradient clipping, we set a threshold value and if the gradient is more than that then it is clipped. In gradient scaling, we try to scale the gradient by normalization
''' 
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, 
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length])) # Training Predictions , Targets
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None] # we have add to some safety here to make sure that tensor of the gradient is existing
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
# Padding the sequences with the <PAD> token
# --------------- before padding -----------
# Question : ['who', 'are', 'you']
# Answer   : ['<SOS>', 'I', 'am', 'a', 'bot', '.', '<EOS>']
# --------------- After padding ------------
# Question : ['who', 'are', 'you', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
# Answer   : ['<SOS>', 'I', 'am', 'a', 'bot', '.', '<EOS>']
# length of the Question Sequence must be Equal to  the length of the Answer Sequence
    
#     max_sequence_length = max([len(sequence) for sequence in batch_of_sequences]) =>
#  max([len([45, 5784, 787] )]) => max([3, 7, 35, ..]) => max_sequence_length = 35
# [[45, 5784, 787] + [8823] * (35 - 3) => [45, 5784, 787] + [8823, 8823, 8823, 8823, ...upto 33]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of Questions and Answers
# Reference test.py is added
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size): # // slash for integers and / slash for float
        start_index = batch_index * batch_size # for batch_index in range(0, 137): => 0*64 => 0 , 1*64 => 64 , 2*64 => 128
        questions_in_batch = questions[start_index : start_index + batch_size] # => (0, 64) => (64, 128) => (128, 192)
        answers_in_batch = answers[start_index : start_index + batch_size]  # # => (0, 64) => (64, 128) => (128, 192)
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2Int)) # convert list into Array 
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2Int))
        yield padded_questions_in_batch, padded_answers_in_batch       

'''
cross validation is a techniques that consists of during the training keeping 10% or 15% percent of the training data aside or 
held out which won't be used to train the neural networks that is which WON'T be basically back propagated it.
and just to keep track of the predictive power of the model on these observations are like new observations.
'''
# Splitting the Questions and Answers into Training and Validation Sets
training_validation_split = int(len(sorted_clean_questions) * 0.15) # 15% percent for validation set. 15/100 = 0.15
training_questions = sorted_clean_questions[training_validation_split:] # here we kept 85% of the data for training questions
training_answers = sorted_clean_answers[training_validation_split:] # here we kept 85% of the data for training answers
validation_questions = sorted_clean_questions[:training_validation_split] # here we kept 15% of the data for validation questions
validation_answers = sorted_clean_answers[:training_validation_split] # here we kept 15% of the data for validation answers

# Training
batch_index_check_training_loss = 100 # training loss every 100 batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1 # len(range(30597, 203984)) => 173387 // 64 => 2812 // 2 => 1406 # Training questions divide by batch size = total no of batches, we have to get half of the batches so divide by 2






