
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
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "") # taking last value and ignore ['A',' B'] => 'A',' B' => A, B => A,B => str(A,B)                                            
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
for question in cleanQuestions:  # Taking the first list list[0] = ['how are you']
    for word in question.split(): # Then split() by space , ['how are you'] => str(how, are, you)
        if word not in word2Count: # Initially given word is not in the dictionary, word2Count['how'] = 1, word2Count['are'] = 1, word2Count['you'] = 1 
            word2Count[word] = 1
        else:
            word2Count[word] += 1  # Second time if word appears then it will incremented to 1, str(how, do, know) => word2Count['how'] = word2Count['how'] + 1
for answer in cleanAnswers:
    for word in answer.split(): # ['That is why i am using split function buz It's not a string to split by comma. it is a Sentence to split by single whitspace']
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

# How can sorting by length of the question help? It’s attributed to reduce loss through reducing the use of padding to help speed up training
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
    # tf.int32 - type bcuz we did data preprocessing steps Everything is converted into Integers
    # shape(None, None) - sorted_clean_questions consists of list of Integers with padding will get 2D-Matrix
    # name = 'Input' - Node or Vertex name is 'Input' for Tensorboard Visualizations
    targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Target')
    # tf.int32 - type bcuz we did data preprocessing steps Everything is converted into Integers
    # shape(None, None) - sorted_clean_answers consists of list of Integers with padding will get 2D-Matrix and we will compare target with sorted_clean_answers
    # name = 'Input' - Node or Vertex name is 'Input' for Tensorboard Visualizations
    lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
    # tf.float32 - shouldn't be Integer
    keep_prob = tf.placeholder(dtype=tf.float32, name='dropout')
    # keep_prob - is the parameter that controls dropout rate
    return inputs, targets, lr, keep_prob 





