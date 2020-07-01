#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:00:00 2020

@author: root
"""

#################################### CLEAR #########################################
# Execute inner loop first then add up Outer loop in Spyder IPython Console


test_sorted_clean_questions = []
test_sorted_clean_answers = []


test = [[4,23,55], [27], [323,12,43], [545,43], [555,777,3334,1233,3434]]
test1 = [[23,55], [25523], [33], [5,343], [6323,545,77,342,753], [21, 65, 23]]

for length in range(1, 5+1):
    for i in enumerate(test):   # i usually as Tuples (0, [....]) 
        if len(i[1]) == length:      # i[1] is a tuple is contains list questionsIntoInt
            test_sorted_clean_questions.append(test[i[0]])
            test_sorted_clean_answers.append(test1[i[0]])


import tensorflow as tf
_ = tf.zeros(shape=(10, 3,3), dtype=tf.float32)

sess = tf.Session()
print(sess.run(_))
sess.close()

# https://www.digitalocean.com/community/tutorials/how-to-index-and-slice-strings-in-python-3
name = "manohar focusing"
# Slicing
print(name)
print(name[-1])
print(name[8:len(name)]) # focusing
print(name[:7])  # manohar
print(name[7:])  #  focusing
print(name[-15:-7]) # anohar f
# Striding
print(name[8:len(name):1]) # focusing
print(name[0:len(name):2]) # mnhrfcsn
print(name[::2]) # mnhrfcsn
'''
The two colons without specified parameter will include all the characters from the original string, 
a stride of 1 will include every character without skipping, 
and negating that stride will reverse the order of the characters.
'''
print(name[::-1]) # gnisucof rahonam 
print(name[::-2]) # giuo aoa



