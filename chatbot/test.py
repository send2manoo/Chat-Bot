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


