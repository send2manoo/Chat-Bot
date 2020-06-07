#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:45:21 2020

@author: root
"""

# pip install tensorboard

# Importing tensorflow library
import tensorflow as tf
# Creating the Vertices in a Graph
a = tf.constant(5)
b = tf.constant(7)
c = tf.add(a, b)
# Launch a Graph in a session
with tf.Session() as sess:
    # Creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))
    

import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

# create graph  
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = tf.add(a, b, name="addition")

# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))      