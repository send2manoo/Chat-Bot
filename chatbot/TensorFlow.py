#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:39:00 2020

@author: root
######################################### IMPORTANT #############################################
1) https://www.easy-tensorflow.com/tf-tutorials/basics/graph-and-session
2) use snippets addon for firefox to get your data
"""

'''
############################################# GRAPH ##################################################
A computational graph (or graph in short) is a series of TensorFlow operations(tf.add(), tf.multiply()) arranged into a graph of nodes". Basically, 
it means a graph is just an arrangement of nodes that represent the operations in your model

################################### SESSION ###################################

To compute anything, a graph must be launched in a session. 
Technically, session places the graph ops on hardware such as CPUs or GPUs and provides methods to execute them.

'''

import tensorflow as tf
# Create a Node in a Graph
a = 2
b = 3
c = tf.add(a, b, name ='Add')
print(c)
#Launch the Graph in a Session
sess = tf.Session()
print(sess.run(c))
sess.close()


# it close automatically without sess.close() method
with tf.Session() as sess:
    print(sess.run(c))
    

'''
This is one of the advantages of defining a graph and running a session on it!
It helps running only the required operations of the graph and skip the rest (remember flexibility). 
This specially saves a significant amount of time for us 
when dealing with huge networks with hundreds and thousands of operations.
'''
# Create a Node in a Graph
add_op = tf.add(a, b, name = 'Add')
mul_op = tf.multiply(a, b, name = 'Multiply')
power_op = tf.pow(add_op, mul_op, name = 'Power')
useless_op = tf.multiply(a, add_op, name = 'Useless')
#Launch the Graph in a Session
with tf.Session() as sess:
    print(sess.run([power_op, useless_op]))



# Create a Node in a Graph
a = tf.constant(2, name = 'a')
b = tf.constant(3, name = 'b')
c = tf.add(a, b, name= 'Addition')
#Launch the Graph in a Session
with tf.Session() as sess:
    print(sess.run([a, b, c]))


# Create a Graph
s = tf.constant(2.57, name = 'scalar', dtype=tf.float32)
m = tf.constant([[1, 2],[3, 4]], name = 'matrix', dtype=tf.int32)
#Launch the Graph in a Session
with tf.Session() as sess:
    print(sess.run([s, m]))


'''
FailedPreconditionError: Attempting to use uninitialized value

 
Upon executing the program, we run into FailedPreconditionError: Attempting to use uninitialized value.
This is because we tried to evaluate the variables before initializing them. 
Let's correct the code by first initializing all the variables and then proceed to evaluate them
'''
# This wont work bcuz tf.Variable needs Initializer rather than tf.constant (Variable is a class, constant is an operation)
# Create a Node in a Graph
s = tf.Variable(5, name = 'scalar')
m = tf.Variable([[1, 2], [3, 4]], name = 'matrix')
w = tf.Variable(tf.zeros([784, 10], name = 'zeros'))

# add an Op to initialize global variables
init_op = tf.global_variables_initializer()
#Launch the Graph in a Session
with tf.Session() as sess:
     # run the variable initializer operation
    sess.run(init_op)
    print(sess.run([s, m, w]))
    

'''
__* IMPORTANT Note:__ Calling tf.Variable to create a variable is the old way of creating a variable.
TensorFlow recommends to use the wraper __tf.get_variable__ instead as it accepts the name, shape, 
etc as its arguments with many more as follow
'''
# Every time you have to restart the Kernal because you do not override s varible ie. constant unless you are not execte the above portions 
# Otherwise you will get ValueError: Variable scalar already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
# Create a Node in a Graph
s = tf.get_variable(name='scalar', initializer=tf.constant(5))
m = tf.get_variable(name='matrix', initializer=tf.constant([[1, 2], [3, 4]], dtype=tf.float32))
W = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.zeros_initializer())
# Initialize All Variable before try to Evaluate Them
init_op = tf.global_variables_initializer()
#Launch the Graph in a Session
with tf.Session() as sess:
    # run the variable initializer operation
    sess.run(init_op)
    print(sess.run([s, m, W]))


# Create the weight and bias matrices for a fully-connected layer with 2 neuron to another layer with 3 neuron.
# In this scenario, the weight and bias variables must be of size [2, 3] and 3 respectively
# Create a Node in a Graph
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
Weights = tf.get_variable(name='weights', shape=(2, 3), initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
biases = tf.get_variable(name='biases', shape=(3), initializer=tf.zeros_initializer())
# Initialize above Variable
init_op = tf.global_variables_initializer()
#Launch the Graph in a Session
with tf.Session() as sess:
     # run the variable initializer operation
    sess.run(init_op)
    # Evaluate the Operation in a Node
    W, b = sess.run([Weights, biases])
    print('weights = {}'.format(W))
    print('biases = {}'.format(b))
    

# Create a constant vector and a placeholder and add them together
# Create a Node in a Graph
a = tf.constant(5, dtype=tf.float32, name='a')
b = tf.placeholder(dtype=tf.float32, shape=(3), name='b')
add = tf.add(a, b, name='add')
# #Launch the Graph in a Session
with tf.Session() as sess:
    # Evaluate the Operation in a Node
    print(sess.run(add))
# You will get an Error, InvalidArgumentError: You must feed a value for placeholder tensor 'b' with dtype float and shape [3]


# Create a Node in a Graph
a = tf.constant(5, dtype=tf.float32, name='a')
b = tf.placeholder(dtype=tf.float32, shape=(3), name='b')
add = tf.add(a, b, name='add')
# #Launch the Graph in a Session
with tf.Session() as sess:
    # Create a Dictionary for Placeholder of shape(3)
    d = {b:[1, 2, 3]} # fed list of values to 'b' Placeholder Tensors, Here b is a Key and also Tensor of shape(3) we mentioned above.
    # Evaluate the Operation in a Node
    print(sess.run(add, feed_dict=d))


# import the tensorflow library
import tensorflow as tf
import numpy as np

tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

# create the input placeholder
X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

# create network parameters
W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[784, 200], initializer=weight_initer)
bias_initer =tf.constant(0., shape=[200], dtype=tf.float32)
b = tf.get_variable(name="Bias", dtype=tf.float32, initializer=bias_initer)

# create MatMul node
x_w = tf.matmul(X, W, name="MatMul")
# create Add node
x_w_b = tf.add(x_w, b, name="Add")
# create ReLU node
h = tf.nn.relu(x_w_b, name="ReLU") 

# Add an Op to initialize variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # initialize variables
    sess.run(init_op)
    # create the dictionary:
    d = {X: np.random.rand(100, 784)}
    # feed it to placeholder a via the dict 
    print(sess.run(h, feed_dict=d))
 