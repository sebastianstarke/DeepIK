DeepIK
======================================================

Description
------------
This project uses neural networks (MLP) for learning inverse kinematics functions in Theano / Keras (Python). It is a basic implementation with several hidden layers with additional drop-out to avoid overfitting by introducing sparsity.

As experimental platform, the kinematic geometry of the PA10 6-DOF robot is used with randomly sampled reachable joint configurations for the training and test data. The input is given as a 7-dimensional vector of XYZ-position and XYZW-rotation (quaternion representation), and the output is the 6-dimensional joint variable vector. All data is normalised with respect to the input and output data.

Having set up your Theano installation, you can test the code simply via calling 'python deepik.py'. You can then observe the training and testing errors at each epoch.
