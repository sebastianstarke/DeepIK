DeepIK
======================================================

Description
------------
This project uses neural networks (MLP) for learning inverse kinematics functions in Theano / Keras. It is a basic implementation with several hidden layers and additional drop-out to avoid overfitting.

As experimental platform, the kinematic geometry of the PA10 6-DOF robot is used with randomly sampled reachable joint configurations for the training and test data. The input is given as a 7-dimensional vector of XYZ-position and XYZW-rotation (quaternion representation), and the output is the 6-dimensional joint variable vector. All input and output data is normalized.

Having set up your Theano installation, you can test the code via calling 'python deepik.py', and can observe the training and test errors at each epoch. It should be noted that learning IK via neural networks can be challenging due to rotation representations, which maylead to ambiguous prediction samples.
