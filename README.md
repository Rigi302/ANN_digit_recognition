# FNN_digit_recognition
Achieve a basic Artificial Neural Network(FNN) without depending on pytorch which is able to perform real time basic-digits recognition. The project is helpful to the layman to understand the basic fundamental logic and running mode of FNN.

The network is trained by famous mnist-dataset and author's handwritten digits.

It contains three major part in python code: neuralNetwork, trainnANNandSave and capture. neuralNetwork is used to build a basic ANN class, including different types of method(e.g. train,query). 

The second part allow users to build their own networks and train them with various dataset. Also the weights will be saved in a csv after training. In capture, it will open users' default camera and recognize the digit with the weights saved in previous csv.
