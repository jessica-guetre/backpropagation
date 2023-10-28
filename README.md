## Part 1

### Building Model1
Uses Pytorch to implement a three-layer Neural Network (input layer - hidden layer - output layer) and update the weights with backpropagation.
- 1. Implement forward and calculate the output
- 2. Calculate errors and loss  
- 3. Update the weights with backpropagation 
- 4. Predict function 
- 5. Activation function (Sigmoid function) 

### Evaluator Function (1 point)  
Implements the evaluator function with Pytorch or Numpy only   
- Evaluation metrics include confusion matrix, accuracy, recall score, precision and F1 score

### Training and Evaluating Model1 (1 point)  
Trains Model1 with customized hidden size, learning rate, number of iterations and batch size.
Uses the predict function to predict the labels with the test dataset.
Evaluates the prediction results.
- Evaluation metrics include confusion matrix, accuracy, recall score, precision and F1 score

## Part 2
Using another machine learning framework (**scikit-learn, Tensorflow and Pytorch**) to build MLP
e.g. 
  1. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
  2. https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
  3. https://pytorch.org/tutorials/beginner/examples_nn/polynomial_nn.html#sphx-glr-beginner-examples-nn-polynomial-nn-py
  
### Building Model2-1
Implements Model2-1 with the same hidden nodes and optimization function as the model in Part 1.
Train and validate model. Use the best model on validation dataset to test on the test dataset.

### Training and Evaluating Model2-1
Evaluates the prediction results  
- Evaluation metrics include confusion matrix, accuracy, recall score, precision and F1 score.

### Building Model2-2
Adds one more hidden layer (2 hidden layers in total) to the model.
Describes Model2-2 (number of hidden nodes)  
Trains and validate model. Uses the best model on validation dataset to test on the test dataset.

### Training and Evaluating Model2-2
Evaluates the prediction results  
- Evaluation metrics include confusion matrix, accuracy, recall score, precision and F1 score.
