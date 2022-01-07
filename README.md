Pls ref:https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

The dataset was originied from Dr. Jason Brownlee's website but the code was modified accordingly.
Most of the description was copied directly from the above web linkage.

# keras_first_network
Keras is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models.  It wraps the efficient numerical computation libraries Theano and TensorFlow and allows you to define and train neural network models in just a few lines of code.  In this tutorial, you will discover how to create your first deep learning neural network model in Python using Keras.  Kick-start your project with my new book Deep Learning With Python, including step-by-step tutorials and the Python source code files for all examples.

Keras Tutorial Overview
There is not a lot of code required, but we are going to step over it slowly so that you will know how to create your own models in the future.

The steps you are going to cover in this tutorial are as follows:

1.Load Data.
2.Define Keras Model.
3.Compile Keras Model.
4.Fit Keras Model.
5.Evaluate Keras Model.
6.Tie It All Together.
7.Make Predictions

This Keras tutorial has a few requirements:

You have Python 2 or 3 installed and configured.
You have SciPy (including NumPy) installed and configured.
You have Keras and a backend (Theano or TensorFlow) installed and configured.
If you need help with your environment, see the tutorial:

How to Setup a Python Environment for Deep Learning
https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/

Create a new file called keras_first_network.py and type or copy-and-paste the code into the file as you go.

1. Load Data
The first step is to define the functions and classes we intend to use in this tutorial.
We will use the NumPy library to load our dataset and we will use two classes from the Keras library to define our model.

We can now load our dataset.

In this Keras tutorial, we are going to use the Pima Indians onset of diabetes dataset. This is a standard machine learning dataset from the UCI Machine Learning repository. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

As such, it is a binary classification problem (onset of diabetes as 1 or not as 0). All of the input variables that describe each patient are numerical. This makes it easy to use directly with neural networks that expect numerical input and output values, and ideal for our first neural network in Keras.

Download the dataset and place it in your local working directory, the same location as your python file.


There are eight input variables and one output variable (the last column). We will be learning a model to map rows of input variables (X) to an output variable (y), which we often summarize as y = f(X).

The variables can be summarized as follows:

Input Variables (X):

1.Number of times pregnant
2.Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3.Diastolic blood pressure (mm Hg)
4.Triceps skin fold thickness (mm)
5.2-Hour serum insulin (mu U/ml)
6.Body mass index (weight in kg/(height in m)^2)
7.Diabetes pedigree function
8.Age (years)

Output Variables (y):

1.Class variable (0 or 1)

Once the CSV file is loaded into memory, we can split the columns of data into input and output variables.

The data will be stored in a 2D array where the first dimension is rows and the second dimension is columns, e.g. [rows, columns].

We can split the array into two arrays by selecting subsets of columns using the standard NumPy slice operator or “:” We can select the first 8 columns from index 0 to index 7 via the slice 0:8. We can then select the output column (the 9th variable) via index 8.


We are now ready to define our neural network model.

Note, the dataset has 9 columns and the range 0:8 will select columns from 0 to 7, stopping before index 8. 


2. Define Keras Model

Models in Keras are defined as a sequence of layers.

We create a Sequential model and add layers one at a time until we are happy with our network architecture.

The first thing to get right is to ensure the input layer has the right number of input features. This can be specified when creating the first layer with the input_dim argument and setting it to 8 for the 8 input variables.

How do we know the number of layers and their types?

This is a very hard question. There are heuristics that we can use and often the best network structure is found through a process of trial and error experimentation (I explain more about this here). Generally, you need a network large enough to capture the structure of the problem.

In this example, we will use a fully-connected network structure with three layers.

Fully connected layers are defined using the Dense class. We can specify the number of neurons or nodes in the layer as the first argument, and specify the activation function using the activation argument.

We will use the rectified linear unit activation function referred to as ReLU on the first two layers and the Sigmoid function in the output layer.

It used to be the case that Sigmoid and Tanh activation functions were preferred for all layers. These days, better performance is achieved using the ReLU activation function. We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

We can piece it all together by adding each layer:
1.The model expects rows of data with 8 variables (the input_dim=8 argument)
2.The first hidden layer has 12 nodes and uses the relu activation function.
3.The second hidden layer has 8 nodes and uses the relu activation function.
4.The output layer has one node and uses the sigmoid activation function.

Note, the most confusing thing here is that the shape of the input to the model is defined as an argument on the first hidden layer. This means that the line of code that adds the first Dense layer is doing 2 things, defining the input or visible layer and the first hidden layer.

3. Compile Keras Model

Now that the model is defined, we can compile it.

Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as Theano or TensorFlow. The backend automatically chooses the best way to represent the network for training and making predictions to run on your hardware, such as CPU or GPU or even distributed.

When compiling, we must specify some additional properties required when training the network. Remember training a network means finding the best set of weights to map inputs to outputs in our dataset.

We must specify the loss function to use to evaluate a set of weights, the optimizer is used to search through different weights for the network and any optional metrics we would like to collect and report during training.

In this case, we will use cross entropy as the loss argument. This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“. You can learn more about choosing loss functions based on your problem here:
https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

We will define the optimizer as the efficient stochastic gradient descent algorithm “adam“. This is a popular version of gradient descent because it automatically tunes itself and gives good results in a wide range of problems. To learn more about the Adam version of stochastic gradient descent see the post:
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

Finally, because it is a classification problem, we will collect and report the classification accuracy, defined via the metrics argument.

4. Fit Keras Model
We have defined our model and compiled it ready for efficient computation.

Now it is time to execute the model on some data.

We can train or fit our model on our loaded data by calling the fit() function on the model.

Training occurs over epochs and each epoch is split into batches.

Epoch: One pass through all of the rows in the training dataset.
Batch: One or more samples considered by the model within an epoch before weights are updated.

One epoch is comprised of one or more batches, based on the chosen batch size and the model is fit for many epochs. For more on the difference between epochs and batches, see the post:

https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the epochs argument. We must also set the number of dataset rows that are considered before the model weights are updated within each epoch, called the batch size and set using the batch_size argument.

For this problem, we will run for a small number of epochs (150) and use a relatively small batch size of 10.

These configurations can be chosen experimentally by trial and error. We want to train the model enough so that it learns a good (or good enough) mapping of rows of input data to the output classification. The model will always have some error, but the amount of error will level out after some point for a given model configuration. This is called model convergence.

This is where the work happens on your CPU or GPU.

No GPU is required for this example, but if you’re interested in how to run large models on GPU hardware cheaply in the cloud, see this post:

https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/


5. Evaluate Keras Model
We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset.

This will only give us an idea of how well we have modeled the dataset (e.g. train accuracy), but no idea of how well the algorithm might perform on new data. We have done this for simplicity, but ideally, you could separate your data into train and test datasets for training and evaluation of your model.

You can evaluate your model on your training dataset using the evaluate() function on your model and pass it the same input and output used to train the model.

This will generate a prediction for each input and output pair and collect scores, including the average loss and any metrics you have configured, such as accuracy.

The evaluate() function will return a list with two values. The first will be the loss of the model on the dataset and the second will be the accuracy of the model on the dataset. We are only interested in reporting the accuracy, so we will ignore the loss value.

6. Tie It All Together
You have just seen how you can easily create your first neural network model in Keras.

Let’s tie it all together into a complete code example.


You can copy all of the code into your Python file and save it as “keras_first_network.py” in the same directory as your data file “pima-indians-diabetes.csv“. You can then run the Python file as a script from your command line (command prompt).

Running this example, you should see a message for each of the 150 epochs printing the loss and accuracy, followed by the final evaluation of the trained model on the training dataset.

It takes about 10 seconds to execute on my workstation running on the CPU.

Ideally, we would like the loss to go to zero and accuracy to go to 1.0 (e.g. 100%). This is not possible for any but the most trivial machine learning problems. Instead, we will always have some error in our model. The goal is to choose a model configuration and training configuration that achieve the lowest loss and highest accuracy possible for a given dataset.

7. Make Predictions

We can adapt the above example and use it to generate predictions on the training dataset, pretending it is a new dataset we have not seen before.

Making predictions is as easy as calling the predict() function on the model. We are using a sigmoid activation function on the output layer, so the predictions will be a probability in the range between 0 and 1. We can easily convert them into a crisp binary prediction for this classification task by rounding them.

Cheers! Pls enjoy the open source code



















