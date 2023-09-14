# # CS 6476 Assignment 1
# 
# This is the second part of the assignment. We aim to develop an understanding of the PyTorch deep learning framework. There are two main problems in this part:   
# 1. Write your own `Linear` layer by subclassing `nn.Module`.
# 2. Train a small model by writing your own optmizer and loss functions.
# 

# ## Brief introduction to PyTorch
# 
# 
# For anyone who is new to PyTorch, here is a quick glance of the features of the framework:   
# - Numpy-like API with GPU support
# - Pythonic object-oriented programming paradigm
# - Dynamic computation graph generation and execution
# - Automatic differentiation
# 
# We will be using PyTorch for the programming assignments of this course. Thus, it is crucial to have clarity on how to build and train networks using torch.
# 
# There are three main components to any deep learning pipeline:
# 1. **Data loading and preprocessing**: PyTorch provides `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` utilities to facilitate data fetching and batch collation.
# 2.  **Model creation**: Most models can be created by subclassing `nn.Module` and overriding the `forward` method. The user does not need to implement the backprop, as it is handled by PyTorch's autograd system.
# 3. **Model training**: Torch provides optimizers and losses through the `torch.optim` and `torch.nn` modules to be used during model training.
# 
# Each of these deserves their own place in the assignment, however in this assignment, we are going to focus only on the model creation and training. I have added some extra reading material in the end in case anyone is interested.

# ## Problem 1: Writing your own `Linear` layer by subclassing `nn.Module`
# 
# In this problem, you have to code up a custom implementation of the [`Linear` layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html). We want the layer to perform the following operation:
# 
# \begin{align}
#     output = XW_1 + X^2W_2 + b \\
# \end{align}
# where $W_1$, $W_2$ and $b$ are trainable weights.
# 
# 
# 
# For this, here are the steps you need to follow:
# 1. Subclass `nn.Module` to create a custom layer class.
# 2. Initialize model parameters $W_1$, $W_2$ and $b$ using `nn.Parameter`. Use the `torch.randn()` method for initialization of tensor.
# 3. Implement the `forward` method in the module.

# Run the following command to install the required libraries:
# ```
# pip install numpy torch gdown
# ```

# ### Imports and set seed


import torch
import random
import numpy as np
import gdown

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ### Implementing `MyLinear` module


import torch
from torch import nn

class MyLinear(nn.Module):
    """
    Subclass of nn.Module.
    Input:
        input_dim: feature dimension of input to the layer
        output_dim: feature dimension of output of the layer

    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        ############## START CODE HERE
        # Warning: Be careful with tensor shapes. Make sure the weights and inputs can be multiplied.
        # Models are sensitive to weight initialization. Initialize the weights
        # from a normal distribution with zero mean and 0.01 standard deviation.

        self.w1 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(self.input_dim, self.output_dim), requires_grad=True))
        self.w2 = nn.Parameter(torch.normal(mean=0, std=0.01, size=(self.input_dim, self.output_dim), requires_grad=True))
        self.b = nn.Parameter(torch.ones(output_dim))

        ############## END CODE HERE

    def forward(self, inputs):
        """
        Implement the forward pass for the given expression.

        Inputs:
            inputs: the input tensor having the shape (batch_size, input_dim)

        Outputs:
            outputs: tensor having the shape (batch_size, output_dim)
        """


        ############## START CODE HERE
        outputs = torch.matmul(inputs, self.w1) + torch.matmul(inputs**2, self.w2) + self.b
        ############## END CODE HERE

        return outputs


# You have completed the first problem of this section.

# ## Problem 2: Train a small model by writing your own optmizer and loss functions.
# 
# In this problem, you have to implement a loss funtion and the stochastic gradient descent algorithm to solve the regression problem.
# 
# 
# Your goal for this problem is to train the model with Mean Squared Error (MSE) loss and Stochastic Gradient Descent (SGD) algorithm. The number of epochs and the learning rate for the model is fixed. The data for the model has been stored as `.pt` files, which will be loaded in `x_train` and `y_train` for you. You will also be provided unseen samples, `x_test`, and you have to use your model to predict on these samples.
# 
# To limit computational burden, the input data will have rank=4 and the output will be a single float. The training data has 500 samples, whereas the testing data has 100 samples. You must submit the predictions in a `.pt` file. We will average over the loss on testing dataset and assign you a score accordingly.

# For your reference, here are the formulae for MSE and SGD:
# 
# 
# MSE:
# \begin{align}
#     loss = \frac{1}{n}\Sigma_{i=1}^{n} (y_i - ỹ_i)^2 \\
# \end{align}
# 
# where $n$ is the number of samples, $y_i$ is the true label and $ỹ_i$ is the predicted label for the $i^{th}$ sample.
# 
# SGD:
# \begin{align}
#     θ_{t+1} := θ_{t} - α⋅\triangledown_{θ}\mathcal{L}(y, ỹ)
# \end{align}
# 
# where $θ_{t}$ and $θ_{t+1}$ are the model weights at timesteps $t$ and $t+1$ respectively, $\alpha$ is the learning rate and $\triangledown_{θ}\mathcal{L}(y, ỹ)$ is the gradient of the loss function $\mathcal{L}$ with respect to model weights.
# 

# ### Loss function


def mse_loss(y_true, y_pred):
    """
    Implement the Mean Squared Error loss function.

    Inputs:
        y_true: expected output values having shape (batch_size,)
        y_pred: predicted output values having shape (batch_size,)

    Output:
        The calculated loss for the given batch.

    """
    ############## START CODE HERE
    loss = torch.mean((y_true - y_pred)**2)
    return loss
    ############## END CODE HERE

# ### Gradient descent


def sgd_step(model, learning_rate):
    """
    Implement a single step of SGD.

    Steps:
        Iterate over all model parameters. For each model parameter:
            1. Access parameter weights using `data` attribute.
            2. Update the parameter using the SGD formula. The gradient is stored in the `grad` attribute of the parameter.
            3. (IMPORTANT) Set the `grad` attribute to zero, so the gradients are not accumulated across timesteps.

    Inputs:
        model: the model containing two `MyLinear` layers
        learning_rate: learning rate to apply SGD with (see \alpha in above equation)

    Outputs:
        None
    """
    ############## START CODE HERE
    with torch.no_grad():
        for _, param in model.named_parameters():
            param -= learning_rate * param.grad
            param.grad = None
    ############## END CODE HERE

# ### Download train and test data


def download_and_load(resource_name):
    file_ids = dict()
    file_ids["x_train"] = "1En9Il86rT2dRgeYFToYfh_2BNNwAn2VJ"
    file_ids["x_test"] = "1CrjZqUWVCKtCon5OyOeKTjFxwn4WMc3A"
    file_ids["y_train"] = "1GVcqF3R3D0gM38ivyD0t2xau7rqAo5Yf"

    output = resource_name + ".pt"
    gdown.download(id=file_ids[resource_name], output=output, quiet=False)

    asset = torch.load(output)
    return asset

# ### Training the model

# Here I have loaded the training and testing data and initialized the model.  


x_train = download_and_load("x_train")
y_train = download_and_load("y_train")

x_test = download_and_load("x_test")

model = nn.Sequential(MyLinear(4, 5), MyLinear(5, 1))
# More info on nn.Sequential: pytorch.org/docs/stable/generated/torch.nn.Sequential.html

# Write the code for one epoch of training below.
# 
# Note that we are not training on minibatches, which is the norm in modern deep learning. We will address this issue in the upcoming assignments.
# 
# We will train the model for 300 epochs with learning rate 0.003.


EPOCHS = 300
LEARNING_RATE = 0.003

def train(x_train, y_train, model, epochs, learning_rate):
    """
    Implement model training.

    Steps:
        Iterate over number of epochs. For each epoch:
            1. Get model predictions.
            2. Calculate loss between model predictions and ground truth labels.
            3. Backpropagate the loss.
            4. Perform SGD step.

    Inputs:
        x_train: model input for training samples
        y_train: expected model output for `x_train`
        model: the model containing two `MyLinear` layers
        epochs: number of epochs to train the model for
        learning_rate: learning rate to apply SGD with (see \alpha in above equation)

    Outputs:
        None
    """
    ############## START CODE HERE
    for epoch in range(epochs):
        y_pred = model.forward(x_train)
        loss = mse_loss(y_train, y_pred)
        print(loss)
        loss.backward()
        sgd_step(model, learning_rate)
    ############## END CODE HERE

# The training loss should be less then 0.0025. 
# 
# **Evaluation will be on the basis of test loss only, training loss is given only for reference.**


def predict(model, x_test):
    """
    Implement model prediction.

    Inputs:
        model: the trained model containing two `MyLinear` layers
        x_test: model input for testing samples

    Outputs:
        model predictions
    """

    ############## START CODE HERE
    predictions = model.forward(x_test)
    ############## END CODE HERE
    return predictions

# Here are some additional resources:
# - [How autograd works](https://www.youtube.com/watch?v=MswxJw-8PvE)
# - [`torch.utils.data`](https://pytorch.org/docs/stable/data.html)

train(x_train, y_train, model, EPOCHS, LEARNING_RATE)
