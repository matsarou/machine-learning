Introduction
A classification task on the CIFAR-10 dataset, using a simple feed forward network, or else fully-connected neural network.
The CIFAR consists of 60,000 colored images, divided in the 10 following classes: Airplanes, Cars, Birds,Cats,
Deer, Dogs, Frogs, Horses, Ships, Trucks. The dataset has 6,000 images of each class, that means that is is balanced.

Data preparation
Data preparation is required because we cannot directly feed the images to the network, as the
results of the classification will be very poor. The data preprocessing consists of these steps:
1. scaling the input image pixels(x_train and x_test) to the range [0, 1]
2. converting the labels(y_train and y_test) from strings to integers.
3. transforming the integer labels into vectors in the range [0, num_classes=10], using the keras. This is the same thing as if I used a OneHotEncoder.


Architecture of the neuron network
A linear stack of layers:
1. one Flatten layer, before the input layer. I flatten each 32x32x3 image into a 1024
dimensional vector, which I then feed it as input to my neural network</p>
2. the input Dense layer.
3. two hidden Dense layers of neurons. (depth of the network=2) Each hidden layer has 484 units. (width of the layer=484)
4. one Dropout layer between the two hidden layers, to perform regularization and prevent overfitting
5. the output layer with 10 neurons or units, one for each class. The output layer gives the final estimate of the output.

Activation functions:
● reLU for the hidden layers, to generate a continuous output
● softmax for the output layer, because I want the data points being mutually exclusive(one data point can belong only to one class).

Regularization with Data augmentation
Image data augmentation is a regularization technique. It is used to expand the training dataset in order to improve the performance and ability of the model to generalize. It didn’t bring any improvement in the model’s performance, so I don’t include it in this report.

How the network learns?
It uses the cost function, the gradient descent and the back propagation, in order to find the minimum weights that give the minimum cost
Before I train the model, I compile it, passing the optimizer, the loss function and the metrics.

Loss function
I use the categorical_crossentropy loss because the task is a classification task. The metrics that I want to report, is the ‘accuracy’.
The optimizer uses the gradient descent to find the best weights and biases that minimize the value of the loss function. A good learning rate must be discovered via trial and error.

Optimizers
I experimented with:
1. Stochastic Gradient Descent optimizer</b>. The hyperparameters that can be optimized in SGD are learning rate, momentum, decay and nesterov.
2. Adaptive gradient descent algorithms</b>: an optimizing way performing the gradient descent, which adapts the learning rate as it goes, starting from larger values and go smaller, as the slope gets closer to zero.

Configuration
Before I train the model, I must configure the following parameters:
1. epochs = </b>time for a training period</p>
2. batch_size= how many images will i feed to the network at once. By enlarging the batch size, I succeed better memory utilization and greater efficiency.
On the other hand, too large of a batch size leads to poor generalization.
The relation between one epoch and number of iterations and batch size is the following: One Epoch = Numbers of Iterations = Train Data / batch size
Larger batch size means fewer iterations and faster processing speed.</p>
3. momentum

Results
In general:
● smaller learning rates require more training epochs
● larger learning rates require fewer training epochs
● smaller batch sizes are better suited to smaller learning rates

I fit the model, passing the optimizer, the epochs, the momentum and the batch_size. I set shuffle=True, which shuffles the images, change their order, for better generalization(the network does not get the same images in order, cats then dogs etc.)
<p>I also use the testing dataset for validation during training. Keras will evaluate the model on the validation set at the end of each epoch and report the loss and any metrics I asked for.
