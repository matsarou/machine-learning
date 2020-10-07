<h2>Classification task on the CIFAR-10 dataset.</h2>
<p>The CIFAR consists of 60,000 colored images, divided in the 10 following classes: Airplanes, Cars, Birds,Cats,
Deer, Dogs, Frogs, Horses, Ships, Trucks. The dataset has 6,000 images of each class, that means that is is balanced.
Data preparation is required because we cannot directly feed the images to the network, as the
results of the classification will be very poor. The data preprocessing consists of these steps:</p>
<ul>
<li>1. scaling the input image pixels(x_train and x_test) to the range [0, 1]</li>
<li>2. converting the labels(y_train and y_test) from strings to integers.</li>
<li>3. transforming the integer labels into vectors in the range [0, num_classes=10], using
the keras. This is the same thing as if I used a OneHotEncoder.</li>
</ul>
<h3>Architecture of the neuron network</h3>
<p>I created a standard, feed forward, or else fully-connected neural network,with the
Sequential class, which represents a linear stack of layers. The layers are listed here in
order:</p>
<p>1. one Flatten layer, before the input layer. I flatten each 32x32x3 image into a 1024
dimensional vector, which I then feed it as input to my neural network</p>
<p>2. the input Dense layer.</p>
<p>3. two hidden Dense layers of neurons. (depth of the network=2) Each hidden layer has
484 units. (width of the layer=484)</p>
<p>4. one Dropout layer between the two hidden layers, to perform regularization and
prevent overfitting</p>
<p>5. the output layer with 10 neurons or units, one for each class. The output layer gives
the final estimate of the output.</p>
<h3>Activation functions:</h3>
<p>● reLU for the hidden layers, to generate a continuous output</p>
<p>● softmax for the output layer, because I want the data points being mutually
exclusive(one data point can belong only to one class).</p>
<h3>Regularization with Data augmentation</h3>
<p>Image data augmentation is a regularization technique. It is used to expand the training
dataset in order to improve the performance and ability of the model to generalize. It didn’t
bring any improvement in the model’s performance, so I don’t include it in this report.</p>
<h3>How the network learns?</h3>
<p>It uses the cost function, the gradient descent and the back propagation, in order to find the
minimum weights that give the minimum cost.</p>
<p>Before I train the model, I compile it, passing the optimizer, the loss function and the metrics.</p>
<h3> Loss function</h3>
<p>I use the categorical_crossentropy loss because the task is a classification task.
The metrics that I want to report, is the ‘accuracy’.</p>
<p>The optimizer uses the gradient descent to find the best weights and biases that minimize
the value of the loss function. A good learning rate must be discovered via trial and error.</p>
<h3>Optimizers</h3>
<p>I experimented with:</p>
<p><b>1. Stochastic Gradient Descent optimizer</b>. The hyperparameters that can be optimized
in SGD are learning rate, momentum, decay and nesterov.</p>
<p><b>2. Adaptive gradient descent algorithms</b>: an optimizing way performing the gradient
descent, which adapts the learning rate as it goes, starting from larger values and go
smaller, as the slope gets closer to zero.</p>
<p>Before I train the model, I must configure the following parameters:</p>
<p><b>1. epochs = </b>time for a training period</p>
<p><b>2. batch_size= </b>how many images will i feed to the network at once</p>
<p>By enlarging the batch size, I succeed better memory utilization and greater efficiency.
On the other hand, too large of a batch size leads to poor generalization.
The relation between one epoch and number of iterations and batch size is the following:</p>
<p>One Epoch = Numbers of Iterations = Train Data / batch size</p>
<p>Larger batch size means fewer iterations and faster processing speed.</p>
</p><b>3. momentum</b></p>
<h3>Results</h3>
<p>In general:</p>
<p>● smaller learning rates require more training epochs</p>
<p>● larger learning rates require fewer training epochs</p>
<p>● smaller batch sizes are better suited to smaller learning rates</p>
<p>I fit the model, passing the optimizer, the epochs, the momentum and the batch_size. I set
shuffle=True, which shuffles the images, change their order, for better generalization(the
network does not get the same images in order, cats then dogs etc.)</p>
<p>I also use the testing dataset for validation during training. Keras will evaluate the model on
the validation set at the end of each epoch and report the loss and any metrics I asked for.</p>
<p>Experiments with Stochastic Gradient Descent (SGD optimizer)</p>