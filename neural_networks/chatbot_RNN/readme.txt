<h2>Encoder</h2>
<p>I use bidirectional GRU type of RNN as it has been approved that they have better results than LSTMs</p>
<h3>Parameters in GRU RNN</h3>
<p>input_size = hidden_size (number of features). Equals to the size of word embeddings.</p>
<p>n_layers = number of layers in RNN</p>
<h3>Word Embeddings</h3>
<p>Iproduce one embedding per word. The module that allows to use embeddings is torch.nn.Embedding</p>
<h2>Decoder</h2>
<p>I build a decoder using “Global attention”, specifically the Luong attention mechanism. The used method is the dot_score.</p>
<h3>Parameters in Decoder</h3>
<p>input_step = one time step (one word) of input sequence batch</p>
<p>last_hidden = final hidden state of encoder</p>
<p>encoder_outputs = encoder model’s output</p>
<h2>Train the model</h2>
<h3>Define Loss function</h3>
<p>I define the loss function that calculates the cost. Some batches of words, which are shorter than the the maximum size contain zeros.
We don’t want to consider the zeros when calculating the loss.</p>
<h3>Run Training</h3>
<p>We construct training batches from the training sequence. We train with teacher forcing.
Instead of feeding the previously generated input, into the bottom of the decoder, we feed the previous true word.</p>
<h3>Evaluation>/h3>
<p>I used Greedy search model that searches through all the possible output sequences based on their likelihood.</p>