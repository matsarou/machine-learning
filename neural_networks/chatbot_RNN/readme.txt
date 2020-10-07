Encoder
I use bidirectional GRU type of RNN as it has been approved that they have better results than LSTMs

Parameters in GRU RNN
1. input_size = hidden_size (number of features). Equals to the size of word embeddings.
2. n_layers = number of layers in RNN

Word Embeddings
Iproduce one embedding per word. The module that allows to use embeddings is torch.nn.Embedding


Decoder
I build a decoder using “Global attention”, specifically the Luong attention mechanism. The used method is the dot_score.

Parameters in Decoder
1. input_step = one time step (one word) of input sequence batch<
2. last_hidden = final hidden state of encoder
3. encoder_outputs = encoder model’s output


Train the model

Define Loss function
I define the loss function that calculates the cost. Some batches of words, which are shorter than the the maximum size contain zeros.
We don’t want to consider the zeros when calculating the loss.

Run Training
We construct training batches from the training sequence. We train with teacher forcing.
Instead of feeding the previously generated input, into the bottom of the decoder, we feed the previous true word.

Evaluation
I used Greedy search model that searches through all the possible output sequences based on their likelihood.
