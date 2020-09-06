### Transformer architecture

Transformer was introduced by ==Vaswani et al. (2017)==

- which based solely on the **attention** mechanisms and 
- feed-forward neural networks. 
- 特征提取功能 （取代CNN,RNN 的特征提取）

A Transformer consists of two main components: 

- an encoding component and 

  All the encoders have the same structure with **different** **weights** and they consist of two sub-layers: 

  - multi-head self-attention mechanism (Multi-Head Attention) and 

  - position-wise feed-forward neural network (Feed Forward). 

    The position-wise feed-forward neural network is applied to each input position independently and therefore, can be executed in parallel. Residual connection and layer normalisation are applied to each of the two sub-layers (Add & Norm).

- a decoding component. 



<img src="https://ryanong.co.uk/wp-content/uploads/2020/01/Transformers.png" alt="img" style="zoom:50%;" />



adding vectors (known as positional encodings) to each of the input embeddings

These vectors will help the model determine the relative or absolute position of each word, 

enabling the model to distinguish the same word at different positions. In Vaswani et al. (2017), 

positional encodings were computed using the sine and cosine functions.



Although the decoder has a very similar structure as the encoder, they do differ in two ways.



1. Firstly, its ==multi-head== **self-attention** layer is masked to ensure that the decoder is only able to attend to earlier positions when generating output sequences, thereby, preserving the property of auto-regressive. 
2. Secondly, the decoder has a third sub-layer, where it performs multi-head attention over the output of the encoder stack, similar to the typical **attention** **mechanisms** in **seq2seq** **models** described in the previous section. In other words, this layer behaves like a normal multi-head self-attention layer except it is taking the output of the previous sub-layer (masked multi-head attention) as a query matrix Q and the key and value matrix K and Q from the output of the encoder. This allows the decoder to attend over all the positions in the sequence to generate its own context values to be fed into the feed-forward neural network.

The output of the decoder stack is a **vector**, which we **feed through** to the **linear layer**, followed by a **softmax layer**. 

- The linear layer is the typical **affine** layer that transforms the vector into logits vector, where each cell corresponds to the score of a unique word. This means that the logits vector has the same size as our vocabulary size. These scores will then be converted into probabilities by our softmax layer and the cell (word) with the highest probability will be the output for this decoding step.



==WHAT ARE THE MECHANICS OF A SEQUENCE-2-SEQUENCE MODEL?==

A **seq-2-seq model** consists of an encoder and a decoder. Encoder and decoder can be same or different architectures (RNN / CNN / Transformers). 

The encoder process the **input sequence** and **captures** all the information into a context **vector**. This context vector is the **last hidden** state of the encoder and is pass to the decoder to be used to generate output sequence. The bottleneck of this seq-2-seq model is that it requires the context vector to hold lots of information and this could be challenging if our input sentences are very **long**. In the case of machine translation, if you were to translate a 100 word sentence, by the time you pass the context vector (100th hidden state) to the decoder, the decoder need to be able to access the information encoded in hidden state 1 or 2 to translate accurately.



HOW DO WE FIX THIS BOTTLENECK?

**Attention mechanism** was introduced to fix this issue. Attention allows the model to attend to the relevant parts when decoding! There are two main difference if attention model:

1. The encoder **passes all the hidden states** to the decoder rather than just the last hidden state
2. The decoder generates context vector at EACH time step, allowing the model to attend to relevant parts! Each encoder hidden state is given a score and these scores are softmaxed where the hidden state with the highest score (most relevant part) dominates. The final context vector is the weighted sum of all the encoder hidden states. The context vector is concatenated with current decoder hidden state and it’s feed into a FFNN to predict the output word at this time step

<img src="https://ryanong.co.uk/wp-content/uploads/2020/07/attention.png" alt="img" style="zoom:60%;" />



==DESCRIBE THE TRANSFORMER ARCHITECTURE.==

The transformer architecture has two components: 

- the encoders and 
- the decoders. 

The **encoders** component consists of a stack of encoders (6 in the original paper) and the decoders component consists of a stack of decoders (6 in the original paper).

The encoders have two sub-layers: 

- Self-attention layer and 

- the FFNN. 

The self-attention layer allows the encoder to look at other words in the sentence to generate word embeddings for each word (**contextualised**). The output of the self-attention layer is feed into the FFNN. Each word position has its own path in the encoder. The self-attention layer has dependencies between words but the FFNN are independent for each word position.

The **decoders** have both the 

- self-attention and 
- FFNN sub-layers, with 
- an additional encoder-decoder attention layer in between. 

This encoder-decoder attention layer allows the decoder to attend to the relevant parts of the sentence, similar to what we describe in the previous question.



The **flow** of the **encoder** is as follows:

1. Convert tokens to embeddings
2. Add **positional encoding** to the embeddings
3. The embeddings go through the self-attention layer to encode each word using other words in the sentence
4. The output of the self-attention layer goes through the residual connection and layer normalisation step
5. The output of the normalisation layer goes through the FFNN (independently)
6. The output of the FFNN layer goes through the residual connection and layer normalisation step
7. The output of the normalisation layer is pass to the next encoder and the whole process is repeated



==What is positional encoding for?==

The positional encoding is to **capture the order** of the words in the input sequence.

==What is residual connection and layer normalisation for?==

The output of self-attention layer and FFNN layer goes through the residual connection and layer normalisation layer. This involves concatenating the output of each layer with the input embeddings and feed it through layer normalisation. This applies to the decoder layers.

==How does self-attention works?==

1. For each word vector, we would create three **vectors**: **query, key, and values**. These vectors are created by multiplying the input vector with three different matrices trained during the training process
2. For each **word vector**, calculate the **attention score** of each word in the sentence against the word vector using the query and key vectors. The score indicates how much focus to place on other words when encoding the word vector. The score is the dot product of the query and key vector
3. Perform **normalisation** and softmax operation on the score
4. Multiply each value vector with its respective softmax value. This is to focus more on the words we care about and drown-out irrelevant words
5. Weighted sum these vectors to create the final output vector for the given word vector

Each word vector in the sentence would have an output vector post the self-attention layer and these output vectors are pass to the FFNN independently.



==What’s multi-headed attention then?==

The query, key, and value vectors are created using the query, key, and value matrices. Multi-headed attention means we have **multiple set of** query, key, and **value weight matrices** and each of these sets is **randomly initialised**. 

In the original transformer paper, there are 8-headed attention meaning that we have 8 different sets of query, key, and value weight matrices, resulting in 8 output matrices from the self-attention layer. To pass these 8 output matrices to the FFNN (only expecting one matrix), we would concatenate these 8 output matrices together and multiply by an output weight matrix to get the final aggregate output matrix for our FFNN.



==The decoder==

The decode is very similar to the encoder except the following:

1. The self-attention layer can only attend to earlier positions in the output sequence so that we don’t leak future information. This is done so through masking
2. The encoder-decoder attention behaves similarly to the multi-head attention layer except it creates the queries matrix from the previous layer and the keys and values matrix are from the output of the encoder stack

The output of the decoder is feed into a linear layer with softmax over the vocabulary to predict which word has the highest probability given the previous decoded sequence.



###### Source:

- ##### https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

- ##### http://jalammar.github.io/illustrated-transformer/