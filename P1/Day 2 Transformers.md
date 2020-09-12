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



###### [手把手教你用Pytorch-Transformers——部分源码解读及相关说明（一）](https://www.cnblogs.com/dogecheng/p/11907036.html)

> github：https://github.com/huggingface/transformers

BertConfig 是一个配置类，存放了 BertModel 的配置。比如：

- **vocab_size_or_config_json_file：**字典大小，默认30522
- **hidden_size：**Encoder 和 Pooler 层的大小，默认768
- **num_hidden_layers：**Encoder 的隐藏层数，默认12
- **num_attention_heads：**每个 Encoder 中 attention 层的 head 数，默认12

###### [BertModel](https://www.cnblogs.com/dogecheng/p/11907036.html)

实现了基本的Bert模型，从构造函数可以看到用到了embeddings，encoder和pooler。

下面是允许输入到模型中的参数，模型至少需要有1个输入： input_ids 或 input_embeds。

- **input_ids** 就是一连串 token 在字典中的对应id。形状为 (batch_size, sequence_length)。

- **token_type_ids** 可选。就是 token 对应的句子id，值为0或1（0表示对应的token属于第一句，1表示属于第二句）。形状为(batch_size, sequence_length)。

  

Bert 的输入需要用 [CLS] 和 [SEP] 进行标记，开头用 [CLS]，句子结尾用 [SEP]

两个句子：

  tokens：[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]

  **token_type_ids：0  0 0  0  0   0    0  0  1 1 1 1  1  1**

一个句子：

  tokens：[CLS] the dog is hairy . [SEP]

  token_type_ids：0  0  0  0 0   0  0



###### BertForQuestionAnswering[#](https://www.cnblogs.com/dogecheng/p/11907036.html#4085574841)

###### tokenization相关[#](https://www.cnblogs.com/dogecheng/p/11907036.html#4027244739)

对于文本，常见的操作是分词然后将 **词-id** 用字典保存，再将分词后的词用 id 表示，然后经过 Embedding 输入到模型中。

Bert 也不例外，但是 **Bert 能以 字级别 作为输入**，在处理中文文本时我们可以不用先分词，直接用 Bert 将文本转换为 token，然后用相应的 id 表示。

tokenization 库就是用来将文本切割成为 字或词 的，下面对其进行简单的介绍

**BasicTokenizer**

基本的 tokenization 类，构造函数可以接收以下3个参数

- **do_lower_case：**是否将输入转换为小写，默认True
- **never_split：**可选。输入一个列表，列表内容为不进行 tokenization 的单词
- **tokenize_chinese_chars：**可选。是否对中文进行 tokenization，默认True

##### WordpieceTokenizer

**tokenize()函数**

这个类的 tokenize() 函数使用 **贪婪最长匹配优先算法**（greedy longest-match-first algorithm） 将一段文本进行 tokenization ，变成相应的 wordpiece，**一般针对英文**

> example ：
>
> ​     input = "unaffable" → output = ["un", "##aff", "##able"]
>
>  # 它将 “unaffable” 分割成了 “un”, “##aff” 和 “##able”



##### BertTokenizer

一个专为 Bert 使用的 tokenization 类，使用 Bert 的时候一般情况下用这个就可以了，构造函数可以传入以下参数

- **vocab_file：**一个字典文件，每一行对应一个 wordpiece
- **do_lower_case：**是否将输入统一用小写表示，默认True
- **do_basic_tokenize：**在使用 WordPiece 之前是否先用 BasicTokenize
- **max_len：**序列的最大长度
- **never_split：**一个列表，传入不进行 tokenization 的单词，只有在 do_wordpiece_only 为 False 时有效

我们可以使用 **tokenize() 函数对文本进行 tokenization**，也可以通过 **encode() 函数对 文本 进行 tokenization 并将 token 用相应的 id 表示**，然后输入到 Bert 模型中



使用 **encode()** 函数将 tokenization 后的内容用相应的 id 表示，主要由以下参数：

**注意 encode 只会返回 ==token id==**，Bert 我们还需要输入==句子 id==，这时候我们可以使用 **encode_plus()**，它返回 token id 和 句子 id

encode() 实际上就是用了 encode_plus，但是只选择返回 token_id，代码如下

```python
encoded_inputs = self.encode_plus(text,
                                  text_pair=text_pair,
                                  max_length=max_length,
                                  add_special_tokens=add_special_tokens,
                                  stride=stride,
                                  truncation_strategy=truncation_strategy,
                                  return_tensors=return_tensors,
                                  **kwargs)

return encoded_inputs["input_ids"]
```

