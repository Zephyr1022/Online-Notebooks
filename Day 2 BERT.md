### Transformer2

- RNN CNN self-attention
- Transformer由论文《Attention is All You Need》提出

Attention is All You Need：

<img src="https://picb.zhimg.com/80/v2-c85c8c8ec423d6c9333e313adcee4934_1440w.jpg" alt="img" style="zoom:35%;" /><img src="https://pic1.zhimg.com/80/v2-1706d9c0984be75b9017b425fcdc9784_1440w.jpg" alt="img" style="zoom:33%;" />

<img src="https://pic4.zhimg.com/80/v2-f5e99be76f0727be85df0d8f4ab88057_1440w.jpg" alt="img" style="zoom:33%;" />

自注意力层的输出会传递到前馈（feed-forward）神经网络中

<img src="https://picb.zhimg.com/80/v2-1cfd35f0ff43407e25da3ab25631f82d_1440w.jpg" alt="img" style="zoom:33%;" />

首先将每个输入单词通过词嵌入算法转换为词向量。

<img src="https://pic4.zhimg.com/80/v2-1f54ae99e2edaa4471a0f0f111e6dea5_1440w.png" alt="img" style="zoom:50%;" />

每个单词都被嵌入为512维的向量，我们用这些简单的方框来表示这些向量, 向量列表大小是我们可以设置的超参数——一般是我们训练集中最长句子的长度。

<img src="https://pic4.zhimg.com/80/v2-d7b0bb93c9f7e7185d690b6df83d8859_1440w.jpg" alt="img" style="zoom:33%;" />



然后我们将以一个更短的句子为例，看看编码器的每个子层中发生了什么。

如上述已经提到的，一个编码器接收向量列表作为输入，接着将向量列表中的向量传递到自注意力层进行处理，然后传递到前馈神经网络层中，将输出结果传递到下一个编码器中。

<img src="https://pic2.zhimg.com/80/v2-7173f8fa4d601a5255af46d48e9d370d_1440w.jpg" alt="img" style="zoom:33%;" />



例如，下列句子是我们想要翻译的输入句子：

> The animal didn't cross the street because it was too tired

这个“it”在这个句子是指什么呢？它指的是street还是这个animal呢？这对于人类来说是一个简单的问题，但是对于算法则不是。

当模型处理这个单词“it”的时候，自注意力机制会允许“it”与“animal”建立联系。

随着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码。



<img src="https://pic3.zhimg.com/80/v2-bac717483cbeb04d1b5ef393eb87a16d_1440w.jpg" alt="img" style="zoom:50%;" />

然后通过softmax传递结果。softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。

这个softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。显然，已经在这个位置上的单词将获得最高的softmax分数，但有时关注另一个与当前单词相关的单词也会有帮助。

第五步是将每个值向量乘以softmax分数(这是为了准备之后将它们求和)。这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)。

第六步是对加权值向量求和（译注：自注意力的另一种解释就是在编码某个单词时，就是将所有单词的表示（值向量）进行加权求和，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。），然后即得到自注意力层在该位置的输出(在我们的例子中是对于第一个单词)。

<img src="https://pic4.zhimg.com/80/v2-609de8f8f8e628e6a9ca918230c70d67_1440w.jpg" alt="img" style="zoom:50%;" />

Matrix ~ Attention

第一步是计算查询矩阵、键矩阵和值矩阵。

<img src="https://picb.zhimg.com/80/v2-ca0c37cadf0a817e4836aa5a985bf1b6_1440w.jpg" alt="img" style="zoom:50%;" />

x矩阵中的每一行对应于输入句子中的一个单词。我们再次看到词嵌入向量 (512，或图中的4个格子)和q/k/v向量(64，或图中的3个格子)的大小差异。

<img src="https://pic3.zhimg.com/80/v2-7a954ddb05bee33c7c8a9c7a99fe7b6e_1440w.jpg" alt="img" style="zoom:50%;" />



**最终的线性变换和Softmax层**

解码组件最后会输出一个实数向量。我们如何把浮点数变成一个单词？这便是线性变换层要做的工作，它之后就是Softmax层。

<img src="https://pic1.zhimg.com/80/v2-ee4d965eab70c86f6f3ac025dc826e21_1440w.jpg" alt="img" style="zoom:50%;" />



Generating Wikipedia by Summarizing Long Sequences



Image Transformer https://arxiv.org/abs/1802.05751



<img src="https://2.bp.blogspot.com/-qRz-hnwUdY4/WulXSQ6Rv4I/AAAAAAAATvQ/shk7KsphA0c3E3nUMsDVASqYaH0PhLPNwCK4BGAYYCw/s1600/GoogleAI_logo_horizontal_color_rgb.png" alt="img" style="zoom:10%;" />

https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html







fine-tuned BERT and perform inference on



**this pipeline is a trained BERT fake news classifier!**

Bert -> Relation Extraction

-> ==semantic role labelling==

WHAT IS SEMANTIC ROLE LABELLING?

- extract predicate-argument structure of a sentence
- find out what’s the event, when did the event happen, who’s involve, where did it happen and so on
  - what was written - target medium
  -  *who read it* - reader/follower
  - what was written about the target medium on Wikipedia. - target medium profile/bio

Argument annotation **For example:**

- Factuality is modeled on a 3-point scale: 

*low*, *mixed*, and *high*. 

- Political bias is modeled on a 7-point scale: 

*extreme-left*, *left*, *center-left*, *center*, *center-right*, *right*, and *extreme- right*. 





BERT-based models

Template:  a sentence and two entity spans (non-overalapping)

Tasks: Fake news detection (平行还是属于)

​	subGoal: predict the relation between the two entities

​		subsubgoal: Factuality Prediction	

​							   Political Bias Prediction

BERT architecture:

![img](https://ryanong.co.uk/wp-content/uploads/2020/04/arhcitecture-1.png)

**Goal:**map them to the correct labels

**Input:** 

==Bert process== is as follows:

finetuned BERT for relevance classification over text. 

1. Input sentence formatting into [CLS] sentence [SEP] predicate [SEP] so that predicate can interact with the whole sentence through attention mechanism
2. Feed the input into the BERT encoder



fine-tuning BERT for relevance classification over text. 

SCIBERT is based on the BERT architecture. Everything is the same as BERT except it is pretrained on scientific corpuses. 

















1. Add special tokens to the input sentence ([CLS] and [SEP]) and mask entity mentions with mask tokens to prevent overfitting. There are two argument type (subject and object) and two entity type (location and person). For example, [S-PER] denote subject entity is a person. See figure above for an example of input sentence with special tokens.
2. Tokenisation using WordPiece tokeniser and feed into BERT encoder to obtain the contextualised representation (==and then averaging the word representations extracted from the second-to-last layer==)
3. Remove any sequence after the first [SEP] token
4. Compute the **position** sequence relative to the subject entity and object entity
5. Convert both position sequence into position embeddings and concatenate it to the contextualised representation
6. Feed it through a one-layer BiLSTM
7. The final hidden states in both direction are feed into a multi-layer (one hidden layer) perceptron



fine-tuned BERT by training a **softmax** layer on top of the [CLS] output vector to **predict the label** (bias or factuality) of news articles that are scrapped from an external list of media to avoid overfitting.

Assumption(Label): The articles’ labels are assumed to be the same as those of the media in which they are published (a form of distant supervision). ??????????





RESULTS





#### Experimental Setup

###### ==WHAT ARE THE DOWNSTREAM NLP TASKS?==

1. Named Entity Recognition (NER)
2. PICO Extraction (PICO)
3. Text Classification (CLS)
4. Relation Extraction (REL)
5. Dependency Parsing (DEP)



*（A downstream task is a task further down the pipeline.）*

Imagine you have a **pre-trained** word embedding model trained on next-word prediction. You can use those vectors on a **down-stream task** (say, **text classification**) by using the word vectors (or rather, pooled word vectors = document vectors) as input to a final ML model (usually a linear SVM or neural network since this is usually a high dimensional space) to **do a downstream task like textual classification**.

**Sentiment analysis**, some kinds of summarization, etc are also downstream tasks. It's "downstream" if your pretrained model wasn't directly trained on that task but you're using it with in combination with a final classifier.

**[Word2Vec: Optimal Hyper-Parameters and Their Impact on NLP Downstream Tasks](https://arxiv.org/abs/2003.11645)**



==FINETUNING BERT==



<img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904215749334.png" alt="image-20200904215749334" style="zoom:50%;" />

[**Tokenization**](https://drive.google.com/file/d/1T0agNwFhYRs5Brj2Spn0gjmSyEalA5UP/view)[](https://drive.google.com/file/d/1T0agNwFhYRs5Brj2Spn0gjmSyEalA5UP/view)

• What is a term, token and type
• E.g., “a rose is a rose is a rose”, 3 types 8 tokens
 **Normalization**
• Car, car, CAR
**Stemming (or reweighting e.g., TF-IDF)**
• Removing articles (i.e., the, a, an)
**Annotation**
• play/Verb, play/Noun, Tokyo/Place, Trump/Person or Place
**Similarity** (==Distance== Function)

- Between words, sentences, paragraphs, and documents

- Similarity: Word Co-occurrence Matrix

- Word Document Matrix for “Bag” of Words (sparse)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gieyuo8p64j315u0mu42i.jpg" alt="image-20200904220237707" style="zoom:50%;" />

<img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904221033171.png" alt="image-20200904221033171" style="zoom:50%;" />



## 运行Fine-Tuning

对于大部分情况，我们不需要重新Pretraining。我们要做的只是根据具体的任务进行Fine-Tuning，因此我们首先介绍Fine-Tuning。这里我们已GLUE的MRPC为例子，我们首先需要下载预训练的模型然后解压，比如作者解压后的位置是：



函数首先调用load_vocab加载词典，建立词到id的映射关系。下面是文件uncased_L-12_H-768_A-12/vocab.txt的部分内容

接下来是构造BasicTokenizer和WordpieceTokenizer。前者是根据空格等进行普通的分词，而后者会把前者的结果再细粒度的切分为WordPiece。

tokenize函数实现分词，它先调用BasicTokenizer进行分词，接着调用WordpieceTokenizer把前者的结果再做细粒度切分。下面我们来详细阅读这两个类的代码。我们首先来看BasicTokenizer的tokenize方法。

首先是用convert_to_unicode把输入变成unicode，这个函数前面也介绍过了。接下来是_clean_text函数，它的作用是去除一些无意义的字符。



### [WordpieceTokenizer](https://fancyerii.github.io/2019/03/09/bert-codes/)

WordpieceTokenizer的作用是把词再切分成更细粒度的WordPiece。关于WordPiece(Byte Pair Encoding)我们之前在机器翻译部分已经介绍过了，它是一种解决OOV问题的方法，如果不管细节，我们把它看成比词更小的基本单位就行。

这其实是贪心的最大正向匹配算法。

text = convert_to_unicode(text)



这里的最关键是convert_single_example函数，读懂了它就真正明白BERT把输入表示成向量的过程，所以请读者仔细阅读代码和其中的注释。

\# 如果有b，那么需要保留3个特殊Token[CLS], [SEP]和[SEP]

<img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200906110630485.png" alt="image-20200906110630485" style="zoom:50%;" />

Transformer model 

*# 在这里，我们是用来做分类，因此我们只需要得到[CLS]最后一层的输出。* *# 如果需要做序列标注，那么可以使用model.get_sequence_output()* *# 默认参数下它返回的output_layer是[8, 768]* 

*# 默认是768* hidden_size **=** output_layer.shape[**-**1].value

output_weights **=** tf.get_variable( 	"output_weights", [num_labels, hidden_size], 	

initializer**=**tf.truncated_normal_initializer(stddev**=**0.02))  output_bias **=** tf.get_variable( 	"output_bias", [num_labels], initializer**=**tf.zeros_initializer()) 



*# 对[CLS]输出的768的向量再做一个线性变换，输出为label的个数。得到logits* 

通过BERT生成自己的word embeddings。

允许研究人员针对特定的任务小小的微调一下（使用少量的数据和少量的计算），就可以得到一个很好的结果

你可以使用这些模型从文本数据中提取高质量的语言特征，也可以使用你自己的数据对这些模型进行微调，以完成特定的任务(分类、实体识别、问题回答等)，从而生成最先进的预测。

###### 为什么要使用BERT的嵌入？

在本教程中，我们将使用BERT从文本数据中**提取**特征，即单词和句子的**嵌入向量**。我们可以用这些词和句子的嵌入向量做什么？

- 首先，这些嵌入对于关键字/搜索扩展、语义搜索和信息检索非常有用。例如，如果你希望将客户的问题或搜索与已经回答的问题或文档化的搜索相匹配，这些表示将帮助准确的检索匹配客户意图和上下文含义的结果，即使没有关键字或短语重叠。

- 其次，或许更重要的是，这些**向量被用作下游模型的高质量特征输入**。NLP模型(如LSTMs或CNNs)需要以数字向量的形式输入，这通常意味着需要**将词汇表和部分语音等特征转换为数字表示**。在过去，单词被表示为惟一索引值(one-hot编码)，或者更有用的是作为**神经单词嵌入**，==其中词汇与固定长度的特征嵌入进行匹配==，这些特征嵌入是由Word2Vec或Fasttext等模型产生的。与Word2Vec之类的模型相比，BERT提供了一个优势，因为尽管Word2Vec下的每个单词都有一个固定的表示，而与单词出现的上下文无关，BERT生成的单词表示是由单词周围的单词动态通知的。例如，给定两句话：







###### word2vec

它的原理大致就是通过背后的CBow和skip-gram模型进行相应的计算，然后得出词向量。

```python
def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,



                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,



                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,



                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):
```

主要关注的就是 sentences、size、window、min_count这几个参数。

​     sentences它是一个**list**，size表示输出向量的维度。

​     **window**：当前词与预测次在一个句子中最大距离是多少。

​     min_count：用于字典阶段，词频少于min_count次数的单词会被丢弃掉，默认为5

​     在训练模型之前，要对收集的预料进行分词处理，同时也要进行停用词的处理。在对数据进行预处理之后就可以训练我们的词向量了。训练的方式，从代码实现角度看，有两种，如下：

###### 文本相似度计算——文档级别



###### word2vec是如何得到词向量的？

**本答旨在阐述word2vec如何将corpus的one-hot向量（模型的输入）转换成低维词向量（模型的中间产物，更具体来说是输入权重矩阵），真真切切感受到向量的变化，不涉及加速算法。**

文本语料库->预处理（**语料库种类**以及**个人目的**有关）->processed corpus->将他们的one-hot向量作为word2vec的输入->通过word2vec训练低维词向量（word embedding）就ok了->目前有两种训练模型（CBOW和Skip-gram），两种加速算法（Negative Sample与Hierarchical Softmax）

###### **CBOW模型流程[可视化**](https://www.zhihu.com/question/44832436)

假设我们现在的Corpus是这一个简单的只有四个单词的document：
{I drink coffee everyday}
我们选coffee作为中心词，window size设为2
也就是说，我们要根据单词"I","drink"和"everyday"来预测一个单词，并且我们希望这个单词是coffee。



##### **Bag-of-Words**

##### **Distributional Word Embeddings**

- Word2Vec
  - framework for mapping word representation to a vector space using a large corpus of text.
- Word2Vec has two variants:
  - Skip-grams and
  -  CBoW
- **Contextualised Word Embeddings**
  - known as distributional (or fixed) word embeddings
  - 