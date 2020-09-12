# Tokenization 

## 什么是分词？

分词是 [自然语言理解 – NLP](https://easyai.tech/ai-definition/nlp/) 的重要步骤。

分词就是将句子、段落、文章这种长文本，分解为以**字词为单位**的数据结构，方便后续的处理分析工作。

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-08-fenci-shili.png" alt="什么是分词？" style="zoom: 33%;" />

## 为什么要分词？

**1.将复杂问题转化为数学问题**

文本都是一些「非结构化数据」-> 转化为「结构化数据」-> 结构化数据就可以转化为数学问题了，而分词就是转化的第一步。

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-08-shuxue.png" alt="为什么要分词？" style="zoom:33%;" />

**2.词是一个比较合适的粒度**

词是表达完整含义的最小单位。

字的粒度太小，无法表达完整含义，比如”鼠“可以是”老鼠“，也可以是”鼠标“。而句子的粒度太大，承载的信息量多.

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-08-lidu.png" alt="词是合适的粒度" style="zoom:33%;" />

**区别2：英文单词有多种形态**

英文单词存在丰富的变形变换。为了应对这些复杂的变换，英文NLP相比中文存在一些独特的处理步骤，我们称为词形还原（Lemmatization）和词干提取（[Stemming](https://easyai.tech/ai-definition/stemming-lemmatisation/)）。中文则不需要

词性还原：does，done，doing，did 需要通过词性还原恢复成 do。

词干提取：cities，children，teeth 这些词，需要转换为 city，child，tooth”这些基本形态



**3种典型的分词方法**

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-08-3ways.png" alt="3种典型的分词方法" style="zoom:33%;" />

 

**基于统计的分词方法**

算法是**HMM、CRF、[SVM](https://easyai.tech/ai-definition/svm/)、深度学习**等算法

 **基于深度学习**

使用双向[LSTM](https://easyai.tech/ai-definition/lstm/)+CRF实现分词器，其本质上是序列标注.

**英文分词工具**

1. [Keras](https://github.com/keras-team/keras)
2. [Spacy](https://github.com/explosion/spaCy)
3. [Gensim](https://github.com/RaRe-Technologies/gensim)
4. [NLTK](https://github.com/nltk/nltk)

自然语言处理（NLP）就是在机器语言和人类语言之间沟通的桥梁，以实现人机交流的目的。

需要利用 NLP 技术，让机器理解这些文本信息，并加以利用

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-23-2data.png" alt="结构化数据和非结构化数据" style="zoom:33%;" />

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-23-lang.png" alt="不同物种有自己的沟通方式" style="zoom: 25%;" /><img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-23-nlp-qiaoliang.png" alt="NLP就是人类和机器之间沟通的桥梁" style="zoom:25%;" />



## NLP 的2大核心任务

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-23-nlu-nlg.png" alt="NLP有2个核心任务：NLU和NLG" style="zoom:25%;" />

**NLP = [NLU](https://easyai.tech/ai-definition/nlu/) + NLG**

NLP 有2个核心的任务：

1. [自然语言理解 – NLU | NLI](https://easyai.tech/ai-definition/nlu/)
2. [自然语言生成 – NLG](https://easyai.tech/ai-definition/nlg/)

*以智能音箱为例，当用户说“几点了？”，首先需要利用 NLU 技术判断用户意图，理解用户想要什么，然后利用 NLG 技术说出“现在是6点50分”。*



**NLU自然语言理解的5个难点：**

1. 语言的多样性
2. 语言的歧义性
3. 语言的鲁棒性
4. 语言的知识依赖
5. 语言的上下文



##### NLG自然语言生成 

[NLG](https://easyai.tech/ai-definition/nlg/) 是为了跨越人类和机器之间的沟通鸿沟，将非语言格式的数据转换成人类可以理解的语言格式，如文章、报告等。

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-24-nlg.png" alt="NLG - 将非语言格式的数据转换成人类可以理解的语言格式" style="zoom:25%;" />



**自然语言生成 – NLG 有2种方式**：

1. text – to – text：文本到语言的生成
2. data – to – text ：数据到语言的生成

**NLG 的6个步骤：**

1. 内容确定 – Content Determination
2. 文本结构 – Text Structuring
3. 句子聚合 – Sentence Aggregation
4. 语法化 – Lexicalisation
5. 参考表达式生成 – Referring Expression Generation|REG
6. 语言实现 – Linguistic Realisation

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-24-3level.png" alt="NLG 的3个 Level" style="zoom: 25%;" /><img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-24-6steps.png" alt="NLG 的6个步骤" style="zoom: 25%;" />



**典型的应用：**

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-24-yingyong.png" alt="NLG的3种典型应用" style="zoom: 25%;" />

**Dreamwriter**，腾讯AI撰稿机器人

除了文本写作，Dreamwriter还是一整个“算法辅助内容生产与运营”项目，具体包括机器写作、文本加工（纠错、摘要、脉络等）、图像处理（识图、配图、修图）、视频制作等几个类目。Dreamwriter可以为内容产品形态的创新提供有力的技术支撑。

**BI 的解读和报告生成**

几乎各行各业都有自己的数据统计和分析工具。这些工具可以产生各式各样的图表，但是输出结论和观点还是需要依赖人。NLG 的一个很重要的应用就是解读这些数据，自动的输出结论和观点.

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-24-report.png" alt="NLG自动生成数据解读的报告" style="zoom: 33%;" />







<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-24-nandian.png" alt="NLP 的5个难点" style="zoom:33%;" />

1. 语言是没有规律的，或者说规律是错综复杂的。
2. 语言是可以自由组合的，可以组合复杂的语言表达。
3. 语言是一个开放集合，我们可以任意的发明创造一些新的表达方式。
4. 语言需要联系到实践知识，有一定的知识依赖。
5. 语言的使用要基于环境和上下文。



## NLP 的4个典型应用

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-23-yingyong.png" alt="NLP的4种典型应用" style="zoom: 33%;" />

**情感分析**：正面/积极的 – 负面/消极的；通过情感分析，可以**快速**了解用户的**舆情**情况。

## NLP 的 2 种途径、3 个核心步骤

NLP - ML/DL

**方式 1：传统机器学习的 NLP 流程**

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-12-ml-nlp.png" alt="传统机器学习的 NLP 流程" style="zoom:33%;" />

**方式 2：深度学习的 NLP 流程**

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-12-dl-nlp.png" alt="深度学习的 NLP 流程" style="zoom:33%;" />



**英文 NLP 语料预处理的 6 个步骤**

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-12-6steps-en-1.png" alt="**英文 NLP 语料预处理的 6 个步骤**" style="zoom:33%;" />

1. [分词 – Tokenization](https://easyai.tech/ai-definition/tokenization/)
2. [词干提取](https://easyai.tech/ai-definition/stemming-lemmatisation/) – [Stemming](https://easyai.tech/ai-definition/stemming-lemmatisation/)
3. [词形还原](https://easyai.tech/ai-definition/stemming-lemmatisation/) – Lemmatization
4. [词性标注 – Parts of Speech](https://easyai.tech/ai-definition/part-of-speech/)
5. [命名实体识别 – NER](https://easyai.tech/ai-definition/ner/)
6. 分块 – Chunking





# Text mining 

文本挖掘是指从大量文本数据中**抽取**事先未知的、可理解的、最终可用的知识的过程，同时运用这些知识更好地组织信息以便将来参考.

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-04-11-yiyi.png" alt="文本挖掘的意义就是从数据中寻找有价值的信息" style="zoom: 25%;" /><img src="http://file.elecfans.com/web1/M00/4E/DB/o4YBAFrMYTiARunYAAAQ3KbBq9s768.png" alt="数据处理,自然语言" style="zoom: 67%;" />



每到春节期间，买火车票和机票离开一线城市的人暴增——**这是数据**

再匹配这些人的身份证信息，发现这些人都是从一线城市回到自己的老家——**这是信息**

回老家跟家人团聚，一起过春节是中国的习俗——**这是知识**



**而文本挖掘的意义就是从数据中寻找有价值的信息，来发现或者解决一些实际问题**

1, **文本挖掘基本流程**

<img src="https://upload-images.jianshu.io/upload_images/3471485-a8aa08922f236f77.png?imageMogr2/auto-orient/strip|imageView2/2/w/518" alt="img"  />

2,**文本挖掘的应用**





<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-04-11-5steps.png" alt="文本挖掘的5个步骤" style="zoom:33%;" />

<img src="https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-04-11-7ways.png" alt="7种文本挖掘的方法" style="zoom:33%;" />

**关键词提取**：对长文本的内容进行分析，输出能够反映文本关键信息的关键词。

**文本摘要**：许多文本挖掘应用程序需要总结文本文档，以便对大型文档或某一主题的文档集合做出简要概述。

**聚类**：聚类是未标注文本中获取隐藏数据结构的技术，常见的有 K均值聚类和层次聚类。更多见 [无监督学习](https://easyai.tech/ai-definition/unsupervised-learning/)

**文本分类**：文本分类使用监督学习的方法，以对未知数据的分类进行预测的机器学习方法。

**文本主题模型 [LDA](https://easyai.tech/ai-definition/latent-dirichlet-allocationlda/)**：LDA（[Latent Dirichlet Allocation](https://easyai.tech/ai-definition/latent-dirichlet-allocationlda/)）是一种文档主题生成模型，也称为一个三层贝叶斯概率模型，包含词、主题和文档三层结构。

**观点抽取**：对文本（主要针对评论）进行分析，抽取出核心观点，并判断极性(正负面)，主要用于电商、美食、酒店、汽车等评论进行分析。

**情感分析**：对文本进行情感倾向判断，将文本情感分为正向、负向、中性。用于口碑分析、话题监控、舆情分析。



**NLP & text mining 区别**

自然语言处理（NLP）关注的是人类的自然语言与计算机设备之间的**交互**关系。 

文本挖掘 ,~NLP，关注的是识别文本数据中有趣并且重要的模式。

如果**原始文本是数据**，那么**文本挖掘就是信息**，**NLP就是知识**，也就是语法和语义的关系。下面的金字塔表示了这种关系：

![数据处理,自然语言处理](http://file.elecfans.com/web1/M00/4E/DB/o4YBAFrMYTiANP3LAACOOsch5kU672.jpg)

这两种任务下对数据的预处理是相同的。

努力**消除歧义**是文本预处理很重要的一个方面，我们希望保留原本的含义，同时消除噪音。

为此，我们需要了解：

关于语言的知识

关于世界的知识

结合知识来源的方法

下图所示的六个因素也**加**大了文本数据处理的**难度**，包括非标准的语言表述、断句问题、习惯用语、新兴词汇、常识以及复杂的名词等等。

![数据处理,自然语言处理](http://file.elecfans.com/web1/M00/4E/DB/o4YBAFrMYTiAe13zAADTB32ORWM157.jpg)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gia85xiv04j313m0kywg7.jpg" alt="image-20200831193518632" style="zoom: 45%;" />

在适当的时候还会进行特征选择和工程设计

语言模型：有限状态机、马尔可夫模型、词义的向量空间建模

机器学习分类器：朴素贝叶斯、逻辑回归、决策树、支持向量机、[神经网络](http://www.elecfans.com/tags/神经网络/)

序列模型：隐藏马尔可夫模型、循环神经网络（RNN）、长短期记忆神经网络（LSTMs）



**Example**

![img](https://tva1.sinaimg.cn/large/007S8ZIlly1gia886519qj30hs08btb5.jpg)

透过社会化媒体，我们可以观察现实世界：

# 以虎嗅网4W+文章的文本挖掘为例，展现数据分析的一整套流程

https://www.jiqizhixin.com/articles/2018-12-20-18?from=synced&keyword=文本挖掘