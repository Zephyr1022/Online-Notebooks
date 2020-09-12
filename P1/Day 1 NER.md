**Temlate**

Research Project TODOs: (edited) 

1. Explore the scispacy python API. Make sure you understand and can figure out how to get everything running

- To get a feel of what is going on, you can use their online demo (https://scispacy.apps.allenai.org/). Just copy text and play with it. I recommend testing the template "<NAME> has <DISEASE>" using the "en_ner_bc5cdr_md" specialized NER model. The main results will be under the heading "Specialized NER". For example names, you can look here: https://www.ssa.gov/oact/babynames/decades/names2010s.html. Using the online demo, I tested "Abigail has diabetes" and it classified "Abigail" as a DISEASE.

- Explore the BC5CDR Dataset. I have uploaded the dataset here. You can use some of the DISEASES in the dataset to fill in your templates when playing with the scispacy API.
- Using the BC5CDR dataset and the two datasets attached below, try to come up with at least 20 possible templates with regard to DISEASE and CHEMICALS.



**Template**

<NAME> takes <CHEMICAL> due to <DISEASE>, <DISEASE>, <DISEASE>

<NAME> takes <CHEMICAL> to deal with <DISEASE>

<NAME> takes <CHEMICAL> to deal with <DISEASE>

The <CHEMICAL> has helped <NAME> cure her <DISEASE>

<NAME> takes <CHEMICAL>

<NAME> has <DISEASE>

<CHEMICAL> may be a factor in the increasing incidence of <DISEASE>, <NAME> said.

```python
names = ['Anthony', 'Xingmeng']
chemicals = ['A', 'B']
for n in names:
   for c in chemicals:
      # Fill in template
```

###### Generating Custom Word Documents From Templates Using Python

https://blog.formpl.us/how-to-generate-word-documents-from-templates-using-python-cb039ea2c890



**SpaCy** models for biomedical text processing

https://pypi.org/project/scispacy/  #scispacy 0.2.5 install

''`To activate this environment, use`

`\#   $ conda activate scispacy`''

`\# To deactivate an active environment, use`

`\#   $ conda deactivate`



1. Create a Conda environment called "scispacy" with Python 3.7:

   ```
   conda create -n scispacy python=3.7
   ```

2. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use scispaCy.

   ```
   source activate scispacy
   ```

Now you can install `scispacy` and one of the models using the steps above.

Once you have completed the above steps and downloaded one of the models below, you can load a scispaCy model as you would any other spaCy model. For example:

```
import spacy
nlp = spacy.load("en_core_sci_sm")
doc = nlp("Alterations in the hypocretin receptor 2 and preprohypocretin genes produce narcolepsy in some animals.")
```

**Available Models**

To install a model, click on the link below to download the model, and then run

```
pip install </path/to/download>
```



##### Create Virtual Environment using “conda” and add it to Jupyter Notebook

[Create Virtual Environment using “conda” and add it to Jupyter Notebook](https://medium.com/analytics-vidhya/create-virtual-environment-using-conda-and-add-it-to-jupyter-notebook-d319a81dfd1)

- `conda install ipykernel`

- source activate scispacy
- python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)"
- jupyter notebook



- conda create -n qq python=3.7 jupyter

- ```text
  conda create -n tfpy3 python=3
  source activate tfpy3
  pip install tensorflow
  jupyter notebook
  conda install nb_conda
  jupyter notebook
  pip install --user ipykernel
  ```

- conda activate qq

###### This error happens when Python can't find pandas in the list of available libraries

`which python` `which pip`

If you used Python before you may have used virtual environments and pip

The issue was not because of multiple Python versions.

（2） 方法二——一步到位的方法

在我创建完我需要的运行环境之后，然后只需要在base运行环境中执行一个命令即可。

(base) C:\Users\lenovo>conda install nb_conda

将会将所有的kernel全部添加进去，这种方法是最快的，而且最不容易出错，推荐使用。

https://www.zhihu.com/question/46309360

#### jupyter notebook 可以做哪些事情？

https://zhuanlan.zhihu.com/p/74950682

https://www.zhihu.com/question/340784347



#### Jupyter Notebook使用多个conda虚拟环境



#### [（译）27 个Jupyter Notebook的小提示与技巧](http://liuchengxu.org/pelican-blog/jupyter-notebook-tips.html)

http://liuchengxu.org/pelican-blog/jupyter-notebook-tips.html

补充知识：****将jupyter 放进你的新环境中**

在新环境下pip install jupyter之后，输入

> python -m ipykernel install --user --name=环境名即可

#### 转载：手把手教你如何开发一个NLP机器学习模型,并将它部署在Flask的Web平台上(译)

https://blog.csdn.net/qq_37486501/article/details/104574418

**Pandas**

https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html

https://jupyter.org/install

**LSTM**

https://www.cnblogs.com/jclian91/p/10886031.html

#### Using scispaCy for Named-Entity Recognition (Part 1)

https://towardsdatascience.com/using-scispacy-for-named-entity-recognition-785389e7918d

#### COVID19- Pre-Processing Pipeline

https://www.kaggle.com/skylord/covid19-pre-processing-pipeline



#### 文本自动摘要

#### 生成式摘要

#### 抽取式摘要



![img](https://ask.qcloudimg.com/http-save/yehe-1599485/e0n1ei8lbm.jpeg?imageView2/2/w/1620)



如何判断两段文本说的是「同一件事情」？- 知乎 https://www.zhihu.com/question/56751077

document level

sentence level

- **paraphrase **即给定一个问题，判断一段文本是不是符合这个问题的回答

- 问答对匹配，或者说检索式**QA**即给定一个问题，判断一段文本是不是符合这个问题的回答 

- **entailment**任务，即判断给定一段文本后能不能推理出另一段给定的文本（判断文本2是否可以根据文本1推理得到）



encoding的模型基本分为CNN系、RNN系、RecNN系以及self-attention系这几种



信息抽取Information Extraction是NLP，包括实体抽取（命名实体识别，Named Entity Recognition）、关系抽取（Relation Extraction）(NLU)和事件抽取（Event Extraction

知识图谱 **三元组（由一对实体和一个关系构成）**

关系抽取从实现的算法来看，主要分为四种：

- 手写规则（Hand-Written Patterns）
- 监督学习算法（Supervised Machine Learning）
- 半监督学习算法（Semi-Supervised Learning，比如Bootstrapping和Distant Supervision）
- 无监督算法（Unsupervised learning）

构造规则模板



用端到端的深度学习方法就没这么费劲了。

比如使用CNN或Bi-LSTM作为句子编码器，把一个句子分词后的词嵌入（Word Embedding）作为Input，用CNN或LSTM做特征的抽取器，最后经过softmax层得到N种关系的**概率**。

这样省略了特征构造这一步，自然不会在特征构造这里引入误差。



如果开发一个非通用NLP模型，专门针对某项具体任务，在降低训练成本的同时，性能会不会提高呢？

这就是谷歌发布的“天马”（**PEGASUS**）模型，它专门为机器生成摘要而生，刷新了该领域的SOTA成绩，并被ICML 2020收录。

“天马”模型仅使用1000个样本进行训练，就能接近人类摘要的水平，大大减少了对监督数据的需求，创造了低成本使用的可能性。

##### 从填空到生成摘要

PEGASUS的全称是：利用提取的间隙句进行摘要概括的**预**训练模型（Pre-training with Extracted **Gap-sentences** for Abstractive Summarization）。就是设计一种间隙句生成的自监督预训练目标，来改进生成摘要的微调性能。

自监督预训练目标越接近最终的**下游任务**，微调性能越好。

**Gap-sentences** : 在“天马”模型的预训练中，研究者从一段文档中删掉一些句子，让模型进行恢复任务。这些隔空删掉的句子即为间隙句。



![img](https://pic1.zhimg.com/v2-6578d9cfaa5a0c6113c8d57333592004_b.webp)

https://zhuanlan.zhihu.com/p/148225554

https://github.com/google-research/pegasus



**Survey**: 谷歌将模型生成的摘要和人类提取的摘要放在一起，给用户进行评估。在3个不同数据集上进行的实验表明，打分的人有时会更喜欢机器生成的摘要。

