词库模型（Bag-of-words model）。

```python
corpus文集 = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]
```

文集包括8个词：`UNC`, `played`, `Duke`, `in`, `basketball`, `lost`, `the`, `game`。文件的单词构成词汇表（vocabulary）。词库模型用文集的词汇表中每个单词的**特征向量**表示每个文档。我们的文集有8个单词，那么每个文档就是由一个包含8位元素的向量构成。构成特征向量的元素数量称为维度（dimension）。用一个词典（dictionary）来表示词汇表与特征向量索引的对应关系。

`CountVectorizer`类会把文档全部转换成小写，然后将文档词块化（tokenize）

文档词块化是把句子分割成**词块**（token）或有意义的字母序列的过程。词块大多是**单词**，但是他们也可能是一些**短语**，如标点符号和词缀。

`CountVectorizer`类通过**正则表达式**用空格分割句子，然后抽取长度大于等于2的字母序列。scikit-learn实现代码如下：

```
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
```

```
[[1 1 0 1 0 1 0 1]
 [1 1 1 0 1 0 1 0]]
{'unc': 7, 'played': 5, 'game': 2, 'in': 3, 'basketball': 0, 'the': 6, 'duke': 1, 'lost': 4}
```

让我们再增加一个文档到文集里：

```
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
```

```
[[0 1 1 0 1 0 1 0 0 1]
 [0 1 1 1 0 1 0 0 1 0]
 [1 0 0 0 0 0 0 1 0 0]]
{'unc': 9, 'played': 6, 'game': 3, 'in': 4, 'ate': 0, 'basketball': 1, 'the': 8, 'sandwich': 7, 'duke': 2, 'lost': 5}
```

对比文档的特征向量，会发现前两个文档相比第三个文档更**相似**。

如果用欧氏距离（**Euclidean distance**）**计算它们的特征向量会比其与第三个文档距离更接近**。两向量的欧氏距离就是两个向量欧氏范数（**Euclidean norm**）或L2范数差的绝对值：

![img](http://latex.codecogs.com/gif.latex?d%3D%5Cleft%20%5C%7C%20x_%7B0%7D-x_%7B1%7D%20%5Cright%20%5C%7C)

向量的欧氏范数是其元素平方和的平方根：

![img](http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20x%20%5Cright%20%5C%7C%3D%5Csqrt%7Bx_%7B1%7D%5E%7B2%7D&plus;x_%7B2%7D%5E%7B2%7D&plus;%5Ccdots%20&plus;x_%7Bn%7D%5E%7B2%7D%7D)

scikit-learn里面的`euclidean_distances`函数可以计算若干向量的距离，表示两个语义最相似的文档其向量在空间中也是最接近的。

```
from sklearn.metrics.pairwise import euclidean_distances
counts = vectorizer.fit_transform(corpus).todense()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))
```

```
文档0与文档1的距离[[ 2.44948974]]
文档0与文档2的距离[[ 2.64575131]]
文档1与文档2的距离[[ 2.64575131]]
```

如果我们用新闻报道内容做文集，词汇表就可以用成千上万个单词。每篇新闻的特征向量都会有成千上万个元素，很多元素都会是**0**。体育新闻不会包含财经新闻的术语，同样文化新闻也不会包含财经新闻的术语。有许多零元素的高维特征向量成为稀疏向量（**sparse vectors**）

用高维数据可以量化机器学习任务时会有一些问题，不只是出现在自然语言处理领域。

- 第一个问题就是高维向量需要占用更大内存。NumPy提供了一些数据类型只显示稀疏向量的非零元素，可以有效处理这个问题。

- 第二个问题就是著名的维度灾难（curse of dimensionality，Hughes effect），维度越多就要求更大的训练集数据保证模型能够充分学习。如果训练样本不够，那么算法就可以拟合过度导致归纳失败。下面，我们介绍一些降维的方法。在第7章，PCA降维里面，我们还会介绍用数值方法降维。

  停用词过滤

  **特征向量降维**的一个基本方法是单词全部转换成小写。这是因为单词的大小写一般不会影响意思。而首字母大写的单词一般只是在句子的开头，而词库模型并不在乎单词的位置和语法。

  另一种方法是去掉文集常用词。这里词称为停用词（**Stop-word**），像`a`，`an`，`the`，助动词`do`，`be`，`will`，介词`on`，`around`，`beneath`等。停用词通常是构建文档意思的功能词汇，其字面意义并不体现。`CountVectorizer`类可以通过设置`stop_words`参数过滤停用词，默认是英语常用的停用词。

- https://blog.csdn.net/u013719780/article/details/51743867





