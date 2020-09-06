### [总目录](https://ai.stanford.edu/courses/)

more information: https://ai.stanford.edu/stanford-ai-courses/ 

##### [CS224U: Natural Language Understanding](http://web.stanford.edu/class/cs224u/index.html) Slide

CS372 Artificial Intelligence for Disease [Diagnosis](http://infolab.stanford.edu/~echang/cs372/cs372-syllabus.html)

https://web.stanford.edu/class/cs224u/materials/cs224u-2020-intro-handout.pdf

http://web.stanford.edu/class/cs224u/

[==视频==](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)

https://online.stanford.edu/artificial-intelligence/free-content?category=All&course=6097

https://www.youtube.com/watch?v=LYH93YnhuyQ&feature=youtu.be

[==课件==](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/) ----CS224n

https://web.stanford.edu/class/cs224n/

https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/

==ML== http://cs229.stanford.edu/syllabus-summer2020.html

[==NER==](https://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture04-neuralnets.pdf)

[==COVID==](http://infolab.stanford.edu/~echang/cs372/cs372-syllabus.html)

[==CS372 Artificial Intelligence for Disease Diagnosis==](http://infolab.stanford.edu/~echang/cs372/cs372-syllabus.html)

[==RNN==](https://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture06-rnnlm.pdf)

Questions: 

1.  Thinking about how vectors can encode the meanings of linguistic units.
2. Foundational concepts for vector-space model (VSMs).
3. A foundation for deep learning NLU models
4. you’re likely to use representations like these:
   - 􏰁  to understand and model linguistic and social phenomena; and/or
   - 􏰁  as inputs to other machine learning models

###### Guiding hypotheses

contextual



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gienshbdzvj31q20u00y0.jpg" alt="image-20200904153957751" style="zoom: 33%;" />

Nearly the full cross-product to explore; only a handful of the combinations are ruled out mathematically. Models like GloVe and **word2vec** offer packaged solutions to design/weighting/reduction and reduce the importance of the choice of comparison method.

###### Designs -- Matrix

- word x word
- word x document
- word x discourse context
- phonological segment × feature values



###### Feature representations of data

- the movie was horrible becomes [4,0,1/4].
- The complex, real-world response of an experimental subject to a particular example becomes [0, 1] or [118, 1].
- A human is modeled as a vector [24, 140, 5, 12].
- A continuous, noisy speech stream is reduced to a restricted set of acoustic features.

###### Vector comparison

- Focus on distance measures -- Euclidean

  #### <img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904155855896.png" alt="image-20200904155855896" style="zoom:50%;" />

  

- Length normalization

- Cosine distance

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gieoe319dkj30va0li76j.jpg" alt="image-20200904160046017" style="zoom:50%;" />

###### Goals of reweighting

- 􏰙  How does it compare to the raw count values?
- 􏰙  How does it compare to the word frequencies?
- 􏰙  What overall distribution of values does it deliver?

###### Normalization

- L2 norming (repeated from earlier)

- Probability distribution
- Observed/Expected

###### Subword modeling

- to improve representations by reducing sparsity,
- thereby increasing the density of connections in a VSM.

- ==Word-piece tokenizing==

  - tokenizers that break some words into subword chunks:

    <img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904160741692.png" alt="image-20200904160741692" style="zoom:33%;" />

###### Visualization

- goal: visualize high-dim space in two or three dim

###### Dimensionality reduction

- Latent Semantic Analysis
- PCA, NMF, etc
- GloVe
- word2vec

**Euclidean with L2-normed vectors** is equivalent to cosine w.r.t. ranking (Manning and Schütze 1999:301).

**Proper distance metric?**

To qualify as a **distance metric**, a vector comparison method d has to be symmetric (d(x, y) = d(y, x)), assign 0 to identical vectors (d(x, x) = 0), and satisfy the **triangle inequality**:

###### Normalization

- L2 norming (repeated from earlier)
- Probability distribution

<img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904210449209.png" alt="image-20200904210449209" style="zoom:33%;" />

**Singular value decomposition** (The LSA method)



- **Calculus and Linear Algebra**: You should understand the following concepts from multivariable calculus and linear algebra: chain rule, gradients, matrix multiplication, matrix inverse.
- **Probability**: You should be familiar with basic probability distributions and be able to define the following concepts for both continuous and discrete random variables: Expectation, independence, probability distribution functions, and cumulative distribution functions.
- **Foundations of Machine Learning (Recommended)**: Knowledge of basic machine learning and/or deep learning is helpful, but not required.