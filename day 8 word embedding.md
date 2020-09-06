[word embedding](https://www.tensorflow.org/tutorials/text/word_embeddings) 

[reference slides](https://drive.google.com/file/d/1T0agNwFhYRs5Brj2Spn0gjmSyEalA5UP/view)

<img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904233542707.png" alt="image-20200904233542707" style="zoom:50%;" />

- encoding
- A vector that encodes information about its co-occurrence with other words
  in the vocabulary
- predict embedding 
- Word2Vec

<img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904235515967.png" alt="image-20200904235515967" style="zoom:33%;" />



- LTSM

<img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200904235845506.png" alt="image-20200904235845506" style="zoom:50%;" />

- Transformer Model Architecture

- <img src="/Users/hxwh/Library/Application Support/typora-user-images/image-20200905000352459.png" alt="image-20200905000352459" style="zoom:50%;" />

  



[Google Colba run example](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word_embeddings.ipynb#scrollTo=Q6mJg1g3apaz)<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1giflxn7a2aj30fo0ggq46.jpg" alt="image-20200905112128833" style="zoom:33%;" />

**diff b/w Fune-tuned bert model:** 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1giflyjwfkpj31rk0kkwh7.jpg" alt="image-20200905112218273" style="zoom:50%;" />

- Add special tokens to the input sentence ([CLS] and [SEP]) 