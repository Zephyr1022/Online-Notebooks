- SciBert -> not result accuracy 
- not help for the document-level task

- using citations as naturally occuring 
- redesign the triplet loss function
- Unlike many prior works, at **inference** time, our model does not require any citation information. This is critical for **embedding new papers that have not yet been cited.** 

- SCIDOCS3

- We present SPECTER, a model for learning representations of scientific papers, based on a Trans-former language model that is pretrained on citation

- ### 1 Pretrained transformer model

  - transformer model architechture as the basic encoding the input paper

  - Bert limitation:   primarily based on masked language modeling objective, only considering **intra-document context** and **do not** use any **inter-document information.**

    This limits their ability to learn optimal document

  - 

  - To learn high-quality document-level representations we propose using citations as an inter-document relatedness signal and formu-late it as a triplet loss learning objective. We then

- model architecture (Devlin et al., 2019) uses multi-ple layers of Transformers (Vaswani et al., 2017) to encode the tokens in a given input sequence. Each layer consists of a self-attention sublayer followed by a feedforward sublayer. The final hidden state associated with the special [CLS] token is usually called the “pooled output”, and is commonly used as an aggregate representation of the sequence.

*Representation learning* is *learning representations* of input data typically by transforming it or extracting features from it(by some means), that makes it easier to perform a task like classification or prediction





https://arxiv.org/pdf/2004.07180.pdf

https://distill.pub/2017/feature-visualization/

https://ai.googleblog.com/2017/11/feature-visualization.html

https://catalog.utsa.edu/graduate/business/#phd_it



https://spacy.io/universe/project/allennlp



https://allennlp.org



https://resources.wolframcloud.com/NeuralNetRepository/resources/SciBERT-Trained-on-Semantic-Scholar-Data

https://ryanong.co.uk/2020/04/24/day-115-nlp-papers-summary-scibert-a-pretrained-language-model-for-scientific-text/



https://mccormickml.com/2020/06/22/domain-specific-bert-tutorial/



https://towardsdatascience.com/how-to-apply-bert-in-scientific-domain-2d9db0480bd9

https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html



https://allenai.org/data/scidocs



https://www.datacamp.com/community/blog/spacy-cheatsheet



http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/



https://spacy.io/api/token?



https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR.md



https://github.com/arundasan91/IS7033/blob/master/CNN_invariance/Assignment_AI_Seminar_Spring_2019_DAS_ARUN.ipynb



http://cs231n.stanford.edu/slides/2020/lecture_1_feifei.pdf





http://182.92.9.171/auth-sign-in