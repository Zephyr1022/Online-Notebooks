### What is Named Entity Recognition (NER)?



Named entity recognition (NER) is a sub-task of information extraction (IE) that seeks out and categorises specified [entities](https://en.wikipedia.org/wiki/Named_entity) in a body or bodies of texts. NER is also simply known as entity identification, entity chunking and entity extraction. NER is used in many fields in Artificial Intelligence ([AI](https://en.wikipedia.org/wiki/Artificial_intelligence)) including Natural Language Processing ([NLP](https://en.wikipedia.org/wiki/Natural_language_processing)) and [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning).



### spaCy for NER

SpaCy is an open-source library for advanced Natural Language Processing in Python. It is designed specifically for production use and helps build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems, or to pre-process text for deep learning. Some of the features provided by spaCy are- Tokenization, Parts-of-Speech (PoS) Tagging, Text Classification and Named Entity Recognition.

SpaCy provides an exceptionally efficient statistical system for NER in python, which can assign labels to groups of tokens which are contiguous. It provides a default model which can recognize a wide range of named or numerical entities, which include *person, organization, language, event etc.* Apart from these default entities, spaCy also gives us the liberty to add arbitrary classes to the NER model, by training the model to update it with newer trained examples.

spaCy has implemented a deep learning implementation for obtaining dynamic word embeddings using an approximate language-modelling objective. 