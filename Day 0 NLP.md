### Day 1: What Is Natural Language Processing

Natural Language Processing (NLP) is a subfield within **Artificial Intelligence** that allows computers to understand human languages. It is broken down into two main areas: 

- Natural Language Understanding (NLU) and
- Natural Language Generation (NLG).



NLU allows the machine to understand and comprehend human languages and classifies them into different intents. Examples of **NLU** tasks include sentiment analysis, topic modelling, and text categorisation. Once the machine **understand** the language input, it might be required to **generate** language output. For example, in machine translation, the machine understand the source language input first and then proceed onto generating the desired language output. This falls under **NLG**.



The two main NLP techniques are **syntax** and **semantic analysis**. 

- Syntax analysis allows machine to use the order and group of words to ensure that sentences are making **grammatical** sense. 
- Semantic analysis, on the other hand, focuses the **meaning and structure** of sentences. 
  - There are two different types of semantic analysis: Lexical and Compositional. 
    - Lexical semantics focuses on the meaning of all words within a sentence whereas
    - compositional semantics focuses on understand group of words. 
    - For example: “How much Chinese silk was exported to Western Europe by the end of the 18th century?”. With **compositional** semantics, we would like the machine to understand what constitutes Western Europe or what does “end of the 18th century” actually means.

###### Challenge

NLP is a difficult problem within computer science. Two fundamental challenges exist when dealing with NLP. 

- Firstly, it is the ambiguity that exists in languages. To fully understand the intended message, machine has to clearly understand the meaning of words and how the words are connected to each other. 
- Secondly, the delivery of human languages (how someone say something) plays a major role in determining the meaning of a message. For example, **sarcasm** is difficult to track based on text input data alone
- **Fairness and bias** 