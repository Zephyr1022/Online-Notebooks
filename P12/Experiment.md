Text Text Text Text Text Text  Text Text Text Text Text Text Text
Text Text  Text Text Text Text Text Text. Figure~\ref{fig:01}
shows that the above method  Text Text Text Text  Text Text Text
Text Text Text  Text Text.  \citep{Bag01} wants to know about
{\ldots}{\ldots} text follows.
\begin{equation}
\sum \text{\it x}+ \text{\it y} =\text{\it Z}\label{eq:01}\vspace*{-10pt}
\end{equation}

%\enlargethispage{12pt}





We designed three experiments using these two key components of the spaCy NLP pipeline and trained multiple NER models using the annotated training data to obtain optimal performance on test data using the spaCy training module [15]. Our experimental set up included working with spaCy version 2.1.4 [18] on an Anaconda Distribution [16], Python 3.6.8 [17] environment running on a machine with x86_64 GNU/Linux, Intel Core Processor (Broadwell) with 16 GB RAM. The experiments can be split into three main methods.



**3.3.1. Method 1: Blank spaCy model**



We trained a blank spaCy English language model (this model has no trained entities) using annotated training data to recognize four custom entities. We did not provide any custom token to vector layer and set the API to use default execution of the spaCy NLP pipeline.



We started with utilizing only 50% of the available training data and trained 5 models (for 100 iterations with dropout rate=0.2) while increasing the training data in increments of 10%. The performance of the trained models was evaluated on the test data.