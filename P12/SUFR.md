**Background**



Natural language processing(NLP) techniques play important roles in our daily life. With NLP, we can easily analyze real-time public attitudes to specific events, and their demographic determinants from social media. For example, the currently hot research is focusing on the adverse drug reaction detection in Covid-19 vaccination, which will help the public to better understand and develop the baseline confidence in vaccines.



However, social prejudice and communication behavior may harm the nlp model accuracy by affecting the way we speak and write. And these are reflected in the written words, which we used to train the machine learning system. When we use biased data to train models, it will decrease the model accuracy, especially for the certain group of people. 



**Objective**



Chemical named entity recognition(NER) is traditional method to detect  drug mentioned on social media, such as Reddit r/AskDocs. And nationwide studies have shown that male and female patients exhibit differences regarding the pharmacology and toxicity of medications and differ in their response to drug treatment( Colomboet al.(2016) ). In this work, we want to test whether there is gender bias in the Chemical NER algorithm by measuring model performance differences across genders.



**Method**



In downstream tasks, the prediction should not depend heavily on the gender of the entity mentioned in the context. To assess whether this is the case, we designed a pairs of sentences as input, where the only difference is the gender specific word. For example, one is "John said he had been taking citalopram for the treatment of illness." And another is "Claire said she had been taking citalopram for the treatment of illness.". If the model does not make decisions based on gender, then it should have the same evaluation score on both sentences. Otherwise, any difference in the score may reflect the signal of gender bias found in the system.



We introduce two metrics to measure these performance differences: Precision and Recall. Precision is associated with False Positive and Recall is associated with False Negative. For instance, In the drug detection, a false positive means that not chemical term is misclassified as drug. The patient may lose importent adverse reaction information if the precision is not high. And If a chemical term(actual positive) is misclassified as not drug (False Negative), The cost associated with False Negative will be extremely high if the adverse reaction is fatal.



**Result**



In our experiments, two corpora were used to train spaCy's NER model or Flair's NER model: the BioCreative V chemical-disease relation (CDR) task corpus (Li *et al.*, 2016) and the chemical compound and drug name recognition (CHEMDNER) corpus (Krallinger *et al.*, 2015) 



Our current analyses show that spaCy's NER model with CDR corpus achieves better performances with male's names than female on the designed test chemical templates(the average Precision for male 99.34% and 92.14% for female; Recall of 76.31% and 74.51% for male and female, respectively). And Flair's NER model with CDR corpus also reaches the better performances with male's names than female on the same test set( the average Precision for male 99.21% and 96.91% for female; Recall of 94.51% and 94.55% for male and female, respectively).



**Conclusion and Discussion**