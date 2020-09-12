8/29/2020

https://www.utsa.edu/today/2020/08/story/algorithm-bias-health-tweets.html

UTSA experts find bias in disease-tracking algorithms that analyze social media

- monitor the spread of diseases such as influenza or coronavirus. 

- ML used to train and classify tweets 
- $exit$ an inherent bias because they do not account for how minority groups potentially communicate health information.
- bias conducted on biomedical content on the microblogging and social networking service Twitter. 

![image-20200830234951673](https://tva1.sinaimg.cn/large/007S8ZIlgy1gi99uz41o5j31ca04gq3v.jpg)

- computers are used to monitor and classify millions of tweets to track how disease content spreads

- health organizations can deploy the algorithms quickly and at large geographic scales.

- **surveillance systems** are based mostly on one dialect and, in essence, don’t account for how a **minority group** might use different terms or a specific communicative style.

- Therefore, organizations can **assume** **incorrectly** that healthy behaviors or enough medical supplies exist within certain regions.

- examined both bias and fairness on influenza-related **tasks**

  - identifying influenza-related tweets, 

  - detecting whether a tweet is about an infection or simply raising awareness, 

  - detecting whether a user is discussing themselves or someone else, and 

  - identifying vaccine-related tweets.

    - Standard American English (SAE)
    - African American Vernacular English (AAE)

    *" If an unfair model is applied to geographical regions with a large number of AAE speakers, then it may not perform as the model developers expected. Because the number of speakers of SAE is larger than AAE speakers, a model can be both highly accurate and unfair,” said Rios.*

    *found that neural networks were more accurate, but simple machine learning methods produced fairer predictions*

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gi99zimh1pj31ae0pswoq.jpg" alt="image-20200830235420580" style="zoom: 50%;" />

**Bias**

- how text is classified or 
- how a system learns about words. 
- ML can generate word embeddings or vector representations for terms that is, representations of words a computer can understand along numerical values.
- **But** the learned representations may become **skewed**.
- In some cases this can lead to potential **gender bias** in which the word *man* is similar to *doctor,* while *woman* is similar to *nurse.*

**fairness** ~ bias

- the researchers explored the integrity of the influenza classifiers built using different machine learning algorithms
  - linear models and 
  - neural networks
  -  ML is fair if the predictive performance (its **accuracy**) is the same when it is applied to two different groups of data for the same task.



**privacy issues** -- limiting the information that epidemiologists need to understand the spread of the virus.

“Although there are still privacy and ethical issues in social media use for research, it is potentially a great way to observe **health trends**, since platforms are agnostic and don’t require people to download anything or check in. Using social media, we can **conduct disease surveillance tasks**, such as predicting infection rates or estimating infection risk. Moreover, social media can be used to **understand the public’s view** about potential treatments and vaccinations,” added Rios.

It’s estimated that **influenza vaccination rates** are lower by 10% among Hispanic and African American communities, resulting in approximately 2,000 preventable deaths per year. Moreover, the timetable for COVID-19 vaccine development is anywhere between six months and two years. It’s for this reason that Rios urges natural language processing data scientists to examine how health-related algorithms are built.

https://anthonyrios.net/other/racial-bias-virus.pdf

It’s for this reason that ML offers immediate benefits and new technology to help with **digital tracing or predicting potential outbreaks**.



There are current **limitations** to the UTSA analysis. Since most **NLP bias research** does not analyze **public health applications**, and curating large **biomedical data** sets is difficult, the findings are based on small samples. This is why the researchers want to bring more attention to the issue of fairness when scientists build **biomedical NLP data sets to train machines to code and classify health-related information written by different populations.**

**Brandon Lwowski**, a UTSA doctoral student and is co-lead in the study, which was funded by the National Science Foundation.



**From Arun Das to Everyone: (7:37 AM)**

https://github.com/arundasan91/adv-ai-ml-research 

git clone https://github.com/arundasan91/adv-ai-ml-research.git



###### [Day 233: Learn NLP With Me – LinkedIn’s Knowledge Graph](https://ryanong.co.uk/2020/08/20/day-233-learn-nlp-with-me-linkedins-knowledge-graph/)

rf: https://engineering.linkedin.com/blog/2016/10/building-the-linkedin-knowledge-graph

**Questions**:

1. WHAT ARE THE ENTITIES IN LINKEDIN KNOWLEDGE GRAPH?
2. HOW IS LINKEDIN’S KNOWLEDGE GRAPH CONSTRUCTED?
3. HOW IS MACHINE LEARNING BEING USED TO BUILD THE KNOWLEDGE GRAPH?
4. DESCRIBE THE CONSTRUCTION OF ENTITY TAXONOMY.
5. HOW DOES LINKEDIN CLEAN UP USER-GENERATED ORGANIC ENTITIES?
6. HOW DOES LINKEDIN GENERATES AUTO-CREATED ENTITIES?
7. DESCRIBE THE ENTITY RELATIONSHIP INFERENCE.
8. DESCRIBE DATA REPRESENTATION AND INSIGHTS DISCOVERY .

**LINKEDIN KNOWLEDGE GRAPH**->LinkedIn features->improve recommender systems, search, monetization, and consumer products.

**entities** {members, jobs, titles, skills, companies, geographical locations, schools, etc}

**relationships**

user-generated content + data extracted from the internet (noisy and duplicates)

needs to scale as new members joined, new jobs posted, change of titles, skills, etc.

ML -> 

- construct entity taxonomy 

- perform inference on entity relationship

- data representation

- insight extraction, and interactive data acquisition from users. LinkedIn’s knowledge graph id a dynamic graph, meaning that new entities and relationships can be added and existing relationships can also be changed.

 identity of an entity and its attributes

- users and features created and maintained by users
- *Auto-created entities*: skills or titles + mining user profiles and utilising external data sources and human validations,  skills, titles, locations, companies, etc. to which we can map users to

**Data**

450 million members, 190 million job listings, 9 million companies, 600+ degrees, and so on.

**Entity attributes:**

1. Relationships to other entities in a **taxonomy**
2. Characteristic **features** not in any taxonomy

For example, a company entity has **attributes** that **refer** to other entities such as members, skills, companies, etc. It also has **attributes** such as logo, revenue, URL, etc that does **not refer** to other entity in any taxonomy. The former relationships to other entities represent **edges** in the LinkedIn knowledge graph. The latter involves text extraction, data ingestion from search engine, data integration from external sources, and other crowdsourcing-based methods.

relationships(explicit (not all trustworthy),inferred)



train a binary classifier for each kind of entity relationship (belong to or not).

**We can embed the knowledge graph into a latent space.**



![img](https://ryanong.co.uk/wp-content/uploads/2020/08/fig4.png)

###### [Day 236: NLP Papers Summary – A BERT Based Sentiment Analysis And Key Entity Detection Approach For Online Financial ](https://ryanong.co.uk/2020/08/23/day-236-nlp-papers-summary-a-bert-based-sentiment-analysis-and-key-entity-detection-approach-for-online-financial-texts/)

Bert -> sentiment analysis + NER (sentence matching task)  combined based on

fine-tuned *RoBERTa* using different methods to **implement** sentiment analysis

- sentiment analysis->
- NER -> 

Methodology



# Bio

I am currently a first year PhD student in the [Stanford AI Lab](https://ai.stanford.edu/). I am graciously funded by the NSF Graduate Research Fellowship. 