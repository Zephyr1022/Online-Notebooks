### Named-Entity [evaluation](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/#:~:text=When%20you%20train%20a%20NER,score%20at%20a%20token%20level.&text=But%20when%20using%20the%20predicted,a%20full%20named%2Dentity%20level) metrics based on entity-level



- typicallym **precision**, **recall** and **f1-score** at a token level.
- metrics
- downstream
- with metrics at a full named-entity level.



- golden standard
- evaluation metrics:
  -  [CoNLL-2003](https://www.aclweb.org/anthology/W03-0419.pdf)
  - language-independent named entity recognition
  - XML format.
  - CoNNL style format



Assumption:  

- considering only this 3 scenarios, and discarding every other possible scenario 
- But of course we are discarding **partial matches**, or other scenarios when the NER system gets the **named-entity surface string correct but the type wrong**, and we might also want to evaluate these scenarios again at a full-entity level.

- we have a simple classification evaluation that can be measured in terms of **false negatives**, **true positives,** **false negatives and false positives**, 
- and subsequently compute **precision**, **recall** and **f1-score** for each named-entity type.



StanfordNER - training a new model and deploying a web service

- ###### Training your own model

- ###### Features: experiments and results

- ##### Distributional Similarity

#### Setting up a web service

Once a model has been trained you can apply it to text just as shown in the beginning of this post, but a most common use case is to have a web service or a HTTP endpoint, where you submit a sentence or articles, and get back the text with the named-entities identified.