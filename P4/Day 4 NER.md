

- **Correct (COR)** : both are the same;
- **Incorrect (INC)** : the output of a system and the golden annotation don’t match;
- **Partial (PAR)** : system and the golden annotation are somewhat “similar” but not the same;
- **Missing (MIS)** : a golden annotation is not captured by a system;
- **Spurius (SPU)** : system produces a response which doesn’t exist in the golden annotation;



consider partial matching for instance



differences - between NER output and golden annotations



```
Difference b/w Vim And Cat
```



- Number of gold-standard annotations contributing to the final score

  POSSIBLE(POS)=COR+INC+PAR+MIS = ==TP+FN==

- Number of annotations produced by the NER system:

  ACTUAL(ACT)=COR+INC+PAR+SPU= ==TP+FP==



Then we can compute precision/recall/f1-score, where roughly describing 

- **precision** is the percentage of correct named-entities found by the NER system, and 
- **recall** is the percentage of the named-entities in the golden annotations that are retrieved by the NER system. This is computed in two different ways depending wether we want an **exact match** (i.e., *strict* and *exact* ) or a **partial match** (i.e., *partial* and *type*) scenario:

#### **Exact Match** (i.e., *strict* and *exact* )

$$\text{Precision} = \frac{COR}{ACT} = \frac{TP}{TP+FP}$$

$$\text{Recall} = \frac{COR}{POS} = \frac{TP}{TP+FN}$$



#### **Partial Match** (i.e., *partial* and *type*)



**Result**:

|  **Measure**  | **Type** | **Partial** | **Exact** | **Strict** |
| :-----------: | :------: | :---------: | :-------: | :--------: |
|  **Correct**  |          |             |     3     |     2      |
| **Incorrect** |          |             |     2     |     3      |
|  **Partial**  |          |             |     0     |     0      |
|  **Missed**   |          |             |     1     |     1      |
|  **Spurius**  |          |             |     1     |     1      |
| **Precision** |   0.5    |    0.66     |    0.5    |    0.33    |
|  **Recall**   |   0.5    |    0.66     |    0.5    |    0.33    |
|    **F1**     |   0.5    |    0.66     |    0.5    |    0.33    |



not assumed to come from prescribed models that are determined by a small number of parameters





##### [Evaluating Precision and Recall of NER](https://support.prodi.gy/t/evaluating-precision-and-recall-of-ner/193)

```python
def precision_recall(dataset, spacy_model, label=None, threshold=0.5):
    """
    Calculate precision and recall of NER predictions.
    """

    # I don't know what to do here.
    def evaluate(model, samples, label, threshold):
        return 10, 2, 1

    log("RECIPE: Starting recipe ner.pr", locals())
    model = EntityRecognizer(spacy.load(spacy_model), label=label)
    log('RECIPE: Initialised EntityRecognizer with model {}'.format(spacy_model), model.nlp.meta)
    samples = DB.get_dataset(dataset)
    tp, fp, fn = evaluate(model, samples, threshold)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    print("Precision {:0.4f}\tRecall {:0.4f}\tF-score {:0.4f}".format(precision, recall, f_score))
```



```python
def evaluate(model, samples, label, threshold):
    tp = fp = fn = 0
    for sample in samples:
        truth = set((span["start"], span["end"]) for span in sample["spans"] if span["label"] == label)
        hypotheses = set((entity.start_char, entity.end_char)
                         for entity in model.nlp(sample["text"]).ents if entity.label_ == label)
        tp += len(truth.intersection(hypotheses))
        fp += len(hypotheses - truth)
        fn += len(truth - hypotheses)
    return tp, fp, fn
```



```python
def gold_to_spacy(dataset, spacy_model, biluo=False):
    #### Ripped from ner.gold_to_spacy. Only change is returning annotations instead of printing or saving
    DB = connect()
    examples = DB.get_dataset(dataset)
    examples = [eg for eg in examples if eg['answer'] == 'accept']
    if biluo:
        if not spacy_model:
            prints("Exporting annotations in BILUO format requires a spaCy "
                   "model for tokenization.", exits=1, error=True)
        nlp = spacy.load(spacy_model)
    annotations = []
    for eg in examples:
        entities = [(span['start'], span['end'], span['label'])
                    for span in eg.get('spans', [])]
        if biluo:
            doc = nlp(eg['text'])
            entities = spacy.gold.biluo_tags_from_offsets(doc, entities)
            annot_entry = [eg['text'], entities]
        else:
            annot_entry = [eg['text'], {'entities': entities}]
        annotations.append(annot_entry)

    return annotations

def evaluate_prf(ner_model, examples):
    #### Source: https://stackoverflow.com/questions/44827930/evaluation-in-a-spacy-ner-model
    scorer = spacy.scorer.Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = spacy.gold.GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

@recipe("ner.stats",
        dataset=recipe_args["dataset"],
        spacy_model=recipe_args["spacy_model"],
        label=recipe_args["entity_label"],
        isPrf=("Output Precsion, Recall, F-Score", "flag", "prf"))

def model_stats(dataset, spacy_model, label=None, isPrf=False):
    """
    Evaluate model accuracy of model based on dataset with no training
    inspired from https://support.prodi.gy/t/evaluating-precision-and-recall-of-ner/193/2
    got basic model evaluation by looking at the batch-train recipe
    """
   
    log("RECIPE: Starting recipe ner.stats", locals())
    DB = connect()
    nlp = spacy.load(spacy_model)
    

    if(isPrf):
        examples = gold_to_spacy(dataset, spacy_model)
        score = evaluate_prf(nlp, examples)
        print("Precision {:0.4f}\tRecall {:0.4f}\tF-score {:0.4f}".format(score['ents_p'], score['ents_r'], score['ents_f']))

    else:
        #ripped this from ner.batch-train recipe
        model = EntityRecognizer(nlp, label=label)
        evaldoc = merge_spans(DB.get_dataset(dataset))
        evals = list(split_sentences(model.orig_nlp, evaldoc))

        scores = model.evaluate(evals)

        print("Accuracy {:0.4f}\tRight {:0.0f}\tWrong {:0.0f}\tUnknown {:0.0f}\tEntities {:0.0f}".format(scores['acc'], scores['right'],scores['wrong'],scores['unk'],scores['ents']))
```



##### A Review of Named Entity Recognition (NER) Using Automatic Summarization of [Resumes](https://towardsdatascience.com/a-review-of-named-entity-recognition-ner-using-automatic-summarization-of-resumes-5248a75de175)







```
mlxtend
```