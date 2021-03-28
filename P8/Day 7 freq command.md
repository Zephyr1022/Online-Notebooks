CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ch.py > chemical_prediction_female.out 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python model_prediction.py > disease_prediction.out 2>&1

##### Monitor Script Output on the Server

https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/

``` python
tail -f model_prediction.py.log
top -M
# Monitor GPU Performance on the Server
watch "nvidia-smi"
# Check What Scripts Are Still Running on the Server
watch "ps -ef | grep python"
```



```python
sentence = Sentence("John takes Topicycline for headache",
                    use_tokenizer=SciSpacyTokenizer())
model.predict(sentence)

for entity in sentence.get_spans():
    print(entity)
```





```
data_sentences,data_chemical
```





CUDA_VISIBLE_DEVICES=1 python model_prediction_ch.py 

```
CUDA_VISIBLE_DEVICES=1 python model_prediction_ch.py 

CUDA_VISIBLE_DEVICES=1 python model_prediction_ds.py 
```







```

CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ch.py > chemical_prediction_name_2010.out 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ch.py > chemical_prediction_name_1880.out 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ch.py > chemical_prediction_name_1890.out 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ch.py > chemical_prediction_name_1900.out 2>&1
CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ch.py > chemical_prediction_name_1910.out 2>&1
CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ch.py > chemical_prediction_name_1920.out 2>&1
CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ch.py > chemical_prediction_name_1930.out 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ch.py > chemical_prediction_name_1940.out 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ch.py > chemical_prediction_name_1950.out 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ch.py > chemical_prediction_name_1960.out 2>&1
CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ch.py > chemical_prediction_name_1970.out 2>&1
CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ch.py > chemical_prediction_name_1980.out 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python model_prediction_ch.py > chemical_prediction_name_1990.out 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python model_prediction_ch.py > chemical_prediction_name_2000.out 2>&1



CUDA_VISIBLE_DEVICES=2 nohup python model_prediction_ds.py > disease_prediction_name_2010.out 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python model_prediction_ds.py > disease_prediction_name_1880.out 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python model_prediction_ds.py > disease_prediction_name_1890.out 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python model_prediction_ds.py > disease_prediction_name_1900.out 2>&1

CUDA_VISIBLE_DEVICES=3 nohup python model_prediction_ds.py > disease_prediction_name_1910.out 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python model_prediction_ds.py > disease_prediction_name_1920.out 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python model_prediction_ds.py > disease_prediction_name_1930.out 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python model_prediction_ds.py > disease_prediction_name_1940.out 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python model_prediction_ds.py > disease_prediction_name_1950.out 2>&1

CUDA_VISIBLE_DEVICES=3 nohup python model_prediction_ds.py > disease_prediction_name_1960.out 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python model_prediction_ds.py > disease_prediction_name_1970.out 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ds.py > disease_prediction_name_1980.out 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python model_prediction_ds.py > disease_prediction_name_1990.out 2>&1
CUDA_VISIBLE_DEVICES=1 nohup python model_prediction_ds.py > disease_prediction_name_2000.out 2>&1

# check the result
 ls -tr
 grep TOTAL chemical_prediction_female.out 
 grep TOTAL test2.out | head -n 500
 tail test_run.out 
 tail -n 3 chemical_prediction_female.out
```



V7J-E84