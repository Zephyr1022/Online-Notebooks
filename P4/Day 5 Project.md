ToDO

- [x] 1. Test with single word disease 

- [ ] 2. Less strict evaluation (do we get some words correct in multi-word disease)
- [ ] 3. Test names for different decades
- [ ] 4. Train new models (LSTM, Transformer, ...)
- [ ] 5. TODO: Find another chemical NER dataset
- [ ] 6. Modifying and developing better templates



```python
import random
random.seed(42)
chemicals = random.sample(chemicals, 10)
```

```python
if ' ' in disease:
    continue
if len(disease.split()) > 1:
  
diseases = [d for d in diseases if len( d.split() ) == 1]
```

```
nohup python disea.py > m_dis_resultsd_single.out 2>&1 &
```



