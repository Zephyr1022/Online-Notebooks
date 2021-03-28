<img src="${image}/image-20201201212448897.png" alt="image-20201201212448897" style="zoom:50%;" />

==stop criterion==



Welcome to Ubuntu 16.04.5 LTS (GNU/Linux 4.15.0-123-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

146 packages can be updated.
0 updates are security updates.

Last login: Tue Nov 24 10:31:24 2020 from 45.41.181.243
xingmeng@aploverseer:~$ cd flair 
xingmeng@aploverseer:~/flair$ ls
download_and_prepare_corpora.py  HunFlair_demov1.py
xingmeng@aploverseer:~/flair$ python download_and_prepare_corpora.py
Traceback (most recent call last):
  File "download_and_prepare_corpora.py", line 5, in <module>
    corpus = CDR()
NameError: name 'CDR' is not defined
xingmeng@aploverseer:~/flair$ vim download_and_prepare_corpora.py
xingmeng@aploverseer:~/flair$ CUDA_VISIBLE_DEVICES=1 python download_and_prepare_corpora.py
2020-12-01 07:14:19,910 Reading data from /home/xingmeng/.flair/datasets/cdr
2020-12-01 07:14:19,910 Train: /home/xingmeng/.flair/datasets/cdr/SciSpacySentenceSplitter_core_sci_sm_0.2.5_SciSpacyTokenizer_core_sci_sm_0.2.5_train.conll
2020-12-01 07:14:19,910 Dev: /home/xingmeng/.flair/datasets/cdr/SciSpacySentenceSplitter_core_sci_sm_0.2.5_SciSpacyTokenizer_core_sci_sm_0.2.5_dev.conll
2020-12-01 07:14:19,911 Test: /home/xingmeng/.flair/datasets/cdr/SciSpacySentenceSplitter_core_sci_sm_0.2.5_SciSpacyTokenizer_core_sci_sm_0.2.5_test.conll
Corpus: 4745 train + 4753 dev + 5005 test sentences
xingmeng@aploverseer:~/flair$ CUDA_VISIBLE_DEVICES=1 python download_and_prepare_corpora.py
2020-12-01 07:16:14,410 Reading data from /home/xingmeng/.flair/datasets/cdr
2020-12-01 07:16:14,410 Train: /home/xingmeng/.flair/datasets/cdr/SciSpacySentenceSplitter_core_sci_sm_0.2.5_SciSpacyTokenizer_core_sci_sm_0.2.5_train.conll
2020-12-01 07:16:14,410 Dev: /home/xingmeng/.flair/datasets/cdr/SciSpacySentenceSplitter_core_sci_sm_0.2.5_SciSpacyTokenizer_core_sci_sm_0.2.5_dev.conll
2020-12-01 07:16:14,410 Test: /home/xingmeng/.flair/datasets/cdr/SciSpacySentenceSplitter_core_sci_sm_0.2.5_SciSpacyTokenizer_core_sci_sm_0.2.5_test.conll
Corpus: 4745 train + 4753 dev + 5005 test sentences
2020-12-01 07:16:26,078 ----------------------------------------------------------------------------------------------------
2020-12-01 07:16:26,078 Model: "SequenceTagger(
  (embeddings): StackedEmbeddings(
    (list_embedding_0): WordEmbeddings('pubmed')
    (list_embedding_1): FlairEmbeddings(
      (lm): LanguageModel(
        (drop): Dropout(p=0.1, inplace=False)
        (encoder): Embedding(275, 100)
        (rnn): LSTM(100, 2048)
        (decoder): Linear(in_features=2048, out_features=275, bias=True)
      )
    )
    (list_embedding_2): FlairEmbeddings(
      (lm): LanguageModel(
        (drop): Dropout(p=0.1, inplace=False)
        (encoder): Embedding(275, 100)
        (rnn): LSTM(100, 2048)
        (decoder): Linear(in_features=2048, out_features=275, bias=True)
      )
    )
  )
  (word_dropout): WordDropout(p=0.05)
  (locked_dropout): LockedDropout(p=0.5)
  (embedding2nn): Linear(in_features=4296, out_features=4296, bias=True)
  (rnn): LSTM(4296, 256, batch_first=True, bidirectional=True)
  (linear): Linear(in_features=512, out_features=12, bias=True)
  (beta): 1.0
  (weights): None
  (weight_tensor) None
)"
2020-12-01 07:16:26,078 ----------------------------------------------------------------------------------------------------
2020-12-01 07:16:26,078 Corpus: "Corpus: 4745 train + 4753 dev + 5005 test sentences"
2020-12-01 07:16:26,078 ----------------------------------------------------------------------------------------------------
2020-12-01 07:16:26,078 Parameters:
2020-12-01 07:16:26,078  - learning_rate: "0.1"
2020-12-01 07:16:26,079  - mini_batch_size: "32"
2020-12-01 07:16:26,079  - patience: "3"
2020-12-01 07:16:26,079  - anneal_factor: "0.5"
2020-12-01 07:16:26,079  - max_epochs: "200"
2020-12-01 07:16:26,079  - shuffle: "True"
2020-12-01 07:16:26,079  - train_with_dev: "False"
2020-12-01 07:16:26,079  - batch_growth_annealing: "False"
2020-12-01 07:16:26,079 ----------------------------------------------------------------------------------------------------
2020-12-01 07:16:26,079 Model training base path: "taggers/ncbi-disease"
2020-12-01 07:16:26,079 ----------------------------------------------------------------------------------------------------
2020-12-01 07:16:26,079 Device: cuda:0
2020-12-01 07:16:26,079 ----------------------------------------------------------------------------------------------------
2020-12-01 07:16:26,079 Embeddings storage mode: cpu
2020-12-01 07:16:26,079 ----------------------------------------------------------------------------------------------------
2020-12-01 07:16:32,464 epoch 1 - iter 14/149 - loss 31.19674444 - samples/sec: 70.18 - lr: 0.100000
2020-12-01 07:16:38,932 epoch 1 - iter 28/149 - loss 21.34490708 - samples/sec: 69.27 - lr: 0.100000
2020-12-01 07:16:45,326 epoch 1 - iter 42/149 - loss 17.27329216 - samples/sec: 70.07 - lr: 0.100000
2020-12-01 07:16:52,203 epoch 1 - iter 56/149 - loss 14.76325913 - samples/sec: 65.15 - lr: 0.100000
2020-12-01 07:16:59,292 epoch 1 - iter 70/149 - loss 13.04379786 - samples/sec: 63.21 - lr: 0.100000
2020-12-01 07:17:06,458 epoch 1 - iter 84/149 - loss 11.69224096 - samples/sec: 62.52 - lr: 0.100000
2020-12-01 07:17:13,433 epoch 1 - iter 98/149 - loss 10.65177334 - samples/sec: 64.23 - lr: 0.100000
2020-12-01 07:17:19,896 epoch 1 - iter 112/149 - loss 9.92913723 - samples/sec: 69.33 - lr: 0.100000
2020-12-01 07:17:27,048 epoch 1 - iter 126/149 - loss 9.31609297 - samples/sec: 62.65 - lr: 0.100000
2020-12-01 07:17:33,372 epoch 1 - iter 140/149 - loss 8.73351908 - samples/sec: 70.85 - lr: 0.100000
2020-12-01 07:17:37,030 ----------------------------------------------------------------------------------------------------
2020-12-01 07:17:37,030 EPOCH 1 done: loss 8.4031 - lr 0.1000000
2020-12-01 07:18:34,294 DEV : loss 2.5814309120178223 - score 0.7958
2020-12-01 07:18:34,713 BAD EPOCHS (no improvement): 0
saving best model
2020-12-01 07:18:41,486 ----------------------------------------------------------------------------------------------------
2020-12-01 07:18:44,384 epoch 2 - iter 14/149 - loss 3.25525747 - samples/sec: 154.61 - lr: 0.100000
2020-12-01 07:18:47,358 epoch 2 - iter 28/149 - loss 3.25952620 - samples/sec: 150.68 - lr: 0.100000
2020-12-01 07:18:50,416 epoch 2 - iter 42/149 - loss 3.30782017 - samples/sec: 146.53 - lr: 0.100000
2020-12-01 07:18:53,455 epoch 2 - iter 56/149 - loss 3.30667959 - samples/sec: 147.46 - lr: 0.100000
2020-12-01 07:18:56,134 epoch 2 - iter 70/149 - loss 3.26977459 - samples/sec: 167.26 - lr: 0.100000
2020-12-01 07:18:58,901 epoch 2 - iter 84/149 - loss 3.16961874 - samples/sec: 161.96 - lr: 0.100000
2020-12-01 07:19:01,809 epoch 2 - iter 98/149 - loss 3.15526357 - samples/sec: 154.13 - lr: 0.100000
2020-12-01 07:19:04,630 epoch 2 - iter 112/149 - loss 3.11786826 - samples/sec: 158.85 - lr: 0.100000
2020-12-01 07:19:07,678 epoch 2 - iter 126/149 - loss 3.10091031 - samples/sec: 146.99 - lr: 0.100000
2020-12-01 07:19:10,625 epoch 2 - iter 140/149 - loss 3.06586456 - samples/sec: 152.05 - lr: 0.100000
2020-12-01 07:19:12,451 ----------------------------------------------------------------------------------------------------
2020-12-01 07:19:12,451 EPOCH 2 done: loss 3.0525 - lr 0.1000000
2020-12-01 07:19:28,035 DEV : loss 1.8826453685760498 - score 0.8509
2020-12-01 07:19:28,459 BAD EPOCHS (no improvement): 0
saving best model
2020-12-01 07:19:37,071 ----------------------------------------------------------------------------------------------------
2020-12-01 07:19:40,106 epoch 3 - iter 14/149 - loss 2.55852801 - samples/sec: 147.68 - lr: 0.100000
2020-12-01 07:19:42,939 epoch 3 - iter 28/149 - loss 2.65964436 - samples/sec: 158.15 - lr: 0.100000
2020-12-01 07:19:45,882 epoch 3 - iter 42/149 - loss 2.59886905 - samples/sec: 152.24 - lr: 0.100000
2020-12-01 07:19:48,742 epoch 3 - iter 56/149 - loss 2.53080704 - samples/sec: 156.71 - lr: 0.100000
2020-12-01 07:19:51,805 epoch 3 - iter 70/149 - loss 2.49725584 - samples/sec: 146.29 - lr: 0.100000
2020-12-01 07:19:54,610 epoch 3 - iter 84/149 - loss 2.42492016 - samples/sec: 159.73 - lr: 0.100000
2020-12-01 07:19:57,762 epoch 3 - iter 98/149 - loss 2.45915039 - samples/sec: 142.17 - lr: 0.100000
2020-12-01 07:20:00,680 epoch 3 - iter 112/149 - loss 2.45250935 - samples/sec: 153.55 - lr: 0.100000
2020-12-01 07:20:03,572 epoch 3 - iter 126/149 - loss 2.45956028 - samples/sec: 154.98 - lr: 0.100000
2020-12-01 07:20:06,457 epoch 3 - iter 140/149 - loss 2.45656304 - samples/sec: 155.29 - lr: 0.100000
2020-12-01 07:20:08,247 ----------------------------------------------------------------------------------------------------
2020-12-01 07:20:08,247 EPOCH 3 done: loss 2.4568 - lr 0.1000000
2020-12-01 07:20:24,702 DEV : loss 1.744465708732605 - score 0.8488
2020-12-01 07:20:25,124 BAD EPOCHS (no improvement): 1
2020-12-01 07:20:25,124 ----------------------------------------------------------------------------------------------------
2020-12-01 07:20:28,204 epoch 4 - iter 14/149 - loss 1.99741387 - samples/sec: 145.49 - lr: 0.100000
2020-12-01 07:20:31,362 epoch 4 - iter 28/149 - loss 2.13729186 - samples/sec: 141.92 - lr: 0.100000
2020-12-01 07:20:34,303 epoch 4 - iter 42/149 - loss 2.20759659 - samples/sec: 152.34 - lr: 0.100000
2020-12-01 07:20:37,190 epoch 4 - iter 56/149 - loss 2.19283241 - samples/sec: 155.22 - lr: 0.100000
2020-12-01 07:20:40,000 epoch 4 - iter 70/149 - loss 2.21170442 - samples/sec: 159.46 - lr: 0.100000
2020-12-01 07:20:42,916 epoch 4 - iter 84/149 - loss 2.18750146 - samples/sec: 153.68 - lr: 0.100000
2020-12-01 07:20:45,645 epoch 4 - iter 98/149 - loss 2.11857976 - samples/sec: 164.21 - lr: 0.100000
2020-12-01 07:20:48,452 epoch 4 - iter 112/149 - loss 2.10365622 - samples/sec: 159.65 - lr: 0.100000
2020-12-01 07:20:51,383 epoch 4 - iter 126/149 - loss 2.09151892 - samples/sec: 152.90 - lr: 0.100000
2020-12-01 07:20:54,211 epoch 4 - iter 140/149 - loss 2.06621960 - samples/sec: 158.41 - lr: 0.100000
2020-12-01 07:20:56,229 ----------------------------------------------------------------------------------------------------
2020-12-01 07:20:56,230 EPOCH 4 done: loss 2.0799 - lr 0.1000000
2020-12-01 07:21:11,886 DEV : loss 1.4393560886383057 - score 0.8688
2020-12-01 07:21:12,306 BAD EPOCHS (no improvement): 0
saving best model
2020-12-01 07:21:20,899 ----------------------------------------------------------------------------------------------------
2020-12-01 07:21:24,025 epoch 5 - iter 14/149 - loss 1.96413973 - samples/sec: 143.36 - lr: 0.100000
2020-12-01 07:21:27,020 epoch 5 - iter 28/149 - loss 1.80910298 - samples/sec: 149.63 - lr: 0.100000
2020-12-01 07:21:30,146 epoch 5 - iter 42/149 - loss 1.76628902 - samples/sec: 143.35 - lr: 0.100000
2020-12-01 07:21:33,158 epoch 5 - iter 56/149 - loss 1.83735956 - samples/sec: 148.77 - lr: 0.100000
2020-12-01 07:21:36,223 epoch 5 - iter 70/149 - loss 1.83562067 - samples/sec: 146.19 - lr: 0.100000
2020-12-01 07:21:39,352 epoch 5 - iter 84/149 - loss 1.84005740 - samples/sec: 143.23 - lr: 0.100000
2020-12-01 07:21:42,337 epoch 5 - iter 98/149 - loss 1.82471827 - samples/sec: 150.12 - lr: 0.100000
2020-12-01 07:21:45,159 epoch 5 - iter 112/149 - loss 1.84148912 - samples/sec: 158.78 - lr: 0.100000
2020-12-01 07:21:47,976 epoch 5 - iter 126/149 - loss 1.83411453 - samples/sec: 159.10 - lr: 0.100000
2020-12-01 07:21:50,803 epoch 5 - iter 140/149 - loss 1.83412797 - samples/sec: 158.48 - lr: 0.100000
2020-12-01 07:21:52,514 ----------------------------------------------------------------------------------------------------
2020-12-01 07:21:52,514 EPOCH 5 done: loss 1.8251 - lr 0.1000000
2020-12-01 07:22:08,247 DEV : loss 1.530491590499878 - score 0.8579
2020-12-01 07:22:08,668 BAD EPOCHS (no improvement): 1
2020-12-01 07:22:08,668 ----------------------------------------------------------------------------------------------------
2020-12-01 07:22:11,840 epoch 6 - iter 14/149 - loss 1.61405892 - samples/sec: 141.27 - lr: 0.100000
2020-12-01 07:22:14,803 epoch 6 - iter 28/149 - loss 1.63295613 - samples/sec: 151.23 - lr: 0.100000
2020-12-01 07:22:17,829 epoch 6 - iter 42/149 - loss 1.63899152 - samples/sec: 148.07 - lr: 0.100000
2020-12-01 07:22:20,715 epoch 6 - iter 56/149 - loss 1.65755001 - samples/sec: 155.29 - lr: 0.100000
2020-12-01 07:22:23,982 epoch 6 - iter 70/149 - loss 1.75768169 - samples/sec: 137.14 - lr: 0.100000
2020-12-01 07:22:26,809 epoch 6 - iter 84/149 - loss 1.70963450 - samples/sec: 158.51 - lr: 0.100000
2020-12-01 07:22:29,571 epoch 6 - iter 98/149 - loss 1.67828458 - samples/sec: 162.25 - lr: 0.100000
2020-12-01 07:22:32,430 epoch 6 - iter 112/149 - loss 1.69388056 - samples/sec: 156.76 - lr: 0.100000
2020-12-01 07:22:35,231 epoch 6 - iter 126/149 - loss 1.67740263 - samples/sec: 159.94 - lr: 0.100000
2020-12-01 07:22:38,214 epoch 6 - iter 140/149 - loss 1.65902693 - samples/sec: 150.25 - lr: 0.100000
2020-12-01 07:22:40,019 ----------------------------------------------------------------------------------------------------
2020-12-01 07:22:40,019 EPOCH 6 done: loss 1.6530 - lr 0.1000000
2020-12-01 07:22:56,512 DEV : loss 1.64898681640625 - score 0.8497
2020-12-01 07:22:56,936 BAD EPOCHS (no improvement): 2
2020-12-01 07:22:56,936 ----------------------------------------------------------------------------------------------------
2020-12-01 07:22:59,894 epoch 7 - iter 14/149 - loss 1.45300286 - samples/sec: 151.48 - lr: 0.100000
2020-12-01 07:23:02,799 epoch 7 - iter 28/149 - loss 1.36972387 - samples/sec: 154.29 - lr: 0.100000
2020-12-01 07:23:05,595 epoch 7 - iter 42/149 - loss 1.42523417 - samples/sec: 160.24 - lr: 0.100000
2020-12-01 07:23:08,914 epoch 7 - iter 56/149 - loss 1.53550267 - samples/sec: 135.00 - lr: 0.100000
2020-12-01 07:23:11,881 epoch 7 - iter 70/149 - loss 1.53974152 - samples/sec: 151.06 - lr: 0.100000
2020-12-01 07:23:14,593 epoch 7 - iter 84/149 - loss 1.54465335 - samples/sec: 165.22 - lr: 0.100000
2020-12-01 07:23:17,386 epoch 7 - iter 98/149 - loss 1.52570386 - samples/sec: 160.44 - lr: 0.100000
2020-12-01 07:23:20,286 epoch 7 - iter 112/149 - loss 1.51027462 - samples/sec: 154.52 - lr: 0.100000
2020-12-01 07:23:23,139 epoch 7 - iter 126/149 - loss 1.49908572 - samples/sec: 157.05 - lr: 0.100000
2020-12-01 07:23:26,198 epoch 7 - iter 140/149 - loss 1.51795653 - samples/sec: 146.49 - lr: 0.100000
2020-12-01 07:23:28,032 ----------------------------------------------------------------------------------------------------
2020-12-01 07:23:28,032 EPOCH 7 done: loss 1.5145 - lr 0.1000000
2020-12-01 07:23:43,796 DEV : loss 1.3237998485565186 - score 0.881
2020-12-01 07:23:44,222 BAD EPOCHS (no improvement): 0
saving best model
2020-12-01 07:23:52,814 ----------------------------------------------------------------------------------------------------
2020-12-01 07:23:55,837 epoch 8 - iter 14/149 - loss 1.33447243 - samples/sec: 148.29 - lr: 0.100000
2020-12-01 07:23:58,842 epoch 8 - iter 28/149 - loss 1.26229716 - samples/sec: 149.13 - lr: 0.100000
2020-12-01 07:24:01,815 epoch 8 - iter 42/149 - loss 1.32576592 - samples/sec: 150.69 - lr: 0.100000
2020-12-01 07:24:04,634 epoch 8 - iter 56/149 - loss 1.32107796 - samples/sec: 158.96 - lr: 0.100000
2020-12-01 07:24:07,549 epoch 8 - iter 70/149 - loss 1.36359169 - samples/sec: 153.71 - lr: 0.100000
2020-12-01 07:24:10,523 epoch 8 - iter 84/149 - loss 1.38009391 - samples/sec: 150.70 - lr: 0.100000
2020-12-01 07:24:13,645 epoch 8 - iter 98/149 - loss 1.39904638 - samples/sec: 143.51 - lr: 0.100000
2020-12-01 07:24:16,411 epoch 8 - iter 112/149 - loss 1.40988286 - samples/sec: 162.04 - lr: 0.100000

2020-12-01 07:24:19,238 epoch 8 - iter 126/149 - loss 1.40542269 - samples/sec: 158.48 - lr: 0.100000
2020-12-01 07:24:22,358 epoch 8 - iter 140/149 - loss 1.40197897 - samples/sec: 143.64 - lr: 0.100000
2020-12-01 07:24:24,495 ----------------------------------------------------------------------------------------------------
2020-12-01 07:24:24,495 EPOCH 8 done: loss 1.4100 - lr 0.1000000
2020-12-01 07:24:40,187 DEV : loss 1.3939486742019653 - score 0.8658
2020-12-01 07:24:40,606 BAD EPOCHS (no improvement): 1
2020-12-01 07:24:40,606 ----------------------------------------------------------------------------------------------------
2020-12-01 07:24:43,396 epoch 9 - iter 14/149 - loss 1.48901906 - samples/sec: 160.61 - lr: 0.100000
2020-12-01 07:24:46,348 epoch 9 - iter 28/149 - loss 1.38736360 - samples/sec: 151.84 - lr: 0.100000
2020-12-01 07:24:49,091 epoch 9 - iter 42/149 - loss 1.33424415 - samples/sec: 163.34 - lr: 0.100000
2020-12-01 07:24:52,020 epoch 9 - iter 56/149 - loss 1.33590942 - samples/sec: 152.99 - lr: 0.100000
2020-12-01 07:24:55,024 epoch 9 - iter 70/149 - loss 1.34131415 - samples/sec: 149.19 - lr: 0.100000
2020-12-01 07:24:57,962 epoch 9 - iter 84/149 - loss 1.36987290 - samples/sec: 152.51 - lr: 0.100000
2020-12-01 07:25:00,987 epoch 9 - iter 98/149 - loss 1.36815824 - samples/sec: 148.13 - lr: 0.100000
2020-12-01 07:25:03,905 epoch 9 - iter 112/149 - loss 1.36970849 - samples/sec: 153.58 - lr: 0.100000
2020-12-01 07:25:06,815 epoch 9 - iter 126/149 - loss 1.34696938 - samples/sec: 153.98 - lr: 0.100000
2020-12-01 07:25:09,687 epoch 9 - iter 140/149 - loss 1.33895643 - samples/sec: 156.02 - lr: 0.100000
2020-12-01 07:25:11,414 ----------------------------------------------------------------------------------------------------
2020-12-01 07:25:11,414 EPOCH 9 done: loss 1.3234 - lr 0.1000000
2020-12-01 07:25:27,921 DEV : loss 1.383641004562378 - score 0.8686
2020-12-01 07:25:28,340 BAD EPOCHS (no improvement): 2
2020-12-01 07:25:28,340 ----------------------------------------------------------------------------------------------------
2020-12-01 07:25:31,269 epoch 10 - iter 14/149 - loss 1.17072383 - samples/sec: 153.00 - lr: 0.100000
2020-12-01 07:25:34,251 epoch 10 - iter 28/149 - loss 1.21370035 - samples/sec: 150.28 - lr: 0.100000
2020-12-01 07:25:37,206 epoch 10 - iter 42/149 - loss 1.23729834 - samples/sec: 151.64 - lr: 0.100000
2020-12-01 07:25:40,086 epoch 10 - iter 56/149 - loss 1.20092370 - samples/sec: 155.57 - lr: 0.100000
2020-12-01 07:25:43,268 epoch 10 - iter 70/149 - loss 1.21409126 - samples/sec: 140.85 - lr: 0.100000
2020-12-01 07:25:46,248 epoch 10 - iter 84/149 - loss 1.19188103 - samples/sec: 150.36 - lr: 0.100000
2020-12-01 07:25:49,008 epoch 10 - iter 98/149 - loss 1.21432530 - samples/sec: 162.39 - lr: 0.100000
2020-12-01 07:25:51,789 epoch 10 - iter 112/149 - loss 1.22385159 - samples/sec: 161.09 - lr: 0.100000
2020-12-01 07:25:54,789 epoch 10 - iter 126/149 - loss 1.24674375 - samples/sec: 149.39 - lr: 0.100000
2020-12-01 07:25:57,623 epoch 10 - iter 140/149 - loss 1.23477527 - samples/sec: 158.13 - lr: 0.100000
2020-12-01 07:25:59,249 ----------------------------------------------------------------------------------------------------
2020-12-01 07:25:59,249 EPOCH 10 done: loss 1.2426 - lr 0.1000000
2020-12-01 07:26:14,974 DEV : loss 1.405371069908142 - score 0.8639
2020-12-01 07:26:15,398 BAD EPOCHS (no improvement): 3
2020-12-01 07:26:15,398 ----------------------------------------------------------------------------------------------------
2020-12-01 07:26:18,549 epoch 11 - iter 14/149 - loss 1.12320497 - samples/sec: 142.20 - lr: 0.100000
2020-12-01 07:26:21,513 epoch 11 - iter 28/149 - loss 1.14371094 - samples/sec: 151.19 - lr: 0.100000
2020-12-01 07:26:24,380 epoch 11 - iter 42/149 - loss 1.14224796 - samples/sec: 156.31 - lr: 0.100000
2020-12-01 07:26:27,421 epoch 11 - iter 56/149 - loss 1.17613801 - samples/sec: 147.33 - lr: 0.100000
2020-12-01 07:26:30,396 epoch 11 - iter 70/149 - loss 1.14408920 - samples/sec: 150.66 - lr: 0.100000
2020-12-01 07:26:33,265 epoch 11 - iter 84/149 - loss 1.16633892 - samples/sec: 156.15 - lr: 0.100000
2020-12-01 07:26:36,126 epoch 11 - iter 98/149 - loss 1.17714418 - samples/sec: 156.66 - lr: 0.100000
2020-12-01 07:26:38,957 epoch 11 - iter 112/149 - loss 1.16785723 - samples/sec: 158.29 - lr: 0.100000
2020-12-01 07:26:42,124 epoch 11 - iter 126/149 - loss 1.17290009 - samples/sec: 141.48 - lr: 0.100000
2020-12-01 07:26:44,821 epoch 11 - iter 140/149 - loss 1.18384075 - samples/sec: 166.13 - lr: 0.100000
2020-12-01 07:26:46,497 ----------------------------------------------------------------------------------------------------
2020-12-01 07:26:46,497 EPOCH 11 done: loss 1.1780 - lr 0.1000000

<img src="${image}/image-20201202002123215.png" alt="image-20201202002123215" style="zoom:30%;" /><img src="${image}/image-20201202002339028.png" alt="image-20201202002339028" style="zoom:30%;" />

https://www.thoughtco.com/difference-between-type-i-and-type-ii-errors-3126414

- Type I errors happen when we reject a true [null hypothesis](https://www.thoughtco.com/null-hypothesis-vs-alternative-hypothesis-3126413)

- Type II errors happen when we fail to reject a false null hypothesis

  

  ### Power analysis — significant Power ???

- $\alpha$ significant criterian 

- Power!!!

- effect size 

<img src="${image}/image-20201202003913384.png" alt="image-20201202003913384" style="zoom:50%;" />

**design study** experiement design 

Conduct the study 

behavior study 

power level -> *sample size* 

initial sample size — literature review

t-test 2 group 

chisquare - 3 more group 

<img src="${image}/image-20201202005431319.png" alt="image-20201202005431319" style="zoom: 33%;" /><img src="${image}/image-20201202005451146.png" alt="image-20201202005451146" style="zoom:33%;" />



<img src="${image}/image-20201202010340335.png" alt="image-20201202010340335" style="zoom:30%;" />

95% CI range  variables fall in the range 

- big range -> 
- narrow - > 