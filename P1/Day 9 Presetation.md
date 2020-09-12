





# <center>What Was Written vs. Who Read It: News Media Profiling Using Text Analysis and Social Media Context System Architecture </center>







### <center>by Ramy Baly, Georgi Karadzhov</center>

### <center>Present by Xingmeng Zhao</center>

<div style="page-break-after: always; break-after: page;"></div>

# Goals



### Detect Fake News

- #### predicting the political ideology 

  - #### left bias

  - #### center

  - #### right bias

- #### predicting factuality of reporting of news media.

  - #### high

  - #### mixed

  - #### low



<!-- fake news” spreads six times faster and reaches much farther than real news. every minutes there is lots of fake news created, it impossible to fact-check every single ones. Alternatively they can look at the news media who most likely to publish fake or biased content. -->

<!-- we assume media's publishion behavior are consistency: The idea is that news media that have published fake or biased content in the past are more likely to do so in the future.-->

<!-- So simply checking the reliability of its source -> we change the problem to prediect political bias and factuality  -->



<div style="page-break-after: always; break-after: page;"></div>

# How do they predict these bias?

<!--how do they predict the bias,For each target medium:  they start from three aspects -> they present our system in the slides. For each target medium, it extracts a variety of features to model-->

<!--(*i*) what was written by the medium-->

<!--(*ii*) the audience of the medium on social media-->

<!--(*iii*) what was written about the medium in Wikipedia. -->

- ## what was written 

  - Articles

    - *Linguistic Features:* <!--extracted such features using the News Landscape (NELA) toolkit. they will refer to them as the NELA features. they averaged the NELA features for the individual articles in order to obtain a NELA representation for a news medium. -->

      <!--Using arithmetic averaging is a good idea as it captures the general trend of articles in a medium-->

      - News Landscape (NELA) toolkit
      - Average NELA features

    - *Embedding Features*

      - BERT (Devlin et al., 2019), first 510 WordPieces <!--encoding each article bert by feeding-->
      - averaging the word representations and fine-tuning bert <!--extracted from the second-to-last layer.-->

      <!--get relevant to our tasks -> by training a softmax layer on top of the special token [CLS], output vector to predict the label (bias or factuality) -->

    - *Aggregated Probabilities:* fine-tuning bert, softmax <!--by training a softmax layer on top of the special token [CLS] from the last hidden layer, didn't use the average of second-to-last layer-->

  - YouTube Video Channels

    - OpenSMILE toolkit

  - Media Profiles in Twitter

    - Sentence BERT<!--too small to fine-tune BERT-->

<!--YouTube Video Channels: OpenSMILE toolkit, to modeling their textual and acoustic contents to predict the political bias and the factuality of reporting of the target news medium.-->

<!--Media Profiles in Twitter: Sentence BERT to encode the profile’s description-->

<!--Why change model? -->



- ## Who Read it	

  - Twitter Followers Bio (Sentence BERT)

  - Facebook Audience (Facebook Marketing API)

- ## What Was Written About the Target Medium on Wikipedia

  - Pre-trained Bert and fine-tuning bert

<!-- Twitter Followers Bio: analyze the self-description (bio) of Twitter users that follow the target news medium. 2 assumption:followers would likely agree with the news medium’s bias; they might express their own bias in their self-description. extract features from 5k followers for each target news medium, encoded  feature using SBERT,then averaged the SBERT representations across the bios in order to obtain a medium-level representation. -->

<!--Facebook Audience: Facebook Marketing API, by given media ID, retrieve estimates of the audience who showed interest in the corresponding medium.->extract the audience distribution over the political spectrum, which is categorized into five classes ranging from *very conservative* to *very liberal*.-->

<!--YouTube Audience Statistics: We retrieved the following metadata to model audience interaction-> metedata(number of views, likes, dislikes, and comments for each video)->averaged these statistics across the videos to obtain a medium-level representation.-->

<!--Wikipedia page: retrieved the Wikipedia page for each medium, -> encode content using the pre-trained bert model. If a medium had no page in Wikipedia, we used a vector of zeros.-->

<!--Similarity, the articles, we fed the encoder with the first 510 tokens of the page’s content, and used as an output representation the average of the word represen- tations extracted from the second-to-last layer. If a medium had no page in Wikipedia, we us-->



<!--we use ==above== to train a classifier to predict the **political bias** and the **factuality** of reporting of news media. -->

<!--they extracted features from several sources of information, including articles published by each medium, what is said about it on Wikipedia, metadata from its Twitter profile and so on -->

<!--to combine linguistic feature and social context to build the model-->

<!--We compared the textual content of what media publish vs. who read it on social media, i.e., on Twitter, Facebook, and YouTube. We further modeled what was written about the target medium in Wikipedia.-->

<!--We have combined a variety of information sources, many of which were not explored for at least one of the target tasks, e.g., YouTube channels, political bias of the Facebook audience, and information from the profiles of the media followers on Twitter. -->

<!--We further modeled different modalities: text, metadata, and speech signal. **The evaluation results have shown that** while what was written matters most, the social media context is also important as it is complementary, and putting them all together yields sizable improvements over the SOTA state of the art.-->

<div style="page-break-after: always; break-after: page;"></div>

# System Architecture 

<img src="Day 9 Presetation.assets/image-20200907184021558.png" alt="image-20200907184021558" style="zoom: 50%;" />

<!--we present our system.-->

<!--Article: features/sources, each target medium, they extracts a variety of features to model the previous question (*i*) what was written by the medium, (*ii*) the audience of the medium on social media, and (*iii*) what was written about the medium in Wikipedia.-->



<div style="page-break-after: always; break-after: page;"></div>

# Bert Model

​	<!--Bidirectional Encoder Representations from transformer-->

<!--**down-stream task**-->

- Pre-trained bert 
- fun-tuning bert 



**softmax function** ${\displaystyle \sigma}$: $\sigma(z)_i =\frac{e^{zi}}{\sum_{j=1}^C e^{zj}} \ for \ i =1, ...,C \ and \ z=(z1,z2,...,z_C) \in \mathbb {R}^C $

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gikmlt0gfwj313z0u077l.jpg" alt="Image-1" style="zoom: 33%;" />





<!--train a classifier to predict the political bias and the factuality of reporting of news media. -->

<!--input vector z and normalize these values by dividing by the sum of all these its posterior probabilities, whether it is predicting the political bias or the factuality of the target news medium.-->

<!--softmax, they can be interpreted as probabilities.-->

<!--These probabilities are produced by training a softmax layer on top of the [CLS] token in the above-mentioned fine-tuned BERT model. We averaged the probability representations across the articles in order to aggregate them at the medium level.-->

##### <!--Tasks - extracts a variety of features from each target medium-->

- #### Bert Architecture

<img src="https://picb.zhimg.com/80/v2-988f9b0d3a2635393a91ead840364644_1440w.jpg" alt="img" style="zoom: 30%;" />

- #### Input 

<img src="Day 9 Presetation.assets/image-20200909195747122.png" alt="image-20200909195747122" style="zoom:50%;" />

<div style="page-break-after: always; break-after: page;"></div>

- #### Transformer

  <img src="Day 9 Presetation.assets/image-20200909200014197.png" alt="image-20200909200014197" style="zoom:40%;" />

<div style="page-break-after: always; break-after: page;"></div>

# Experimental setup and results

- ## Dataset

  ### Media Bias/Fact Check (MBFC) dataset

  <img src="Day 9 Presetation.assets/image-20200907202437100.png" alt="image-20200907202437100" style="zoom:50%;" />

<!--dataset got reduced to 864 news media-->

<!--which consists of a list of news media along with their labels of both political bias and factuality of reporting-->

<!--Factuality is modeled on a 3-point scale: *low*, *mixed*, and *high*.-->

<!--Political bias is modeled on a 3-point scale (*left*, *center*, and *right*)-->

<!--Statistical Description: (unbalanced)-->

<!--be able to retrieve Wikipedia pages for 61.2% of the media-->

<!--Twitter profiles for 72.5% of the media-->

<!--Facebook pages for 60.8% of the media-->

<!--YouTube channel for 49% of the media-->

- ## Experimental Setup

  - SVM classifiers

  <!--So, when generating features for each article to train the SVM, they don't just use the CLS token. They average all of the word embeddings in the second to last layer.-->

  <!--This is done for every article for each "media source". The averaged vectors are then averaged again (across each source's articles) to obtain the "media source" embeddings. -->

  - Combine features from each sources
  - 5-fold cross-validation to train and to evaluate an SVM model
  - macro-average F1 score and accuracy

   <!--i.e., averaging over the classes,-->

  <!--we evaluated the model on the remaining unseen fold. Ultimately, we report both macro-F1 score, and accuracy.-->

<!--they evaluated the three aspects mention above about news media separately and in combinations: (*i*) what the target medium wrote, (*ii*) who read it, and (*iii*) what was written about that medium.  -->

<!--train SVM classifiers for predicting the political bias and the factuality of reporting of news media.-->

<!--combining the best feature(s) from each aspect to obtain a combination that achieves even better results.-->

<!--used 5-fold cross-validation to train and to eval- uate an SVM model using different features and feature combinations. -->

<!--macro-F1 score, and accuracy.-->

<!--compared our results to the majority class-->

<!--compared our results NELA features from articles, (*ii*) embedding representations of Wikipedia pages using averaged GloVe word embeddings, (*iii*) metadata from the media’s Twitter profiles, and (*iv*) URL structural features. Since we slightlymodified the MBFC dataset, we retrained the old-->

<!--two strategies to evaluate feature combinations-->

<!--The first one trains a single classifier using all features. The second one trains a separate classifier for each feature type and then uses an ensemble by taking a weighted average of the posterior probabilities of the individual models.-->



<div style="page-break-after: always; break-after: page;"></div>

# Results

<img src="Day 9 Presetation.assets/image-20200907212252529.png" alt="image-20200907212252529" style="zoom:48%;" />

<!--individual features, while the lower ones show combinations thereof.-->

<!--rows 3--5 show that averaging embeddings fromfine-tuned BERT to encode arti- cles (row 4) works better than using NELA features (row 3) -->

<!--They also show that using the posterior probabilities obtained from applying a softmax on top of BERT’s [CLS] token (row 5) performs worse than using average embeddings (row 4). -->

<!--better to incorporate information from the articles’ word representations-->

<!--rows 7--10 show that captions are the most useful type of feature among those ex- tracted from YouTube.-->

<!--Rows 11-16 show the results for systems that combine article, Twitter, and YouTube features, ei- ther directly or in an ensemble. -->

<!--compare rows 6 and 17, -->

<!--20--23 show that the YouTube metadata features,YouTube metadata features improve the perfor- mance, Facebook audi- ence features’ performance hurts the overall performance-->

<!--Row 24 shows that the Wikipedia features per- form worse than most individual features above,-->

<!--rows 25--32 in Table 3 show the evaluation results when combining all aspects.-->



<!--see that the best results are achieved when using the best features from each of the three aspects,-->

<!--Similarly to the results for political bias predic- tion,-->

<!--11--16 show that combining the Twitter profile features with the BERT-encoded articles improves the performance -->

<!--Comparing rows 6 and 17 in Table 3, we can see that the Twitter follower features perform worse than using Twitter profiles features-->

<!--Finally, rows 25--32 show the results for mod- eling combinations of the three aspects we are ex- ploring in this paper. The best results are achieved using the best features selected from the *What was written* and the *What was written about the target medium* aspects, concatenated together.-->

<img src="Day 9 Presetation.assets/image-20200907220337771.png" alt="image-20200907220337771" style="zoom:50%;" />

<div style="page-break-after: always; break-after: page;"></div>

# The End

<!--For the group presentations, make sure your slides at least cover the following: 1,What is the task/problem the paper is addressing?2,How did they solve the problem? Describe the method in detail. (edited) 3,What are the results of their solution/analysis?4,Are there any obvious ways to build on the results in the paper? Are there any ways to apply it to our current projects? This part is based on your opinion. If it can't be applied to current projects, it is okay to state that. The goal is general **brainstorming** of projects. (edited) -->

We further want to model the network structure, e.g., using graph em- beddings (Darwish et al., 2020). Another research direction is to profile media based on their stance with respect to previously fact-checked claims

Finally, we plan to experiment with other languages.