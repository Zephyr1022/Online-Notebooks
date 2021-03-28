Policy Death 

医疗水平

求y值介于0和1之间，是因为数据中的分类特征（例如"方向"）属于"字符"类型。您需要使用 as.factor（data $ Direction）将它们转换为" factor"类型。因此： glm（Direction〜lag2，data = ...）无需声明stock.direction。 

# Forecasting the US elections

https://projects.economist.com/us-2020-forecast/president/how-this-works



# Forecasting the 2020 US Elections with Decision Desk HQ: Methodology for Modern American Electoral Dynamics

https://hdsr.mitpress.mit.edu/pub/gach7e59/release/1?readingCollection=c6cf45bb

Both categories span various domains, including candidate fundraising, demographic information, economic indicators, electoral history, and political environment.

Using the computed probabilities for each House, Senate, and Presidential race, we predict the aggregate number of seats we expect the GOP to win and the probability of maintaining control of the House and Senate.



- We don't know the actural situation 
- so we use weight to simulation the data

- We perform over 10 million simulations to create a distribution of potential outcomes.
- For example, we can find the most likely path to victory for a candidate, contingent on them winning or losing in a specific set of states.
- For the Presidential race, we draw from a **Binomial distribution** for each state and then calculate electoral college totals in order to determine the overall distribution of electoral votes. Attempts to force certain correlations between states did not produce significantly different simulation results.

# ==Data Science and Predictive Analytics (UMich HS650)==

## Improving Model Performance

http://www.socr.umich.edu/people/dinov/courses/DSPA_notes/14_ImprovingModelPerformance.html

##### 1 Improving model performance by parameter tuning



### **Step 1: Understand what tuning machine learning model is**

https://www.kdnuggets.com/2019/01/fine-tune-machine-learning-models-forecasting.html



AUC for logistic regression model is equal to 0.9580181

AUC for LDA is equal to 0.996588

AUC for QDA is equal to 0.991692

AUC for Naive Bayes is equal to 0.9462988

AUC for Knn k=1 is equal to 0.898679

AUC for Knn k=7 is equal to 0.9192256

AUC for scale Knn k=1  is equal to 0.9589082

AUC for scale Knn k=7 is equal to 0.9589082



https://bradleyboehmke.github.io/HOML/knn.html





nohup python test_m_s_d.py > results_d_s_m.tsv 2>&1 &





##### ==US voter demographics: election 2020 ended up looking a lot [like](https://www.theguardian.com/us-news/2020/nov/05/us-election-demographics-race-gender-age-biden-trump) 2016==

So far, the picture appears to be strikingly similar to what it was in 2016, said the political science professor Charles H Stewart, founding director of MIT’s Election Data and Science Lab.

“There were slight changes, but the changes in the electorate, at least the ones who showed up to vote on election day, are much less dramatic than we were being led to believe by the pre-election polls,” Stewart said.

Pollsters had predicted this election would see the widest gender gap since women won the vote 100 years ago, but that does not appear to have transpired.

Stewart noticed a slight widening of the gender gap – with women voting 56-43 for Biden, while the two candidates were almost tied among men

But one of the biggest divides that did come to pass was between older voters and those aged under 30, who became even “less enamoured of President Trump than before”.

“The other age groups, 30-44, 45-64, 65 and over, it’s a pretty close divide between Biden and Trump. So it’s really young people who are overwhelmingly anti-Trump and that’s really noticeable.”

He said Trump also lost some appeal among low-income voters, who were more attracted to Biden, but the president gained among voters with family incomes over $100,000 a year.

“That right now appears to be the biggest demographic shift I’m seeing. And you can tie that to [Trump’s] tax cuts [for the wealthy] and lower regulations.”

He added: “For as much as we talk about the culture wars and all of those sorts of things, it looks like the big thing was good old-fashioned pocketbook economics.”



While evidence is lacking in exactly who voted, he said the increase in turnout probably came from young people and the [Latino community](https://www.theguardian.com/us-news/2020/nov/05/florida-latino-voters-joe-biden-donald-trump), who he said “historically have been significantly underrepresented in the electorate.”



**America’s demographic landscape is not only becoming increasingly more [diverse](https://www.pewresearch.org/2020/09/23/the-changing-raci al-and-ethnic-composition-of-the-u-s-electorate/), it’s also shifting national voting behavior.** In a recent [analysis](https://www.pewresearch.org/politics/2020/06/02/in-changing-u-s-electorate-race-and-education-remain-stark-dividing-lines/) of political-party composition, the Pew Research Center noted that “The Republican and Democratic coalitions, which bore at least some demographic similarities in past decades, have strikingly different profiles today.” This is especially true in terms of certain demographic traits, including race and ethnicity and religious affiliation. For example, the GOP has the lead among non-Hispanic white voters and religious people, while the Democratic Party is favored more by women and nonwhite voters. When it comes to racial minorities in particular, they make up [four in 10](https://www.pewresearch.org/politics/2020/06/02/the-changing-composition-of-the-electorate-and-partisan-coalitions/) Democratic voters compared to two in 10 Republican voters.



who voted for Barack Obama in 2008 and Donald Trump in 2016 in order to build “2008 and 2016-Like” voting models to simulate future election outcomes. 

**Our Assumption:** 

To let readers test their own assumptions about how these kinds of demographic shifts might affect the election, we’ve created an interactive tool that accompanies this article.



## 誤差[[编辑](https://zh.wikipedia.org/w/index.php?title=票站調查&action=edit&section=2)]

和其他民意調查一樣，票站調查也會有「誤差」。最著名的案例是1992年[英國](https://zh.wikipedia.org/wiki/英國)國會大選。當時兩個票站調查都推算，選舉結果將會令國會內沒有任何政黨可以取得過半議席，但是實際點票結果顯示，梅傑領導的保守黨政府雖然議席大減，但是仍以些微少數可以獨自組成多數派政府。

票站調查的一個要害是選取調查對像時會有偏差。雖然一般來說，票站調查訪問的人數比民意調查更多，但是始終只是所有選民的一小部份。由於調查機構多數由[傳媒](https://zh.wikipedia.org/wiki/傳媒)贊助，而傳媒往往希望可以在投票結束後第一時間公布投票的推算結果，調查機構唯有在投票結束前幾小時停止搜集數據以爭取時間整理推算，所以沒有將投票最後一段時間計算在內，容易出現偏差。因為某些選民，例如老年人和主婦，比較喜歡在早上投票，他們的意向可能會在票站中被不成比例地放大。相反，較遲投票的人就會被忽略。同時，選民願意接受訪問與否，以及會不會如實地說出自己的投票選擇，亦都會左右誤差幅度。

為了減少誤差，過去有調查機構透過 "polling" 方式，將各自搜集到的數據共同享用，以增加準確度。2005年英國大選，英國廣播公司同獨立電視ITV就透過這種方式，預測英國工黨將會在大選中贏得 66 席，和後來官方公布的選舉結果脗合。2007年[澳洲](https://zh.wikipedia.org/wiki/澳洲)大選，澳洲多間傳媒機構亦透過共享數據方式，準確預測澳洲工黨以百分之五十三的得票率擊敗執政聯盟。

## 爭議



##### How Demographics Will [Shape](https://fivethirtyeight.com/videos/how-demographics-will-shape-the-2016-election/) The 2016 Election

##### Demographic Shift Poised to Test Trump’s 2020 Strategy 

Base of white, working-class voters projected to decline and be replaced by Democratic-leaning groups



##### Demographic shifts since 2016 could be enough to defeat Trump. But it's [complicated](https://www.nbcnews.com/politics/2020-election/demographic-shifts-2016-could-be-enough-defeat-trump-it-s-n1240724).

[Tools](https://www.nbcnews.com/specials/swing-the-election/#active=race_education&t-cew=63&t-ncw=76&t-black=39&t-latino=71&t-ao=71&d-cew=54&d-ncw=31&d-black=92&d-latino=72&d-ao=73&t-age1829=433&t-age3044=53&t-age4564=635&t-age65up=682&d-age1829=596&d-age3044=554&d-age4564=471&d-age65up=473)



##### [Importing](https://www.statmethods.net/input/importingdata.html) Data



https://electionlab.mit.edu/data

http://2020-us-election-apis.postman.com



## 美国历任总统[名单](http://114.xixik.com/us-president/)





https://github.com/BuzzFeedNews/2020-10-electoral-college-effect-by-demographic