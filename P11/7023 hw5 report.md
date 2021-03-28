**HW**5:

 Access STATA. Cut and paste: “use http://www.stata-press.com/data/r13/nlswork” 

1. Create dummy variables for “year” 

   a.   Test for multicollinearity, heteroscedasticity and outliers. 

2. DV = lnwages; IVs = college grad, grade, year, union. 

   a.   Test for **multicollinearity**, **heteroscedasticity** and outliers. Any changes needed?

   b.   Use random effects (using idcode)

   c.   Use fixed effects (using idcode)

   d.   Conduct a **Hausman** test to determine whether fixed effects or random effects is appropriate.

   e.   Try “areg”. Compare the results versus the FE model. 



Note that **grade** and **black**were omitted from the model because they do not vary within person.



## Robust regression analysis

In most cases, we begin by running an OLS regression and doing some diagnostics. We will begin by running an OLS regression. The **lvr2plot** is used to create a graph showing the leverage versus the squared residuals, and the **mlabel** option is used to label the points on the graph with the two-letter abbreviation for each state. 

https://stats.idre.ucla.edu/stata/dae/robust-regression/

As we can see, DC, Florida and Mississippi have either high leverage or large residuals. Let’s compute Cook’s D and display the observations that have relatively large values of Cook’s D. To this end, we use the **predict** command with the **cooksd** option to create a new variable called **d1** containing the values of Cook’s D. Another conventional cut-off point is **4/n**, where **n** is the number of observations in the data set. We will use this criterion to select the values to display. 

> 



```
xtreg ln_wage collgrad grade union i.year, fe  cluster(idcode)

areg ln_wage collgrad grade union i.year, absorb(idcode)

areg y i.year, absorb(industry)

```





The VIFs look fine here.

http://econometricstutorial.com/2015/03/panel-data-on-stata-fe-re/

The regression command for panel data is **xtreg**

```
xtreg ln_wage age race tenure, fe  cluster(idcode)// without the fe option by default is random effect
xtreg ln_w grade age c.age#c.age ttl_exp c.ttl_exp#c.ttl_exp tenure
```

or using another command that is **areg**, which syntax is:

```
areg ln_wage age race tenure, absorb(idcode)
```

When using areg you always have to specify which variable you want to absorb. The common practice is to generate the dummies for each observation and absorb them. Even though they are not displayed, the overall F-test of the model report their presence.  Areg has three different syntax you can use. I report them ordered:



```
quietly xtreg ln_wage collgrad grade union i.year, re
```



## **Hausman Test**



```
xttest1 ln_wage collgrad grade union i.year

regress ln_wage collgrad grade union i.year
xtreg ln_wage collgrad grade union, i(year) fe

xtreg ln_wage collgrad grade union, i(year) fe cluster(idcode)

xtreg ln_wage collgrad grade union i.year

xtreg ln_wage collgrad grade union i.year, fe vce(robust) basel
xtreg ln_wage collgrad grade union i.year, fe vce(robust) 
xttest3


xtgls ln_wage collgrad grade union year, igls panels(heteroskedastic)
xtserial ln_wage collgrad grade union year
```

One way to measure **multicollinearity** is the variance inflation factor (VIF)



```text
检验自相关&异方差
横截面数据主要考虑异方差，时间序列主要考虑自相关。
同时存在异方差和自相关，先考虑产生自相关的原因是模型误设还是纯粹的自相关。

```



```
xtgls depvar indepvars, igls panels （heteroskedastic）
estimates store heterotest
xtgls depvar indepvars
local df = e(N_g) - 1
lrtest heterotest . , df(`df')
```

**minixi 是用LR test**

**xttest3是用wald test**

**您可以用二者分別檢定,看結果是否有一致**

