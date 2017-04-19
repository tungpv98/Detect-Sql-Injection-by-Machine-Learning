SQL-Injection Detect with Machine Learning
=============================================

<br>This python code is **detect sql injection** with **machine-learning**.
<br>I use classification method and GBT(Gradient Boosting Tree) algorithm.
<br>You can also use RandomForest, SVM, etc.(easy too use from sklearn.)
<br>**Before run this code you should change basedir path!**

## Here is a result
```
Scott:StudyML Scott$ python ml_detect_sqli.py
sqli     5921
plain    3694
Name: type, dtype: int64
Gradient Boosting Tree Acurracy: 1.000000 (Oh!! 100% it is true?)
[SQL-Injection]: -1988 union select scott, python, machine, learning, study, version, 1--

<predict sql or not in file>
Scott:StudyML Scott$ python ml_detect_sqli.py
sqli     5921
plain    3694
Name: type, dtype: int64
Gradient Boosting Tree Acurracy: 1.000000
----- Result: -----
Machine Check Count: 13780
Sql-injection: 9731
Plain-text: 4049
--- 286.365072966 seconds --- (my mac is old... lol mac's fan is angry and yelling.)
```

## Limitation
I am now just learning about machine learning.
<br>So I need the advice of an expert like you.
<br>I hope you can give me some advice to make it a better program.
<br>I hope this script will be more advanced and useful in many places.
