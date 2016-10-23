# Hamilton_Project

#Introduction
Below, I will describe the process I chose for using the Randow Forest
machine learning techique for predicting the class variable in this dataset

##Step 1: Data Cleaning and remove unuseable attributes

The first step is to load the data and remove all of the NA values in the 
dataset. Next, I removed the attributes that aren't useful in the prediction.
These variables for removal are x, user_name, raw_timestamp_part 1 and 2, cvtd_timestamp
new_window, and num_window.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(e1071)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)

#load data
traindata <- read.csv("pml-training.csv", header=T, na.strings=c("NA", "#DIV/0!"))
testdata <- read.csv("pml-testing.csv", header=T, na.string=c("NA", "#DIV/0!"))

#remove nas

traindata_NARemoved<-traindata[, apply(traindata, 2, function(x) !any(is.na(x)))]
testdata_NARemoved<-testdata[, apply(testdata, 2, function(x) !any(is.na(x)))]

#Remove features that will not be used for feature detection
finaltrainingdata <- traindata_NARemoved[,-c(1:7)]
finaltestdata <- testdata_NARemoved[,-c(1:7)]
```

##Step 2: Model Setup
During this step, I partitioned the data with 75% for training and 25% for testing.
I used the K-folds technique to cross valid my data set by creating 10 folds. Lastly
I ran the Random Forest on the training set.


```r
#Split training set in training and test set and ran Random forest algorthim 
#The cross vaildation technique was used was K-Folds in which I create 10
#of them.
inTrain<-createDataPartition(y=finaltrainingdata$classe, p=0.75,list=F)
training<-finaltrainingdata[inTrain,] 
test<-finaltrainingdata[-inTrain,] 
set.seed(1)
FoldsSplit <-trainControl(method="cv", number=10, allowParallel=T, verbose=T)
ModFit <-train(classe~.,data=training, method="rf", trControl=FoldsSplit, verbose=F)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## + Fold01: mtry= 2 
## - Fold01: mtry= 2 
## + Fold01: mtry=27 
## - Fold01: mtry=27 
## + Fold01: mtry=52 
## - Fold01: mtry=52 
## + Fold02: mtry= 2 
## - Fold02: mtry= 2 
## + Fold02: mtry=27 
## - Fold02: mtry=27 
## + Fold02: mtry=52 
## - Fold02: mtry=52 
## + Fold03: mtry= 2 
## - Fold03: mtry= 2 
## + Fold03: mtry=27 
## - Fold03: mtry=27 
## + Fold03: mtry=52 
## - Fold03: mtry=52 
## + Fold04: mtry= 2 
## - Fold04: mtry= 2 
## + Fold04: mtry=27 
## - Fold04: mtry=27 
## + Fold04: mtry=52 
## - Fold04: mtry=52 
## + Fold05: mtry= 2 
## - Fold05: mtry= 2 
## + Fold05: mtry=27 
## - Fold05: mtry=27 
## + Fold05: mtry=52 
## - Fold05: mtry=52 
## + Fold06: mtry= 2 
## - Fold06: mtry= 2 
## + Fold06: mtry=27 
## - Fold06: mtry=27 
## + Fold06: mtry=52 
## - Fold06: mtry=52 
## + Fold07: mtry= 2 
## - Fold07: mtry= 2 
## + Fold07: mtry=27 
## - Fold07: mtry=27 
## + Fold07: mtry=52 
## - Fold07: mtry=52 
## + Fold08: mtry= 2 
## - Fold08: mtry= 2 
## + Fold08: mtry=27 
## - Fold08: mtry=27 
## + Fold08: mtry=52 
## - Fold08: mtry=52 
## + Fold09: mtry= 2 
## - Fold09: mtry= 2 
## + Fold09: mtry=27 
## - Fold09: mtry=27 
## + Fold09: mtry=52 
## - Fold09: mtry=52 
## + Fold10: mtry= 2 
## - Fold10: mtry= 2 
## + Fold10: mtry=27 
## - Fold10: mtry=27 
## + Fold10: mtry=52 
## - Fold10: mtry=52 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 2 on full training set
```

##Step 3: Results 
The model has an accuracy of 99% in prediction rate of the test set. Also, I 
ran the model against the test case for the quiz and it was 100% correct.


```r
#Prediction Results
predResults<-predict(ModFit, newdata=test)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
confusionMatrix(predResults, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1392    3    0    0    0
##          B    3  944    5    0    0
##          C    0    2  848   18    0
##          D    0    0    2  785    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9929         
##                  95% CI : (0.9901, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.991          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9947   0.9918   0.9764   0.9989
## Specificity            0.9991   0.9980   0.9951   0.9993   0.9998
## Pos Pred Value         0.9978   0.9916   0.9770   0.9962   0.9989
## Neg Pred Value         0.9991   0.9987   0.9983   0.9954   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2838   0.1925   0.1729   0.1601   0.1835
## Detection Prevalence   0.2845   0.1941   0.1770   0.1607   0.1837
## Balanced Accuracy      0.9985   0.9964   0.9934   0.9878   0.9993
```

```r
#prediction against test cases
testcaseResults <- predict(ModFit, newdata = finaltestdata)
testcaseResults
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

