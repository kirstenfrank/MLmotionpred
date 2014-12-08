# Prediction of exercise form
Kirsten Frank  
November 23, 2014  

# Summary

Machine learning was used to evaluate the feasibility of qualitatively assessing exercise quality in real time. The high accuracy of classification of exercise types demonstrated the fruitfulness of this approach.  

# Background

The use of inexpensive motion sensors, accelerometers, magnetometers,  and other devices promises to revolutionize exercise coaching. This devices allow a coach to examine fine details of the exerciser's position, motion and timing. Other devices can be attached to the exercise equipment and measure the position, motion and timing of the equipment. This data requires a lot of processing before it can interpreted. 

There are two proposals for methods to do this processing. One is to process into a physical representation of the motion and compare it to an idealized model of the exercise. The other is to measure people doing the exercise and let computers find the pattern and make a model using machine learning. 

# Methods

## Data collection
Data collected on six male volunteers each doing a dumbbell curl in five different ways. Four of these ways are mistakes that a coach would correct and the other is the correct way. 

The data was made available freely, thanks to researchers. Data is kindly supplied by http://groupware.les.inf.puc-rio.br/har with citations to their paper [@ugulino2012].

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3Ibz6YXYU


```r
URLtrain<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(URLtrain,destfile="training.csv",method="curl")
trainset<-read.csv("training.csv",stringsAsFactors=FALSE)

URLtest<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(URLtest,destfile="testing.csv",method="curl")
testset<-read.csv("testing.csv",stringsAsFactors=FALSE)
```
## Data processing
In order to use machine learning on this corpus of data, we need to divide it into a training set and a validation set. The purpose of a validation set is to allow us to estimate out-of-sample error. 40% of the observations were put into the validation set, and the remaining 60% were used to train the model.


```r
library(caret)
set.seed(1111)
inTrain<-createDataPartition(trainset$classe,p=0.6,list=FALSE)
MLtrain<-trainset[inTrain,]
MLvalid<-trainset[-inTrain,]
```

The data have 160 columns. Many of them are summary columns and don't contain values at every point. In order to use this data simply and effectively, columns that contained mostly NAs or blank characters were removed.


```r
temp <- MLtrain[, which(as.numeric(colSums(is.na(MLtrain))) < 500)]
lowVar<-nearZeroVar(temp)  

    # This identifies variables with little variance.
    # Few observations in low frequency classes can cause a problem
    # in cross-validation methods. These tend to DIV by 0 errors.
    # Column 6 is a Yes/No factor variable
    # Keep column 6 but remove all the others in lowVar
lowVar<-lowVar[-1]
temp<-temp[,-lowVar]
```

Several other columns contained information that was specific to the experiment and unlikely to be generalizable, such as the datestamp and elapsed time. The X variable turned out to be an observation number (like a row number) and not generalizable. All of these were removed. The spelling error of picth instead of pitch was corrected and column names were saved to be used to remove columns from the validation set and the test set.


```r
names(temp)<-gsub("picth","pitch",names(temp))
temp$raw_timestamp_part_1<-NULL
temp$raw_timestamp_part_2<-NULL
temp$cvtd_timestamp<-NULL
temp$X<-NULL      ## this column is a row number
keepnames<-names(temp)
```

The user name was kept because the test set is a variety of exercises by the same 6 test subjects. With that information, our implementation will recognize an exercise type by any one of the 6 test subjects. We do acknowledge that using the user name as a feature prevents generalization to an arbitary individual. Several text columns were converted to factors.


```r
    temp$classe<-as.factor(temp$classe)
    temp$user_name<-as.factor(temp$user_name)
    temp$new_window<-as.factor(temp$new_window)
```

## Data exploration

In order to interpret the results of machine learning, we did not plan to use principal component analysis. This meant that we could explore the remaining 55 features using their original names.

Preliminary plots comparing pairs of features were generated from the data. Data was chosen every 150 rows, so as to avoid overplotting that would obscure the data. An ideal pair of features would have points in 5 organized color groups, well separated, but no pair like that was found.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
#featurePlot(x=temp[seq(1,nrow(temp),by=150),c("num_window",
#                                              "pitch_arm",
#                                              "roll_dumbbell",
#                                              "pitch_dumbbell",
#                                              "roll_belt",
#                                              "pitch_forearm")],
#            y=temp$classe[seq(1,nrow(temp),150)],plot="pairs") 

#featurePlot(x=temp[seq(1,nrow(temp),by=150),c("roll_forearm",
#                                              "pitch_forearm",
#                                              "yaw_forearm",
#                                              "total_accel_forearm",
#                                              "gyros_forearm_x")],
#            y=temp$classe[seq(1,nrow(temp),by=150)],plot="pairs") 
```

## Machine Learning and Prediction

The method chosen for the machine learning was Random Forests [@breiman2001]. This method is very useful for classification prediction with many predictors. It is considered to be more robust to noise than many other classification parameters. 

It is an ensemble extension of decision trees with an additional random component. Multiple trees are built using a random subset of the predictors. It can be built using k-fold cross-validation. After many attempts at training with various tunable parameters (number of trees, number of folds in the cross-validation, grid resolution for tuning the number of random predictors per tree), these parameters were chosen (number of trees=350, number of folds=10, grid resolution=3 divisions). Numerical variables were centered and scaled in preprocessing. 


```r
    fitControl<-trainControl(method="repeatedcv",
                             number=10,
                             repeats=10)
    randforestfit<-train(classe~.,temp,
                         method="rf", 
                         ntree=350,
                         preProc=c("center","scale"),
                         tuneLength =3,
                         trControl=fitControl)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```
# Results and Conclusion

The resulting model fit had an in-training-set accuracy of 0.9961. We then evaluated the influence of the various predictors.


```r
influence<-varImp(randforestfit)
influence[1]
```

```
## $importance
##                          Overall
## user_namecarlitos      0.6961922
## user_namecharles       0.5489560
## user_nameeurico        1.4466662
## user_namejeremy        0.4493306
## user_namepedro         0.2830500
## new_windowyes          0.0000000
## num_window           100.0000000
## roll_belt             67.4943416
## pitch_belt            29.4688755
## yaw_belt              35.2325501
## total_accel_belt       3.1232502
## gyros_belt_x           1.7602157
## gyros_belt_y           1.8557558
## gyros_belt_z           5.6228756
## accel_belt_x           1.7052365
## accel_belt_y           1.9555456
## accel_belt_z          10.6764469
## magnet_belt_x          7.3021364
## magnet_belt_y          8.3180873
## magnet_belt_z          8.9447548
## roll_arm               6.1117663
## pitch_arm              3.3791350
## yaw_arm                4.6242797
## total_accel_arm        1.5470762
## gyros_arm_x            1.9736383
## gyros_arm_y            2.3972560
## gyros_arm_z            0.8426558
## accel_arm_x            4.5708387
## accel_arm_y            2.1012234
## accel_arm_z            1.8009981
## magnet_arm_x           5.1597391
## magnet_arm_y           3.6135121
## magnet_arm_z           2.6103264
## roll_dumbbell         12.8568549
## pitch_dumbbell         3.4147514
## yaw_dumbbell           6.0753674
## total_accel_dumbbell  10.2426721
## gyros_dumbbell_x       2.2509868
## gyros_dumbbell_y       4.5570528
## gyros_dumbbell_z       1.5465841
## accel_dumbbell_x       4.7695026
## accel_dumbbell_y      14.3654511
## accel_dumbbell_z       9.5579568
## magnet_dumbbell_x     11.9735186
## magnet_dumbbell_y     31.7667912
## magnet_dumbbell_z     31.6368281
## roll_forearm          24.2569721
## pitch_forearm         41.7159390
## yaw_forearm            3.4956735
## total_accel_forearm    1.4911065
## gyros_forearm_x        1.0131511
## gyros_forearm_y        1.9125129
## gyros_forearm_z        1.2471705
## accel_forearm_x       11.2058456
## accel_forearm_y        2.1733287
## accel_forearm_z        4.7461287
## magnet_forearm_x       4.3615746
## magnet_forearm_y       4.0705655
## magnet_forearm_z       7.9263984
```

We used the list of retained columns to prepare the validation set for prediction. Prediction on the validation set allows us to calculate an out-of-sample error rate.


```r
# make MLvalid same as MLtrain (factors, remove columns)
    names(MLvalid)<-gsub("picth","pitch",names(MLvalid))
    validset<-MLvalid[,which(colnames(MLvalid) %in% keepnames)]
    validset$classe<-as.factor(validset$classe)
    validset$user_name<-as.factor(validset$user_name)
    validset$new_window<-as.factor(validset$new_window)
```


```r
# do the prediction on validation set
    prediction<-predict(randforestfit,validset)
    validationnumbers<-confusionMatrix(prediction,validset$classe)
```

Overall accuracy on the validation set (out-of-sample accuracy) was 0.996. The accuracy by class was 0.999 for class A (the correctly performed exercise); 0.997 for class B; 0.997 for class C; 0.994 for class D; and 0.999 for class E.

In conclusion, machine learning using Random Forests works well for this type of data. 

# References

---
