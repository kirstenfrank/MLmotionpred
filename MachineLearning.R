URLtrain<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(URLtrain,destfile="training.csv",method="curl")
trainset<-read.csv("training.csv")

URLtest<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(URLtest,destfile="testing.csv",method="curl")
testset<-read.csv("testing.csv")

## There are only 20 records in testing.csv. This is not enough for 
## validation of a real machine-learning algorithm. 
## So, we need to divide training.csv into a training and test
## set, which we will call a validation set to avoid further 
## confusion. 
library(caret)
inTrain<-createDataPartition(trainset$classe,p=0.6,list=FALSE)
MLtrain<-trainset[inTrain,]
MLvalid<-trainset[-inTrain,]
plot(MLtrain$classe)
nearZeroVar(MLtrain)  #This identifies variable with little variance.
