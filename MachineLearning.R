URLtrain<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(URLtrain,destfile="training.csv",method="curl")
trainset<-read.csv("training.csv",stringsAsFactors=FALSE)

URLtest<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(URLtest,destfile="testing.csv",method="curl")
testset<-read.csv("testing.csv",stringsAsFactors=FALSE)

## There are only 20 records in testing.csv. This is not enough for 
## validation of a real machine-learning algorithm. 
## So, we need to divide training.csv into a training and test
## set, which we will call a validation set to avoid further 
## confusion. 
library(caret)
set.seed(1111)
inTrain<-createDataPartition(trainset$classe,p=0.6,list=FALSE)
MLtrain<-trainset[inTrain,]
MLvalid<-trainset[-inTrain,]

## Exploratory Data Analysis on the training set.
plot(MLtrain$classe)  
    # Histogram to see the ratio of different classes.
temp <- MLtrain[, which(as.numeric(colSums(is.na(MLtrain))) < 500)]
lowVar<-nearZeroVar(temp)  

    # This identifies variables with little variance.
    # Few observations in low frequency classes can cause a problem
    # in cross-validation methods. These tend to DIV by 0 errors.
    # Column 6 is a Yes/No factor variable
    # Keep column 6 but remove all the others in lowVar
lowVar<-lowVar[-1]
temp<-temp[,-lowVar]
temp$classe<-as.factor(temp$classe)
# spelling correction in names of the features.
names(temp)<-gsub("picth","pitch",names(temp))
temp$raw_timestamp_part_1<-NULL
temp$raw_timestamp_part_2<-NULL
temp$cvtd_timestamp<-NULL
temp$X<-NULL      ## this column is a row number
keepnames<-names(temp)


# Plots to reveal any structure.
    featurePlot(x=temp[,c("num_window","roll_belt","pitch_belt","yaw_belt","pitch_arm","roll_dumbbell")],
            y=temp$classe,plot="pairs")  
    # roll_dumbbell and pitch_arm seem to separate well with num_window.
    featurePlot(x=temp[,c("total_accel_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x")],
            y=temp$classe,plot="pairs")  
    # only a little separation
    featurePlot(x=temp[,c("accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z")],
            y=temp$classe,plot="pairs")                
    # spreads out the red class from the yellow class
    featurePlot(x=temp[,c("roll_arm","pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x")],
            y=temp$classe,plot="pairs")  
    # some separation in roll_arm, pitch_arm,yaw_arm, total_accel_arm.
    # This is the first thing that separates the green class (pitch_arm)
    featurePlot(x=temp[,c("gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z")],
            y=temp$classe,plot="pairs") 
    # a bit of separation between pink and yellow and red and yellow
    featurePlot(x=temp[,c("magnet_arm_x","magnet_arm_y","magnet_arm_z","roll_dumbbell","pitch_dumbbell")],
            y=temp$classe,plot="pairs") 
    # Dancing slugs, but pitch_dumbbell separates pinks
    featurePlot(x=temp[,c("yaw_dumbbell","total_accel_dumbbell","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z")],
            y=temp$classe,plot="pairs") 
    # Some separation pink and red in gyros_dumbbell_x
    featurePlot(x=temp[,c("accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z")],
            y=temp$classe,plot="pairs") 
    # magnet_dumbbell_z with magnet_dumbbell_x separate pink well
    # accel_dumbbell_z separates blue well
    featurePlot(x=temp[,c("roll_forearm","pitch_forearm","yaw_forearm","total_accel_forearm","gyros_forearm_x")],
            y=temp$classe,plot="pairs") 
    # Blue separates out a lot in roll_forearm and pitch_forearm, red separates in 
    featurePlot(x=temp[,c("gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y","accel_forearm_z")],
            y=temp$classe,plot="pairs") 
    # Some blue separation in gyros_forearm_z
    featurePlot(x=temp[,c("magnet_forearm_x","magnet_forearm_y","magnet_forearm_z")],
            y=temp$classe,plot="pairs") 
    # Wow, these look like worms!!
   

# Training with Random Forest
    temp$classe<-as.factor(temp$classe)
    temp$user_name<-as.factor(temp$user_name)
    temp$new_window<-as.factor(temp$new_window)
    fitControl<-trainControl(method="repeatedcv",
                             number=10,
                             repeats=10)
    randforestfit<-train(classe~.,temp,
                         method="rf", 
                         ntree=350,
                         preProc=c("center","scale"),
                         tuneLength =3,
                         trControl=fitControl)

# make MLvalid same as MLtrain (factors, remove columns)
    names(MLvalid)<-gsub("picth","pitch",names(MLvalid))
    validset<-MLvalid[,which(colnames(MLvalid) %in% keepnames)]
    validset$classe<-as.factor(validset$classe)
    validset$user_name<-as.factor(validset$user_name)
    validset$new_window<-as.factor(validset$new_window)

# do the prediction on validation set
    prediction<-predict(randforestfit,validset)
    confusionMatrix(prediction,validset$classe)

# do the prediction on test set
    names(testset)<-gsub("picth","pitch",names(testset))
    cleantest<-testset[,which(colnames(testset) %in% keepnames)]
    prediction2<-predict(randforestfit,cleantest)