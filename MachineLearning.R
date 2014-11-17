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
lowVar<-nearZeroVar(MLtrain)  
    # This identifies variables with little variance.
    # Few observations in low frequency classes can cause a problem
    # in cross-validation methods. Inspect these columns by hand.
    # Column 6 is a Yes/No variable, but other columns contain 
    # #DIV/0 errors. (12,13,14,15,16,17,20,26,69,70,71,72,
    # 73,74,87,90,92,101,125,126,127,128,129,130,139) and 
    # NAs (51,52,53,54,55,56,57,58,59,78,81,89,142,143,144,145,
    # 146,147,148,149,150) 
    # and other ones with character values 
    # (23, 88, 91, 95, 98, 133, 136)
    # Keep column 6 but remove all the others in lowVar
lowVar<-lowVar[-1]
MLtrainsmall<-MLtrain[,-lowVar]
MLtrainsmall$classe<-as.factor(MLtrainsmall$classe)
# spelling correction in names of the features.
names(MLtrainsmall)<-gsub("picth","pitch",names(MLtrainsmall))
featurePlot(x=MLtrainsmall[,c("num_window","roll_belt","X","pitch_belt")],
            y=MLtrainsmall$classe,plot="pairs")  
    # X seems to separate well with num_window.
featurePlot(x=MLtrainsmall[,c("yaw_belt","total_accel_belt","gyros_belt_x","gyros_belt_y")],
            y=MLtrainsmall$classe,plot="pairs")  
    # maybe yaw_belt
featurePlot(x=MLtrainsmall[,c("gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z")],
            y=MLtrainsmall$classe,plot="pairs")                
    # maybe gyros_belt_z And combo of gyros_belt_z and accel_belt_z
featurePlot(x=MLtrainsmall[,c("magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm")],
            y=MLtrainsmall$classe,plot="pairs")  
    # some separation in roll_arm
featurePlot(x=MLtrainsmall[,c("pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x")],
            y=MLtrainsmall$classe,plot="pairs") 
    # some separation in pitch_arm and yaw_arm
featurePlot(x=MLtrainsmall[,c("var_yaw_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Outlier in var_yaw_belt, separation in gyros_belt_z
featurePlot(x=MLtrainsmall[,c("accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x")],
            y=MLtrainsmall$classe,plot="pairs") 
    # No separation
featurePlot(x=MLtrainsmall[,c("magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Some separation in pitch_arm and in magnet_belt_y
featurePlot(x=MLtrainsmall[,c("yaw_arm","total_accel_arm","var_accel_arm","gyros_arm_x")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Some separation in yaw_arm
featurePlot(x=MLtrainsmall[,c("gyros_arm_z","accel_arm_x","accel_arm_y","gyros_arm_y")],
            y=MLtrainsmall$classe,plot="pairs") 
    # No good separation
featurePlot(x=MLtrainsmall[,c("accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Dancing slugs, but no good separation
featurePlot(x=MLtrainsmall[,c("max_roll_arm","max_pitch_arm","max_yaw_arm","min_roll_arm")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Some separation in all
featurePlot(x=MLtrainsmall[,c("min_pitch_arm","min_yaw_arm","amplitude_roll_arm","amplitude_pitch_arm")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Set of about 6 outliers (different group) in amplitude_pitch_arm
featurePlot(x=MLtrainsmall[,c("amplitude_yaw_arm","roll_dumbbell","pitch_dumbbell","yaw_dumbbell")],
            y=MLtrainsmall$classe,plot="pairs") 
    # amplitude_yaw_arm is good separation.
featurePlot(x=MLtrainsmall[,c("max_roll_dumbbell","min_roll_dumbbell","max_pitch_dumbbell","min_pitch_dumbbell")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Spread but not much separation
featurePlot(x=MLtrainsmall[,c("amplitude_roll_dumbbell","total_accel_dumbbell","amplitude_pitch_dumbbell","var_accel_dumbbell")],
            y=MLtrainsmall$classe,plot="pairs") 
    # Two outliers in var_accel_dumbbell, good spread
featurePlot(x=MLtrainsmall[,c("avg_roll_dumbbell","stddev_roll_dumbbell","var_roll_dumbbell","avg_pitch_dumbbell")],
            y=MLtrainsmall$classe,plot="pairs") 
    # combination of stddev_roll_dumbbell and var_roll_dumbbell spreads on a curve
featurePlot(x=MLtrainsmall[,c("stddev_pitch_dumbbell","var_pitch_dumbbell","avg_yaw_dumbbell","stddev_yaw_dumbbell")],
            y=MLtrainsmall$classe,plot="pairs") 
    # One outlier in var_pitch_dumbbell and one in stddev_yaw_dumbbell
featurePlot(x=MLtrainsmall[,c("gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x")],
            y=MLtrainsmall$classe,plot="pairs") 
    # some separation in gyros_dumbbell_x
featurePlot(x=MLtrainsmall[,c("accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y")],
            y=MLtrainsmall$classe,plot="pairs")
    # Some separation for blue in accel_dumbbell_z
featurePlot(x=MLtrainsmall[,c("magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm")],
            y=MLtrainsmall$classe,plot="pairs")
    # some separation in pitch_forearm, maybe some in magnet_dumbbell_z
featurePlot(x=MLtrainsmall[,c("max_roll_forearm","min_roll_forearm","max_pitch_forearm","min_pitch_forearm")],
            y=MLtrainsmall$classe,plot="pairs")
    # separation in all four variables
featurePlot(x=MLtrainsmall[,c("amplitude_roll_forearm","amplitude_pitch_forearm","total_accel_forearm","var_accel_forearm")],
            y=MLtrainsmall$classe,plot="pairs")
    # some separation in amplitude_roll_forearm
featurePlot(x=MLtrainsmall[,c("gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x")],
            y=MLtrainsmall$classe,plot="pairs")
    # very little separation
featurePlot(x=MLtrainsmall[,c("accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y")],
            y=MLtrainsmall$classe,plot="pairs")
    # No good separation
featurePlot(x=MLtrainsmall[,c("pitch_forearm","max_roll_forearm","amplitude_roll_forearm","magnet_forearm_z")],
            y=MLtrainsmall$classe,plot="pairs")
    # Little separation in magnet_forearm_z
featurePlot(x=MLtrainsmall[,c("pitch_forearm","gyros_dumbbell_x","amplitude_yaw_arm","yaw_arm","num_window","X")],
            y=MLtrainsmall$classe,plot="pairs")  
    # X and num_window are still the best separators


# Training with Linear Discriminant Analysis
    randforestfit<-train(classe~.,MLtrainsmall,method="lda",preProc=c("center","scale"))



