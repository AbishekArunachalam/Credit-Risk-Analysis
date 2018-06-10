###################
#SVM 
###################

library(doParallel)
library(caret)
library(ROCR)

Credit.train <- read.csv("Clean_Credit_Train.csv") #Load data
summary(Credit.train)

#SVM - Polynomial kernel

#Stratified sampling
set.seed(1)
TrainDataIndex <- createDataPartition(Credit.train$default,p=0.8,list=F) #80:20 train-test split
trainset <- Credit.train[TrainDataIndex,-which(names(Credit.train) %in% c("ID","SEX","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove insignificant varaibles in train set.
testset <- Credit.train[-TrainDataIndex,-which(names(Credit.train) %in% c("ID","SEX","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove insignificant variables in test set.
trainset$default <- as.factor(trainset$default) #Convert default to factor.
levels(trainset$default) <- c("No","Yes")

set.seed(2)
Cv_folds <- createMultiFolds(trainset,times = 5, k=10) #Create 10 folds 5 times.

cl <- makeCluster(detectCores()) #Detect cores for parallel processing.
registerDoParallel(cl) #Register cores.

ctrl <- trainControl(method = "repeatedcv",index = Cv_folds, classProb=TRUE ,allowParallel = TRUE) #Set model training parameters.
set.seed(123)
svm_mod <- train(default~., data=trainset, method = "svmPoly", trControl = ctrl, preProcess = c("center", "scale"),
                 tuneLength = 5) #Train svm polynomial model
plot(svm_mod) #plot model.
svm_pred <- predict(svm_mod,newdata=testset[,-which(names(testset) %in% c("default"))],response = "response") #Predict model.
stopCluster(cl) #Stop cluster.
levels(svm_pred) <- c(0,1)
testset$default <- as.factor(testset$default)

confusionMatrix(data = svm_pred,reference = testset$default,mode="everything") #Confusion matrix F1 -0.8770, Kappa- 0.3687

pr <- prediction(as.integer(svm_pred), as.integer(testset$default))
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance.
plot(prf,col="red") # Plot ROC curve.
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #Area under the curve 0.726956


#SVM- Radial Kernel

#Stratified sampling
set.seed(1)
TrainDataIndex <- createDataPartition(Credit.train$default,p=0.8,list=F) #80:20 train-test split
trainset <- Credit.train[TrainDataIndex,-which(names(Credit.train) %in% c("ID","SEX","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove insignificant varaibles in train set.
testset <- Credit.train[-TrainDataIndex,-which(names(Credit.train) %in% c("ID","SEX","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove insignificant variables in test set.
trainset$default <- as.factor(trainset$default) #Convert default to factor.
levels(trainset$default) <- c("No","Yes")

set.seed(2)
Cv_folds <- createMultiFolds(trainset,times = 5, k=10) #Create 10 folds 5 times.
cl <- makeCluster(detectCores()) #Detect cores for parallel processing.
registerDoParallel(cl) #Register cores.


ctrl <- trainControl(method = "repeatedcv",index = Cv_folds, classProb=TRUE ,allowParallel = TRUE) #Set model training parameters.
set.seed(123)
svm_mod <- train(default~., data=trainset, method = "svmRadial", trControl = ctrl, preProcess = c("center", "scale"), metric="Accuracy",
                 tuneLength = 10)
plot(svm_mod) #plot model.
svm_pred <- predict(svm_mod,newdata=testset[,-which(names(testset) %in% c("default"))],response = "response") #Predict model.
stopCluster(cl) #Stop cluster.
levels(svm_pred) <- c(0,1) #convert predictions to 1 and 0.
testset$default <- as.factor(testset$default) #convert to factor.

confusionMatrix(data = svm_pred,reference = testset$default,mode="everything") #Confusion matrix F1 -0.8801, Kappa- 0.3195

pr <- prediction(as.integer(svm_pred), as.integer(testset$default))
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance.
plot(prf,col="red") # Plot ROC curve.
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #Area under the curve - 0.628659



