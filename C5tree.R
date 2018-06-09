#####################
#C5
#####################


library(MLmetrics)
library(C50)
library(dplyr)


Credit.train <- read.csv("Clean_Credit_Train.csv") # Import the clean train set.
summary(Credit.train)

# Train-test split
set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.8),replace = F) #80:20 Train-test split.
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID"))] #Keep all predictors except ID.
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID"))] # Keep all predictors except ID.
trainset$default <- as.factor(trainset$default)
C5.ctrl <- C5.0Control(subset=T, winnow = T,seed = 50) #subset- For the model to evaluate a group of discrete predictors for splits. winnow- Feature selection
C5.tree <- C5.0(default~.,data = trainset,trials = 35,control=C5.ctrl) #Run C5 tree model.
C5imp(C5.tree) #Variable importance in the model.
plot(C5.tree,trial = 35) #Plot C5 tree with 35 runs.
summary(C5.tree)
C5.pred <- predict(C5.tree,newdata = testset[,-which(names(testset) %in% c("default"))],type="class") #Predict using C5 tree model.
C5.pred
testset$default <- as.factor(testset$default)

library(caret)
confusionMatrix(C5.pred,testset$default,mode="everything",positive = levels(testset$default)[2]) #Kappa value - 0.4886, F1-score - 0.5907, False negative-539.

library(ROCR)
pr <- prediction(as.integer(C5.pred), as.integer(testset$default)) 
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance with true positive rate as accuracy measure.
plot(prf,colour="red") # Plot ROC curve.
auc <- performance(pr, measure = "auc") #Calculate performance with AUC as accuracy measure.
auc <- auc@y.values[[1]]
auc #Area under the curve 0.7208
