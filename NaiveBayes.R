###################
#Naive Baye's
###################

library(e1071)
# Train-test split

Credit.train <- read.csv("Clean_Credit_Train.csv")
summary(Credit.train)
# Train-test split
set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.8),replace = F) #80:20 Train-test split.

trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID"))] #Remove ID column in trainset.
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID"))] #Remove ID column in testset.
trainset$default <- as.factor(trainset$default)
naive.mod <- naiveBayes(default~.,data = trainset) #Run Naive Bayes model on the whole dataset.
naive.pred <- predict(naive.mod,newdata = testset[,-which(names(testset) %in% c("default"))],type = "class") #Predict using the model.
testset$default <- as.factor(testset$default)

library(caret)
confusionMatrix(naive.pred,testset$default) #The model performs poorly with all the predictors. Kappa value is 0.1402 and Accuracy is 0.4854.

library(Boruta) #Load Boruta package.
boruta_train <- Boruta(default~.,Credit.train)
print(boruta_train) #Print the import predictors.
final.boruta <- TentativeRoughFix(boruta_train) #Reject unimportant predictors.
boruta.df <- attStats(final.boruta) #Predictor importance stats.
print(boruta.df) #Print the predictor importance scores.

#Model with predictors that have MeanImportance >30.
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","AMT_PC3","AMT_PC4","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove unimportant predictors and create trainset.
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","AMT_PC3","AMT_PC4","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove unimportant predictors and create testset.
trainset$default <- as.factor(trainset$default)
naive.mod <- naiveBayes(default~.,data = trainset) #Naive Bayes model.
naive.pred <- predict(naive.mod,newdata = testset[,-which(names(testset) %in% c("default"))],type = "class") #Predict using the model.
testset$default <- as.factor(testset$default)
levels(testset$default)

library(caret)
confusionMatrix(naive.pred,testset$default) #The model has improved. Kappa score is 0.3817 and Accuracy is 0.778.


library(ROCR)
pr <- prediction(as.integer(naive.pred), as.integer(testset$default)) 
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance.
plot(prf,colour="red") # Plot ROC curve.
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #Area under the curve 0.6887759

