######################
#Random forest
######################

library(randomForest)

Credit.train <- read.csv("Clean_Credit_Train.csv")
# Train-test split
set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.8),replace = F) #80:20 Train-test split.

trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove ID column in trainset.
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove ID column in testset.

typeColNum <- grep("default",names(testset))
#Build random forest model
Credit.rf <- randomForest(default ~.,data = trainset,
                          ntree=1200,ir.measure.auc=T,type="classification")
#quantitative measure of variable importance
importance(Credit.rf)
#sorted plot of importance
dev.off()
varImpPlot(Credit.rf)


predicted <- predict(Credit.rf,newdata = testset[,-typeColNum], type="response")
pred <- rep(1,length(predicted))
pred[predicted <0.5] <- 0

testset$default <- as.integer(testset$default)
#accuracy for test set
mean(pred==testset$default)#0.82695

#confusion matrix
library(caret)
confusionMatrix(data = as.factor(pred),reference = as.factor(testset$default),mode = "everything") # Kappa-0.4584, F1 - 0.8923

library(ROCR)
pr <- prediction(pred, testset$default)
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance.
plot(prf) # Plot ROC curve.
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #Area under the curve 0.7017579

#Validation set
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")
summary(Credit.test)
str(Credit.test)

Credit.test$EDUCATION[Credit.test$EDUCATION==6 | Credit.test$EDUCATION==0] <- 5

Credit.test$MARRIAGE[Credit.test$MARRIAGE==0] <- 3
Credit.test <- Credit.test[,-which(names(Credit.test)%in%c("ID","SEX","MARRIAGE","AMT_PC5","AMT_PC6","AMT_PC7"))]

validation.predicted <- predict(Credit.rf,newdata = Credit.test,type = "response")
pred <- rep(0,length(validation.predicted))
pred[validation.predicted > 0.5] <- 1
unique(pred)
predictions <- as.data.frame(cbind(Credit.test$ID,pred))
colnames(predictions) <- c("ID","default")
write.csv(predictions,"RFPredictions.csv",row.names = F)