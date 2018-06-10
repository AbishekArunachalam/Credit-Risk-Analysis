#################
#GBM
#################


library(gbm)
library(caret)
library(dplyr)

Credit.train <- read.csv("Clean_Credit_Train.csv")
summary(Credit.train)
set.seed(1)
#Stratified sampling
TrainDataIndex <- createDataPartition(Credit.train$default,p=0.8,list=F) #80:20 split
trainset <- Credit.train[TrainDataIndex,-which(names(Credit.train) %in% c("ID","SEX","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove insignificant columns.
testset <- Credit.train[-TrainDataIndex,-which(names(Credit.train) %in% c("ID","SEX","AMT_PC5","AMT_PC6","AMT_PC7"))]

# defining some parameters
gbm_depth <- c(4,6,8) #maximum nodes per tree
gbm_n.min <- c(2,4,6,8) #minimum number of observations in the trees terminal, important effect on overfitting
gbm_shrinkage <- c(0.001,0.01,0.1,0.5,1) #learning rate
cores_num <- 4 #number of cores
gbm_cv.folds=10 #number of cross-validation folds to perform
num_trees <- 38000 #number of trees to grow
#trainset$default <- as.numeric(trainset$default)
# fit initial model
gbm_fit <- gbm(default~.,
               data=trainset,
               distribution='bernoulli', #bernoulli for classification model
               n.trees=num_trees, #the number of GBM interaction
               interaction.depth= gbm_depth,
               n.minobsinnode = gbm_n.min,
               shrinkage=gbm_shrinkage,
               cv.folds=gbm_cv.folds,
               verbose = T, #print the output
               n.cores = cores_num
)

summary(gbm_fit)
best.iter <- gbm.perf(gbm_fit, method = "cv") #pick the best iter
prediction_gbm <- predict(gbm_fit, testset[,-which(names(testset) %in% c("default"))], n.trees = best.iter, type = "response") #predict with best iter
pred <- rep(0,length(prediction_gbm))
pred[prediction_gbm >= 0.5] <- 1
pred = as.factor(pred)
testset$default = as.factor(testset$default)
library(caret)
confusionMatrix(testset$default, pred,mode = "everything") #Confusion matrix

library(ROCR)
pr <- prediction(as.integer(pred), as.integer(testset$default)) 
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance.
plot(prf,col="red") # Plot ROC curve.
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #Area under the curve 0.7182396

gbm_fit1 <- gbm(default~., #train on entire Credit_train dataset
               data=Credit.train,
               distribution='bernoulli',
               n.trees=38000, #the number of GBM interaction
               interaction.depth= gbm_depth,
               n.minobsinnode = gbm_n.min,
               shrinkage=gbm_shrinkage,
               cv.folds=gbm_cv.folds,
               verbose = T, #print the preliminary output
               n.cores = cores_num
)
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv") #Read credit test.
summary(Credit.test)
str(Credit.test)

sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x)) #There is level 0 in Education and Marriage.

Credit.test$EDUCATION[Credit.test$EDUCATION==6 | Credit.test$EDUCATION==0] <- 5 #Replace level 6 and 0 with level 5.

Credit.test$MARRIAGE[Credit.test$MARRIAGE==0] <- 3 #Replace level 0 with level 3.
Credit.test <- Credit.test[,-which(names(Credit.test)%in%c("ID","SEX","AMT_PC5","AMT_PC6","AMT_PC7"))] 

valid_pred <- predict(gbm_fit, Credit.test, n.trees = best.iter, type = "response")
pred <- rep(1,length(valid_pred))
pred[valid_pred < 0.5] <- 0
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")
gbm_pred <- as.data.frame(cbind(Credit.test$ID,pred))
names(gbm_pred) <- c("ID","default")
write.csv(gbm_pred,"GBMfinal.csv",row.names = F)
