############
#XgBoost
############

library(caret)
Credit.train <- read.csv("Clean_Credit_Train.csv")
summary(Credit.train)
# Train-test split
set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.8),replace = F) #80:20 split
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","AMT_PC5","AMT_PC6","AMT_PC7"))] #Remove unwanted predictors.
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","AMT_PC5","AMT_PC6","AMT_PC7"))]


#XgBOOST
library(doParallel)
cl <- makeCluster(detectCores()) #detect cores
registerDoParallel(cl)
set.seed(2)
X_train <- xgb.DMatrix(as.matrix(trainset %>% select(-default))) #Generate Xgb matrix for trainset.
Y_train <- as.factor(trainset$default)
levels(Y_train) <- c("No","Yes")
X_test <- xgb.DMatrix(as.matrix(testset %>% select(-default)))#Generate Xgb matrix for testset.
Y_test <- testset$default
master_data_x <- xgb.DMatrix(as.matrix(Credit.train %>% select(-default)))#Generate Xgb matrix for entire credit_train,
master_data_y <- as.factor(Credit.train$default)
xgb_trcontrol <- trainControl(
  method = "repeatedcv",
  repeats = 1,
  number = 10,
  classProbs = TRUE,
  allowParallel = TRUE
) #Train control parameters

xgbGrid <- expand.grid(nrounds = c(1000,1050,1100), #The number of iterations the model runs
                       max_depth = 4, #Depth of the trees to grow
                       colsample_bytree = 0.4,
                       ## The values below are default values in the sklearn-api.
                       eta = c(0.009,0.006), #Shrinkage parameter
                       gamma= 0, #Minimum loss reduction
                       min_child_weight = 1,
                       subsample = 1
)

set.seed(5) 
xgb_model <- train(
  X_train, Y_train,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  # tuneLength = 5,
  method = "xgbTree",
  metric = "Kappa",
  verbose=T,
  nthread=3
)
print(xgb_model)
plot(xgb_model)
predicted_values <- predict(xgb_model,newdata = X_test,type = "raw")
Y_test <- as.factor(Y_test)
levels(Y_test) <- c("No","Yes")

library(caret)
confusionMatrix(predicted_values,Y_test,mode="everything") #Kappa value - 0.5007, F1-score - 0.8990, False negative-545.

library(ROCR)
levels(Y_test) <- c(0,1)
levels(predicted_values) <- c(0,1)
pr <- prediction(as.integer(predicted_values), as.integer(testset$default)) 
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance with true positive rate as accuracy measure.
plot(prf,colour="red") # Plot ROC curve.
auc <- performance(pr, measure = "auc") #Calculate performance with AUC as accuracy measure.
auc <- auc@y.values[[1]]
auc #Area under the curve 0.72311

xgb_model1 <- train(
  master_data_x, master_data_y,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  # tuneLength = 5,
  method = "xgbTree",
  metric = "Kappa",
  verbose=T,
  nthread=3
)

#Validation set
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")
summary(Credit.test)
str(Credit.test)

sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x)) #There is level 0 in Education and Marriage.

Credit.test$EDUCATION[Credit.test$EDUCATION==6 | Credit.test$EDUCATION==0] <- 5

Credit.test$MARRIAGE[Credit.test$MARRIAGE==0] <- 3
Credit.test <- Credit.test[,-which(names(Credit.test)%in%c("ID","SEX","MARRIAGE","AMT_PC5","AMT_PC6","AMT_PC7"))]

sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x))

Credit_test <- xgb.DMatrix(as.matrix(Credit.test))
Out_pred <- predict(xgb_model1,newdata = Credit_test,type = "raw")
pred1 <- rep(0,length(Out_pred))
pred1[Out_pred >0.5] <- 1
pred1[Out_pred <0.5] <- 0
pred1
pred <- cbind(Credit.test$ID,pred1)
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")
colnames(pred) <- c("ID","default")
write.csv(pred,"Xg1BoostPrediction.csv",row.names = F)

