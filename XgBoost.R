#Validation set
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")
summary(Credit.test)
str(Credit.test)

sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x)) #There is level 0 in Education and Marriage.

Credit.test$EDUCATION[Credit.test$EDUCATION==6 | Credit.test$EDUCATION==0] <- 5

Credit.test$MARRIAGE[Credit.test$MARRIAGE==0] <- 3
Credit.test <- Credit.test[,-which(names(Credit.test)%in%c("ID","SEX","MARRIAGE","EDUCATION","PAY_PC2","PAY_PC3"))]
# Credit.test$EDUCATION <- as.factor(Credit.test$EDUCATION)
# Credit.test$MARRIAGE <- as.factor(Credit.test$MARRIAGE)
# Credit.test$SEX <- as.factor(Credit.test$SEX)

sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x))

Credit.train <- read.csv("Clean_Credit_Train.csv")

# Credit.train$EDUCATION <- as.factor(Credit.train$EDUCATION)
# Credit.train$MARRIAGE <- as.factor(Credit.train$MARRIAGE)
# Credit.train$SEX <- as.factor(Credit.train$SEX)
# Credit.train$default <- as.factor(Credit.train$default)

# Train-test split
set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.8),replace = F)
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","EDUCATION","PAY_PC2","PAY_PC3"))]
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE","EDUCATION","PAY_PC2","PAY_PC3"))]

#XgBOOST
set.seed(2)
X_train <- xgb.DMatrix(as.matrix(trainset %>% select(-default)))
Y_train <- as.factor(trainset$default)
levels(Y_train) <- c("No","Yes")
X_test <- xgb.DMatrix(as.matrix(testset %>% select(-default)))
Y_test <- testset$default


xgb_trcontrol <- trainControl(
  method = "repeatedcv",
  repeats = 1,
  number = 10,
  classProbs = TRUE,
  allowParallel = TRUE
)

xgbGrid <- expand.grid(nrounds = c(500,800,1000,1200),
                       max_depth = c(2,4,6,8,10,14),
                       colsample_bytree = seq(0.4,1,0.1),
                       ## The values below are default values in the sklearn-api.
                       eta = c(0.01,0.05,0.1,0.5),
                       gamma=0,
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
pred <- rep(0,length(predicted_values))
pred[predicted_values >0.5] <- 1
pred[predicted_values <0.5] <- 0
levels(pred)
confusionMatrix(data = as.factor(pred),reference= as.factor(Y_test),mode = "everything")
library(pROC)
AUC(pred,Y_test)
Out_pred <- predict(xgb_model,newdata = Credit.test,type = "raw")
pred1 <- rep(0,length(Out_pred))
pred1[Out_pred >0.5] <- 1
pred1[Out_pred <0.5] <- 0
pred1
pred <- cbind(Credit.test$ID,pred1)
