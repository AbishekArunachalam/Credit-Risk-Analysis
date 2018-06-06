library(dplyr)
library(xgboost)
library(caret)
library(MLmetrics)

Credit.train <- read.csv("AT3_credit_train_STUDENT.csv")

head(Credit.train)
str(Credit.train)#23101 obs. of  17 variables
summary(Credit.train) #One of the observations has maximum age of 141. There is a chance for data impurity.
sapply(Credit.train[,c(3,4,5,6,17)],function(x)unique(x)) #Find values in each column of a data frame.

#Clean LIMIT_BAL
#There is min value of -99 in LIMIT_BAL which is not possible.
Credit.train %>% filter(LIMIT_BAL < 0) #There are 50 rows. Lets remove them.
Credit.train <- Credit.train[Credit.train$LIMIT_BAL >0,]
Credit.train$LIMIT_BAL <- as.integer(Credit.train$LIMIT_BAL) #Convert LIMIT_BAL to integer.
summary(Credit.train)

#Clean SEX 
Credit.train %>% filter(SEX == "dolphin" | SEX== "cat" | SEX=="dog") #Check the number of rows containing undefined levels in SEX.
Credit.train <- Credit.train[Credit.train$SEX==1 | Credit.train$SEX==2, ] #Retain Sex=1 and Sex=2.
Credit.train$SEX <- factor(Credit.train$SEX)

#Clean EDUCATION
Credit.train$EDUCATION[Credit.train$EDUCATION==6 | Credit.train$EDUCATION==0] <- 5 # Levels 5 and 6 of education are unknown. Merge them as a single level.
Credit.train$EDUCATION <- as.factor(Credit.train$EDUCATION)

#Clean MARRIAGE
Credit.train$MARRIAGE[Credit.train$MARRIAGE==0] <- 3 #Set unknown level marriage 0 to 3-Other. 
Credit.train$MARRIAGE <- as.factor(Credit.train$MARRIAGE)

#Clean AGE
#The maximum age of observations in validation set is 79. Filter observations till age 79.
Credit.train %>% filter(AGE > 79) #There are 50 observations with abnormal ages. There are no observations for age between 79-128.
Credit.train <- Credit.train[Credit.train$AGE <=79,]

#Clean Default
Credit.train$default <- as.character(Credit.train$default)
Credit.train$default[Credit.train$default=='N'] <- 0 #Assign 0 for N
Credit.train$default[Credit.train$default=='Y'] <- 1 #Assign 1 for Y
Credit.train$default <- as.factor(Credit.train$default)

sapply(Credit.train[,c(3,4,5,6,17)],function(x)unique(x)) #Check the columns are in appropriate format.

sum(sapply(Credit.train[,c(3,4,5,6,17)],function(x)is.na(x))) #Check it any columns contain NA.

dim(Credit.train) #23000 observations and 17 variables.

write.csv(Credit.train,"Clean_Credit_Train.csv",row.names = F)

rm(list=ls())

# Credit.cor <- cor(Credit.train)
# corrplot::corrplot(Credit.cor)
# class(Credit.train$EDUCATION)

#Validation set
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")
summary(Credit.test)
str(Credit.test)

sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x)) #There is level 0 in Education and Marriage.

Credit.test$EDUCATION[Credit.test$EDUCATION==6 | Credit.test$EDUCATION==0] <- 5

Credit.test$MARRIAGE[Credit.test$MARRIAGE==0] <- 3
Credit.test <- Credit.test[,-which(names(Credit.test)%in%c("ID,SEX","MARRIAGE"))]
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
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE"))]
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID","SEX","MARRIAGE"))]
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

xgbGrid <- expand.grid(nrounds = c(500,800,1000,1200,1400),
                       max_depth = c(2,4,6,8,10,12,14),
                       colsample_bytree = seq(0.4,1,0.1,0.2),
                       ## The values below are default values in the sklearn-api.
                       eta = c(0.01,0.05,0.1,0.5,1),
                       gamma=1,
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
