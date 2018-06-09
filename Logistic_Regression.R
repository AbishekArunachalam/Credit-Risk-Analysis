########################
#Logistic Regression
########################

set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.8),replace = F) #80:20 train-test split
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID"))] #Use all columns except ID for train set.
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID"))] #Use all columns except ID for test set.
glm.fit <- glm(default~.,data = trainset,family = binomial) # Run logistic regression with all the predictors.
anova(glm.fit,test = "Chisq") #Analysis of variance of predictors.
summary(glm.fit) #Some predictors are not significant.

library(leaps)
regfit.full <- regsubsets(default~.,data = trainset,nvmax=30,nbest = 1,method = "exhaustive") #Pick the best subset of predictors.
reg.summary<-summary(regfit.full)
names(reg.summary)
plot.new()
par(mfrow=c(2,2)) #Divide the plot area into 4 cross-sections.

rsq.max <- which.max(reg.summary$rsq) #Plot for max R-squared.
plot(reg.summary$rsq ,xlab="Number of Variables ",ylab="Rsq", type='l')
points(rsq.max,reg.summary$rsq[rsq.max],col="red",cex=2,pch=20)

cp.min = which.min(reg.summary$cp ) #Plot for min Cp.
plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp", type='l')
points(cp.min,reg.summary$cp[cp.min],col="red",cex=2,pch=20)

bic.min <- which.min(reg.summary$bic ) #Plot for min BIC.
plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",type='l')
points(bic.min,reg.summary$bic[bic.min],col="red",cex=2,pch=20)

#Model with the 8 predictors.
glm.fit1 <- glm(default~LIMIT_BAL+EDUCATION+AGE+PAY_PC1+PAY_PC2+PAY_PC3+AMT_PC1+AMT_PC2,data = trainset,family = "binomial")

glm.pred <- predict(glm.fit1,newdata = testset[,-which(names(testset) %in% c("default"))],type="response") #Predict using the model.
pred <- rep(1,length(glm.pred)) #Repeat 1 for the length of glm.pred.
pred[glm.pred < 0.5] <- 0 #Replace 1 with 0 when prob is > 0.5.
pred <- as.factor(pred)
testset$default <- as.factor(testset$default)

library(caret)
confusionMatrix(pred,testset$default,mode ="everything",positive = levels(testset$default)[2]) #There are 833 false negative and 120 false positive.

library(ROCR)
pr <- prediction(glm.pred, testset$default)
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #Calculate performance.
plot(prf,colour="red") # Plot ROC curve.
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #Area 0.726956
