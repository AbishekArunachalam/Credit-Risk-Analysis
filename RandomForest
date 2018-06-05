#Random forest


library(randomForest)
set.seed(42)

typeColNum <- grep("default",names(testset))
#Build random forest model
Credit.rf <- randomForest(default ~.,data = trainset,
                          ntree=1200,ir.measure.auc=T,type="classification")
predicted <- predict(Credit.rf,newdata = testset[,-typeColNum], type="response")
#model summary
summary(Credit.rf)
#variables contained in model 
names(Credit.rf)

pred <- rep(0,length(predicted))
pred[predicted > 0.5] <- 1
unique(pred)

#accuracy for test set
mean(pred==as.numeric(testset$default))#0.8252174
#confusion matrix
library(pROC)
auc(as.numeric(pred),as.numeric(testset$default))#Area under the curve: 0.7803 0.7785



#Area under the curve: 0.7743 - Random forest
#quantitative measure of variable importance
importance(Credit.rf)
#sorted plot of importance
varImpPlot(Credit.rf)

pred1 <- predict(Credit.rf,newdata = Credit.test,type = "response")
pred <- as.data.frame(cbind(Credit.test$ID,as.character(pred1)))
colnames(pred) <- c("ID","default")
write.csv(pred,"RF8Predictions.csv",row.names = F)
