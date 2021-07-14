# Author: Lim Jia Qi
# Date: 27/05/2021
# Decision tree, random forest and gradient boosting

# Set up the working directory where the data file is located
data=read.csv('diabetes.csv')
vars=setdiff(colnames(data),"Outcome")

# for reproducibility
set.seed(123)
randn=runif(nrow(data))
train_idx=randn<=0.8
train=data[train_idx,]
test=data[!train_idx,]

# function to assess model performance
library(sigr)
performance=function(y,pred){
  confmat_test=table(truth=y,predict=pred>0.5)
  acc=sum(diag(confmat_test))/sum(confmat_test)
  precision=confmat_test[2,2]/sum(confmat_test[,2])
  recall=confmat_test[2,2]/sum(confmat_test[2,])
  auc=calcAUC(pred,y)
  c(acc,precision,recall,auc)
}

# Decision tree -----------------------------------------------------------

library(rpart)
library(rpart.plot)
library(wrapr)
# formula
data_formula=mk_formula("Outcome",vars)
treemodel=rpart(data_formula,train,method = "class")

# Plot
rpart.plot(treemodel,type=5,extra=2,cex=0.65)

# predict
pred=predict(treemodel,newdata = train)[,2]
perf_train=performance(train$Outcome,pred)
pred=predict(treemodel,newdata=test)[,2]
perf_test=performance(test$Outcome,pred)
perf=rbind(perf_train,perf_test)
colnames(perf)=c("accuracy","precision","recall","AUC")
perf


# random forest -----------------------------------------------------------

library(randomForest)
library(GA)
library(rsample)

#function of random forest hyperparameter optimization
# x1 refers to maxnodes, x2 refers to nodesize
rf_param_opt=function(x) {
  # further split the training dataset
  split=initial_split(train,prop = 0.8,strata = "Outcome")
  train_split=training(split)
  test_split=testing(split)
  # Use the train_split to train with different combination x1 and x2, then evaluate
  # model with test_split. Set the ntree to a large number. x1 and x2 must be integers
  model_rf=randomForest(train_split[,vars],y=as.factor(train_split$Outcome),ntree=1000,
                        maxnodes = floor(x[1]),nodesize=floor(x[2]))
  pred=predict(model_rf,test_split[,vars])
  # accuracy
  mean(test_split$Outcome==pred)
}

# GA parameter optimization
GA=ga(type="real-valued",fitness = rf_param_opt, lower = c(2,1),upper = c(50,10),
      popSize = 10, maxiter = 100,run=20,seed=1)
summary(GA)
plot(GA)

# model training with optimal meta-parameters
model_rf=randomForest(train[,vars],y=as.factor(train$Outcome),ntree=1000,importance = TRUE,
                      maxnodes = floor(GA@solution[1]),nodesize = floor(GA@solution[2]))

# model evaluation using test data
pred=predict(model_rf,train[,vars],type="prob")[,2]
perf_train=performance(train$Outcome,pred)
pred=predict(model_rf,test[,vars],type="prob")[,2]
perf_test=performance(test$Outcome,pred)
perf_rf=rbind(perf_train,perf_test)
colnames(perf_rf)=c("accuracy","precision","recall","AUC")
perf_rf

# variable importance plot
varImpPlot(model_rf,type=1)


# gradient boosting -------------------------------------------------------

library(xgboost)
input=as.matrix(train[,vars])
cv=xgb.cv(input,label=train$Outcome,params = list(objective="binary:logistic",eta=0.005,max_depth=3),
          nfold = 5,nrounds = 1000,print_every_n = 50,metrics = "logloss")
evalframe=as.data.frame(cv$evaluation_log)
(ntree=which.min(evalframe$test_logloss_mean))

# plot
library(ggplot2)
ggplot(evalframe,aes(x=iter,y=test_logloss_mean)) +
  geom_line() +
  geom_vline(xintercept = ntree, linetype=2, color="red")

# Use the optimal parameter settings to train the model
model=xgboost(data=input,label=train$Outcome,params = list(objective="binary:logistic",eta=0.005,max_depth=3),
              nrounds = ntree,verbose = FALSE)

pred=predict(model,input)
perf_train=performance(train$Outcome,pred)
test_input=as.matrix(test[,vars])
pred=predict(model,test_input)
perf_test=performance(test$Outcome,pred)
perf_gb=rbind(perf_train,perf_test)
colnames(perf_gb)=c("accuracy","precision","recall","AUC")
perf_gb

# feature importance plot
importance_matrix=xgb.importance(vars,model=model)
xgb.plot.importance(importance_matrix ,rel_to_first = TRUE)

# effects of important predictors
xgb.plot.shap(test_input, model = model, features = c("Glucose","BMI","Age"))


# performances of each classifier -----------------------------------------

perf_all=rbind(perf[2,],perf_rf[2,],perf_gb[2,])
rownames(perf_all)=c("Decision tree","Random forest","Gradient boosting")
library(knitr)
kable(perf_all)
