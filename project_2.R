# Author: Lim Jia Qi
# Date: 27/05/2021
# predict outcome of diabetes in subjects surveyed (>=21 years old of Pima Indian
# heritage) using GAM

# load data
data=read.csv('diabetes.csv')
# Split data into train and test
set.seed(123)
randn=runif(nrow(data))
train_idx=randn<=0.8
train=data[train_idx,]
test=data[!train_idx,]


# gam model training ------------------------------------------------------

form_gam=as.formula("Outcome==1~s(Pregnancies)+s(Glucose)+s(BloodPressure)+
                    s(SkinThickness)+s(Insulin)+s(BMI)+s(DiabetesPedigreeFunction)+
                    s(Age)")

library(mgcv)
gam_model=gam(form_gam,data=train,family = binomial(link="logit"))
gam_model$converged
summary(gam_model)


# visualization of effect of individual predictors on response ------------

plot(gam_model)

# lets try using ggplot
library(ggplot2)
terms=predict(gam_model,type="terms")
terms=cbind(Outcome=train$Outcome,terms)

# data frame
frame1=as.data.frame(terms)
colnames(frame1)=gsub('[()]','',colnames(frame1))
vars=setdiff(colnames(train),"Outcome")
#vars=setdiff(colnames(frame1),"Outcome")
colnames(frame1)[-1]=sub(".","",colnames(frame1)[-1])
library(cdata)
frame1_long=unpivot_to_blocks(frame1,nameForNewKeyColumn = "basis_function",
                              nameForNewValueColumn = "basis_value",
                              columnsToTakeFrom = vars)

data_long=unpivot_to_blocks(train,nameForNewKeyColumn = "basis_function",
                            nameForNewValueColumn = "value",
                            columnsToTakeFrom = vars)
frame1_long=cbind(frame1_long,value=data_long$value)

ggplot(data=frame1_long,aes(x=value,y=basis_value)) +
  geom_smooth() +
  facet_wrap(~basis_function,ncol = 3,scales = "free")


# model evaluation --------------------------------------------------------

# prediction
train$pred=predict(gam_model,newdata = train,type = "response")
test$pred=predict(gam_model,newdata=test,type="response")

library(sigr)
# function to compute model performance
performance=function(y,pred){
  confmat_test=table(truth=y,predict=pred>0.5)
  acc=sum(diag(confmat_test))/sum(confmat_test)
  precision=confmat_test[2,2]/sum(confmat_test[,2])
  recall=confmat_test[2,2]/sum(confmat_test[2,])
  auc=calcAUC(pred,y)
  c(acc,precision,recall,auc)
}

# Posterior probability
train$pred=predict(gam_model,newdata = train,type = "response")
test$pred=predict(gam_model,newdata=test,type="response")

# model performance evaluated using training data
perf_train=performance(train$Outcome,train$pred)
perf_test=performance(test$Outcome,test$pred)
perf_mat=rbind(perf_train,perf_test)
colnames(perf_mat)=c("accuracy","precision","recall","AUC")
round(perf_mat,4)
