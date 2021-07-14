# project 6: credit risk assessment by applying SVM
# Reference: Grömping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set. Report 4/2019, 
#            Reports in Mathematics, Physics and Chemistry, Department II, Beuth University of Applied Sciences Berlin.

# Run the read_SouthGermanCredit.R file and save the 'dat' variable in .RDATA format
load("credit.RData")
str(dat)
summary(dat)
# no missing values
response='credit_risk'
library(rsample)
set.seed(10)
split=initial_split(dat,prop=0.8,strata = response)
train=training(split)
test=testing(split)

# Data visualization -----------------------------------------------

# Based on the output of str() function,
# we know that there are 17 categorical predictors and 3 continuos predictors. There are 
# 700 good credit and 300 bad credits. From the paper, it is stated that the bad credit instances
# are oversampled. Normally, the prevalence of bad credit instances is approximately 5%

# Continuos predictors
idx_con=c("duration","amount","age")
# Grouped boxplot. Turn train data from wide to long format
library(cdata)
train_con=cbind(train[,idx_con],credit_risk=train[,response])
train_con_long=unpivot_to_blocks(train_con,nameForNewKeyColumn = "variables",
                                 nameForNewValueColumn = "values",
                                 columnsToTakeFrom = idx_con)

library(ggplot2)
ggplot(data=train_con_long, aes(x=credit_risk,y=values)) +
  geom_boxplot(color="blue",fill="blue",alpha=0.2,notch=TRUE,
               outlier.color="red",outlier.fill = "red",outlier.size = 2) +
  facet_wrap(~variables,ncol=3,scales = "free")

# grouped scatter plot matrix
library(GGally)
# ggpairs(train_con[,idx_con],title = "scatter plot matrix")
ggpairs(train_con,columns = 1:3, ggplot2::aes(colour=credit_risk))


# Visualization of categorical predictors ---------------------------------

# categorical predictors. Employ chi-square test to investigate the association between
# categorical features and response/outcome
train_cat=train[,!(colnames(train) %in% idx_con)]
# (tt=chisq.test(train$status,train$credit_risk))

t=c()
idx=c()
for (i in (1:(ncol(train_cat)-1))) {
  t[i]=chisq.test(train[,i],train$credit_risk)$p.value
  # u[i]=fisher.test(train[,i],train$credit_risk)$p.value
  if (!is.list(tryCatch( { result <- chisq.test(train[,i],train$credit_risk) }
                         , warning = function(w) { print("TRUE") }))) {
    idx=c(idx,i)
  }
}

idx_sig=which(t<=0.05)

idx_int=!(idx_sig %in% idx)
colnames(train_cat)[idx_sig[idx_int]]
# Unsure of 'credit history' and 'savings' association with response variable

# Visualization of two categorical variables: method 1
ggplot(data=train) +
  geom_count(aes(x=housing,y=credit_risk))

# method 2
library(magrittr)
library(dplyr)
train %>%
  count(credit_history,credit_risk) %>%
  ggplot(aes(x=credit_history,y=credit_risk)) +
  geom_tile(aes(fill=n)) +
  theme(axis.text.x = element_text(angle = 45,hjust=1))

# contigency table (count)
table(train$housing,train$credit_risk)
# contingency table (probability)
round(prop.table(table(train$housing,train$credit_risk)),4)
# contingency table (marginal probability)
prop.table(table(train$housing,train$credit_risk),1)


# Feature engineering ----------------------------------------------------------

# Perform one hot encoding for categorical (nominal) predictors.
library(caret)
var_cat=c("status","credit_history","purpose","savings","employment_duration",
          "personal_status_sex","other_debtors","property","other_installment_plans",
          "housing","job","people_liable","telephone","foreign_worker")

train_cat=train[,colnames(train) %in% var_cat]
dummy=dummyVars("~.",data=train_cat)
newdata=data.frame(predict(dummy,newdata=train_cat))

# label encoding for ordinal variables
# chooses the related variables
var_ord=c("installment_rate","present_residence","number_credits")
train_cont=train[,colnames(train) %in% var_ord]
train_cont=transform(train_cont,installment_rate=as.numeric(installment_rate)-1,
                     present_residence=as.numeric(present_residence)-1,
                     number_credits=as.numeric(number_credits)-1)

# Min-max normalization of continuos predictors
var_cont=c("amount","age","duration")
dat_cont=train[,colnames(train) %in% var_cont]
process=preProcess(dat_cont,method = c("range"))
scaled_dat_cont=predict(process,dat_cont)

# concatenate all the predictors with response variable by columns
train_new=cbind(newdata,train_cont,
                scaled_dat_cont,credit_risk=train$credit_risk)


# Test (hold-out) data preparation ----------------------------------------

# Be mindful of data leakage. Repeat the same steps as what has been carried out 
# on training dataset, but we can preprocess the test data using the information
# derived from training data.

# One-hot encoding
test_cat=test[,colnames(test) %in% var_cat]
newdata=data.frame(predict(dummy,newdata=test_cat))

# Label encoding
test_cont=test[,colnames(test) %in% var_ord]
test_cont=transform(test_cont,installment_rate=as.numeric(installment_rate)-1,
                    present_residence=as.numeric(present_residence)-1,
                    number_credits=as.numeric(number_credits)-1)
# Min-max normalization
dat_cont=test[,colnames(test) %in% var_cont]
scaled_dat_cont=predict(process,dat_cont)

# concatenate by columns
test_new=cbind(newdata,test_cont,
               scaled_dat_cont,credit_risk=test$credit_risk)

# Training using SVM ------------------------------------------------------

library(e1071)

# Hyper-parameters tuning cost function
cost_matrix=matrix(c(0,1,5,0),ncol=2)
err=function(truth,pred){
  t=table(truth=truth,pred=pred)
  tot_cost=sum(t*cost_matrix)
  tot_cost
}

range_exp=seq(-10,10,by=2)
set.seed(200)  # for reproducibility
# linear kernel SVM. No scaling is needed as it had been performed beforehand.
# class weight is set to be inversely proportional to the number of samples in 
# each class
svm_tune=tune(svm,credit_risk~.,data = train_new,kernel='linear', scale=FALSE,
              probability=TRUE, class.weights='inverse',
              ranges = list(cost=c(2^range_exp)),
              tunecontrol = tune.control(cross=5,error.fun = err))
summary(svm_tune)
min_cost=svm_tune$performances$cost[which.min(svm_tune$performances$error)]

# Visualization 
svm_tune$performances %>%
  ggplot(aes(x=cost,y=error)) +
  geom_line() +
  scale_x_continuous(name = "cost, C",trans = "log2") + 
  ylab("misclassification cost") +
  geom_vline(xintercept = min_cost,
             color="red",linetype=2)

# Extract the best model in term of misclassification cost
svm_lin=svm_tune$best.model

# Coefficients of linear SVM can give insight on how individual predictor affects
# the outcome
coef_lin=data.frame(names=names(coef(svm_lin))[-1],coef=coef(svm_lin)[-1])
coef_lin_10=coef_lin[order(-abs(coef_lin$coef))[1:10],]
rownames(coef_lin_10)=NULL
library(knitr)
kable(coef_lin_10)

# Visualization of coefficients estimates
ggplot(data=coef_lin_10,aes(x=names,y=coef)) +
  geom_pointrange(aes(ymin=0,ymax=coef)) +
  coord_flip() +theme_classic() + ylab("coefficient estimates")

data.frame(fitted=svm_lin$fitted[1:10],dv=svm_lin$decision.values[1:10])


# RBF kernel SVM ----------------------------------------------------------

range_exp_sigma=seq(-5,5,by=2)
set.seed(300)
svm_tune=tune(svm,credit_risk~.,data = train_new,kernel='radial', scale=FALSE,
              probability=TRUE, class.weights='inverse',
              ranges = list(cost=c(2^range_exp),gamma=c(2^range_exp_sigma)),
              tunecontrol = tune.control(cross=5,error.fun = err))
summary(svm_tune)
# svm_rbf=svm_tune$best.model

# Best parameters from grid search
idx_min=which.min(svm_tune$performances[,'error'])
# countour plot
ggplot(svm_tune$performances,aes(x=cost,y=gamma)) +
  geom_raster(aes(fill=error)) +
  geom_contour(aes(z=error),color='white') +
  scale_x_continuous(name='cost',trans = "log2") +
  scale_y_continuous(name='gamma',trans = "log2") +
  geom_point(aes(x=cost[idx_min],y=gamma[idx_min]),shape=19,color='red',size=2) +
  geom_text(data=svm_tune$performances[idx_min,],
            aes(x=cost,y=1.05*gamma, color='yellow',
                label=sprintf("cost of misclassification: %.2f",error)),
            show.legend = FALSE)

# fine grid search
range_exp=seq(4,10,by=1)
range_exp_gamma=seq(-6,-2,by=1)
set.seed(400)
svm_tune=tune(svm,credit_risk~.,data = train_new,kernel='radial', scale=FALSE,
              probability=TRUE, class.weights='inverse',
              ranges = list(cost=c(2^range_exp),gamma=c(2^range_exp_gamma)),
              tunecontrol = tune.control(cross=5,error.fun = err))
summary(svm_tune)
svm_rbf=svm_tune$best.model

idx_min=which.min(svm_tune$performances[,'error'])
# contour plot
ggplot(svm_tune$performances,aes(x=cost,y=gamma)) +
  geom_raster(aes(fill=error)) +
  geom_contour(aes(z=error),color='white') +
  scale_x_continuous(name='cost',trans = "log2") +
  scale_y_continuous(name='gamma',trans = "log2") +
  geom_point(aes(x=cost[idx_min],y=gamma[idx_min]),shape=19,color='red',size=2) +
  geom_text(data=svm_tune$performances[idx_min,],
            aes(x=cost,y=1.05*gamma, color='yellow',
                label=sprintf("cost of misclassification: %.2f",error)),
            show.legend = FALSE)


# double density plot and ROC curve ---------------------------------------

# Function to evaluate the performance of binary classifiers
library(sigr)
performance=function(truth,pred){
  tab=table(truth=truth,prediction=pred)
  acc=sum(diag(tab))/sum(tab)
  cost=sum(cost_matrix*tab)
  recall=tab[1,1]/sum(tab[1,])
  precision=tab[1,1]/sum(tab[,1])
  F1=(2*precision*recall)/(precision+recall)
  truth_logical=as.numeric(truth)-1
  AUC=calcAUC(attr(pred,"decision.values"),truth_logical,yTarget = FALSE)
  round(c(accuracy=acc,misclass_cost=cost,recall=recall,precision=precision,
          f1_measure=F1,AUC=AUC),4)
}

pred_svm_lin=predict(svm_lin,newdata=test_new,decision.values = TRUE)
pred_svm_rbf=predict(svm_rbf,newdata = test_new,decision.values = TRUE)
dat_plot=data.frame(outcome=test_new$credit_risk,dv_svm_linear=attr(pred_svm_lin,"decision.values")[1:nrow(test_new)],
                    dv_svm_rbf=attr(pred_svm_rbf,"decision.values")[1:nrow(test_new)])

library(WVPlots)
DoubleDensityPlot(dat_plot,xvar="dv_svm_linear",truthVar = "outcome",
                  title="Distribution of linear svm scores (test data)") +
  geom_vline(xintercept = 0, color="red", linetype=2)

DoubleDensityPlot(dat_plot,xvar="dv_svm_rbf",truthVar = "outcome",
                  title="Distribution of RBF svm scores (test data)") +
  geom_vline(xintercept = 0, color="red", linetype=2)

ROCPlotPair(dat_plot,xvar1="dv_svm_linear",xvar2 = "dv_svm_rbf",truthVar = "outcome",
            truthTarget = "bad", title="ROC plots for svm models (test data)")


# Tabulation of classfiers' performance -----------------------------------

perf_svm_lin=performance(test_new$credit_risk,pred_svm_lin)
perf_svm_rbf=performance(test_new$credit_risk,pred_svm_rbf)
tab_perf=rbind(perf_svm_lin,perf_svm_rbf)
rownames(tab_perf)=c("linear svm","rbf svm")
kable(tab_perf)

# Confusion matrix for linear svm
table(test_new$credit_risk,pred_svm_lin)
# Confusion matrix for RBF svm
table(test_new$credit_risk,pred_svm_rbf)

# R version
##platform       x86_64-w64-mingw32          
##arch           x86_64                      
##os             mingw32                     
##system         x86_64, mingw32             
##status                                     
##major          4                           
##minor          0.2                         
##year           2020                        
##month          06                          
##day            22                          
##svn rev        78730                       
##language       R                           
##version.string R version 4.0.2 (2020-06-22)
