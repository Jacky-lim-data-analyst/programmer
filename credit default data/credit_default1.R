library(readxl)   # read excel xlsx file

setwd("~/machine learning/test_2")
# load the data with headers as column names
dat = read_excel('default of credit card clients.xls', sheet="Data", range = "B2:Y30002")

t = dat
colnames(t)[dim(t)[2]]= c("default")

# Change categorical features to factor
t$default = factor(t$default)
# encode sex
t$SEX = ifelse(dat$SEX==2, 'F', 'M')
# encode marriage
t$MARRIAGE = ifelse(dat$MARRIAGE==1,'married',ifelse(dat$MARRIAGE==2,'single','others'))
# encode education
t$EDUCATION = ifelse(dat$EDUCATION==1, 'grad_school',
                     ifelse(dat$EDUCATION==2, 'university', 
                            ifelse(dat$EDUCATION==3, 'high_school',
                                   ifelse(dat$EDUCATION==4, 'others', 'unknown'
                                   ))))

features_cat = c(colnames(t)[2:4], colnames(t)[6:11])

library(dplyr)
for (i in features_cat[1:3]){
  t[i] = factor(pull(t[i]))
}

# Data partition ----------------------------------------------------------

library(rsample)

response = "default"
set.seed(100)
split = initial_split(t, prop = 0.8, strata = response)

train = training(split)
test = testing(split)

# Preprocessing -----------------------------------------------------------

# Select the continuous predictors
features_cont = colnames(train)[c(1, 5, c(12:23))]
train_cont = train[,features_cont]

# Min-max normalization
library(caret)
preprocess_min_max = preProcess(train_cont, method = c("range"))
scaled_train_cont = predict(preprocess_min_max, train_cont)

train_scaled = cbind(scaled_train_cont, train[, features_cat], default = train$default)

m_train = dim(train_scaled)[1]
# Function to encode montly payment status: one-hot encoding and ordinal encoding
my_cat_encode = function(predictor, ncol, col_names, m){
  # data is the predictor (data frame); ncol is the number of columns
  # m = dim(predictor)[1]; col_names is the names of column header
  mat = matrix(, nrow = m, ncol = ncol)
  # Custom encoding: one hot encoding + ordinal encoding
  for (i in (1:m)){
    data = predictor[i]
    if (data == -2){
      vec = c(1,0,0,0)
    } else if (data == -1){
      vec = c(0,1,0,0)
    } else if (data == 0){
      vec = c(0,0,1,0)
    } else {
      vec = c(0,0,0,data)
    }
    mat[i,] = vec
  }
  df_ps0 = data.frame(mat)
  # colnames(df_ps0) = c("neg2_pay0", "neg1_pay0", "zero_pay0","delay_months")
  colnames(df_ps0) = col_names
  df_ps0
}

library(glue)
df = data.frame(matrix(,nrow = m_train, ncol = 0))
f_pay = c("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")

for (j in f_pay){
  predictor = train_scaled[,j]
  col_names = sapply(c("neg2_{j}", "duly_{j}", "zero_{j}", "delay_months_{j}"), glue)
  names(col_names) = NULL
  df1 = my_cat_encode(predictor, ncol = 4, col_names = col_names, m = m_train)
  df = cbind(df, df1)
}

# concatenate original dataframe
train_new = cbind(train_scaled, df)
train_new = train_new[, !(colnames(train_new) %in% f_pay)]

# For test data
scaled_test_cont = predict(preprocess_min_max, test[,features_cont])
test_scaled = cbind(scaled_test_cont, test[, features_cat], default = test$default)

m_test = dim(test_scaled)[1]
# Repeat same pipeline as training data
df = data.frame(matrix(,nrow = m_test, ncol = 0))
f_pay = c("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")

for (j in f_pay){
  predictor = test_scaled[,j]
  col_names = sapply(c("neg2_{j}", "duly_{j}", "zero_{j}", "delay_months_{j}"), glue)
  names(col_names) = NULL
  df1 = my_cat_encode(predictor, ncol = 4, col_names = col_names, m = m_test)
  df = cbind(df, df1)
}

test_new = cbind(test_scaled, df)
test_new = test_new[, !(colnames(test_new) %in% f_pay)]

cat('Training data no. of samples: ', dim(train_new)[1], 
    ' ; no. of features: ', dim(train_new)[2])
cat('Test data no. of samples: ', dim(test_new)[1], 
    ' ; no. of features: ', dim(test_new)[2])

# train_new and test_new are the dataframe used for logistic regression, lasso and GAM.

# performance measures ----------------------------------------------------

performance = function(truth, pred, threshold=0.5){
  # Expect pred to be posterior probability
  if (typeof(pred) == 'double'){
    tab = table(truth, pred>=threshold)
  } else {
    tab = table(truth, pred)
  }
  acc = (sum(diag(tab)))/sum(tab)
  recall = tab[2,2]/sum(tab[2,])
  precision = tab[2,2]/sum(tab[,2])
  f1_score = (2*recall*precision)/(recall+precision)
  # Calculate AUC, change the response variable to boolean variable
  AUC = calcAUC(pred, as.numeric(truth)-1)
  round(c(accuracy = acc, recall = recall, precision = precision,
          f1_score = f1_score, AUC = AUC),4)
}

# models ----------------------------------------------------------------

variables = glue('log({features_cont}+1)')
response = 'default'
cat_var = setdiff(colnames(train_new), c(features_cont, response))
all_vars = c(variables, cat_var)
f = as.formula(
  paste(response, paste(all_vars, collapse = " + "), sep = " ~ ")
)
print(f)

model_logreg = glm(f, data = train_new, family = binomial(link = "logit"))
summary(model_logreg)

# Show just the statistical significant coefficient estimates
coef_mat = summary(model_logreg)$coefficients
coef_sig = coef_mat[coef_mat[,4]<=0.05,]
coef_sig

library(car)
library(knitr)
VIF = vif(model_logreg)
kable(data.frame(vif(model_logreg)), caption = "VIF for predictors")

kable(data.frame(VIF[VIF[,3]>5,]), caption = "VIF for strongly correlated predictors")

pred = predict(model_logreg, newdata = train_new, type = "response")
pred_test = predict(model_logreg, newdata = test_new, type = "response")

test$pred_logreg = pred_test
test$binary_default = (test$default==1)

library(sigr)
(perf_train_logreg = performance(train_new$default, pred = pred))
(perf_test_logreg = performance(test_new$default, pred = pred_test))

# Precision recall trade-off
# Double density plot
DoubleDensityPlot(test, xvar = "pred_logreg", truthVar = "default", 
                  title = "Distribution of scores of LR model in test data") +
  geom_vline(xintercept = 0.5, color = 'red', linetype = 2)

# Enrichment-recall plot
train$pred = pred
train$binary_default = train$default == 1
PRTPlot(train, 'pred', 'binary_default', TRUE, 
        plotvars = c('enrichment', 'recall'), thresholdrange = c(0,1),
        title = 'Enrichment/Recall vs thresholds for LR model')

# lasso -------------------------------------------------------------------

library(glmnet)
library(glmnetUtils)
model_lasso = cv.glmnet(f, data = train_new, alpha = 0, 
                        family = "binomial", standardize = FALSE)
# summary(model_lasso)
coeff = coef(model_lasso)
coef_frame <- data.frame(coef = rownames(coeff)[-1],
                         value = coeff[-1,1])
ggplot(coef_frame, aes(x = coef, y = value)) +
  geom_pointrange(aes(ymin = 0, ymax = value)) +
  ggtitle("Coefficients of lasso model") +
  coord_flip() +
  theme(axis.text.y = element_text(size = 6))

# Performance evaluation
pred = predict(model_lasso, newdata = train_new, type = "response")
pred_test = predict(model_lasso, newdata = test_new, type = "response")

test$pred_lasso = pred_test

(perf_train_lasso = performance(train_new$default, pred = pred))
(perf_test_lasso = performance(test_new$default, pred = pred_test))


# GAM ---------------------------------------------------------------------

library(mgcv)

# Construct the formula
variables = glue('s(log({features_cont}+1))')
f_add = train_new %>% select(tail(names(.),24)) %>% colnames()
f_cat = c("SEX", "EDUCATION", "MARRIAGE")
all_vars = c(variables, c(f_add,f_cat))
# Change the response to binary variable (Boolean)
train_new$binary_default = (train_new$default==1)

f = as.formula(
  paste("binary_default", paste(all_vars, collapse = " + "), sep = " ~ ")
)
print(f)

gam_model = gam(f, data = train_new, family = binomial(link = "logit"), 
                standardize = FALSE)
gam_model$converged   # check the convergence of gam

# To Visualize some "interesting" variables
(sum_gam = summary(gam_model))

# Visualize s() outputs
terms = predict(gam_model, type="terms")

frame1 = as.data.frame(terms)
# Remember that 1 is negative, while 2 is positive
# vars=setdiff(colnames(train_new),c("default","binary_default"))

df10= sum_gam$chi.sq
names(df10)
vars = names(sum_gam$chi.sq)[which(sum_gam$s.pv<0.05)]
vars
clean_vars = gsub('[()]','',vars)
clean_vars = sub("slog",'',clean_vars)
clean_vars = gsub('\\+','',clean_vars)
clean_vars = gsub('\\s','', clean_vars)
clean_vars = substr(clean_vars,1,nchar(clean_vars)-1)

frame1_visual = frame1[,vars]
train_gam_visual = train[,clean_vars]

# Long pivot
library(cdata)
frame1_visual_long = unpivot_to_blocks(frame1_visual, nameForNewKeyColumn = "basis_function",
                                       nameForNewValueColumn = "basis_values",
                                       columnsToTakeFrom = vars)
train_gam_visual_long = unpivot_to_blocks(train_gam_visual, 
                                          nameForNewKeyColumn = "predictors",
                                          nameForNewValueColumn = "values",
                                          columnsToTakeFrom = clean_vars)

dat_visual = cbind(frame1_visual_long, train_gam_visual_long)

dat_visual %>% ggplot(aes(x = values, y = basis_values)) +
  geom_smooth() +
  facet_wrap(~predictors, ncol = 3, scales = "free")

pred = predict(gam_model, newdata = train_new, type = "response")
pred_test = predict(gam_model, newdata = test_new, type = "response")

test$pred_gam = pred_test

(perf_train_gam = performance(train_new$default, pred = pred))
(perf_test_gam = performance(test_new$default, pred = pred_test))

# kNN  --------------------------------------------------------------------

# Use train_new
dummy=dummyVars("~.",data=train_new[,f_cat])
cat_new=data.frame(predict(dummy,newdata=train_new[,f_cat]))
var_removed = c("binary_default", f_cat)

train_onehot = train_new[, !(colnames(train_new) %in% var_removed)]
train_onehot = cbind(cat_new, train_onehot)

# Test data
cat_new = data.frame(predict(dummy,newdata=test_new[,f_cat]))
test_onehot = test_new[, !(colnames(test_new) %in% var_removed)]
test_onehot = cbind(cat_new, test_onehot)

cat('Training data no. of samples (kNN): ', dim(train_onehot)[1], 
    ' ; no. of features: ', dim(train_onehot)[2])
cat('Test data no. of samples (kNN): ', dim(test_onehot)[1], 
    ' ; no. of features: ', dim(test_onehot)[2])

# Use train-test validation approach
set.seed(3)
split = initial_split(train_onehot, prop = 0.8, strata = "default")
train_knn = training(split)
test_knn = testing(split)

target_train = train_knn$default==1
target_test = as.numeric(test_knn$default)-1
train_knn = train_knn[, !(colnames(train_knn) %in% c("default"))]
test_knn = test_knn[, !(colnames(test_knn) %in% c("default"))]

library(class)

#knn_pred = knn(train_onehot, test_onehot, (train_new$default==1), k=155, prob = TRUE)
#tab = table(test_new$default==1, knn_pred)
#sum(diag(tab))/sum(tab)

library(sigr)
#k = ifelse(knn_pred==TRUE, 1, 0)
#AUC = calcAUC(k, as.numeric(test_new$default)-1)
#AUC
#default_boolean = as.numeric(test_new$default) - 1
k_range = seq(from = 115, to = 175, by = 2)
AUC_vec = rep(NA, length(k_range))
iter = 0
for (K in (k_range)) {
  iter = iter + 1
  knn_pred = knn(train_knn, test_knn, target_train, k=K)
  AUC_vec[iter] = calcAUC(ifelse(knn_pred==TRUE, 1, 0), target_test)
}

df_plot_k = data.frame(k = k_range, AUC = AUC_vec)
k_opt = df_plot_k[which.max(AUC_vec),1]

df_plot_k %>% 
  ggplot(aes(x = k, y = AUC)) +
  geom_line() +
  geom_vline(xintercept = k_opt, linetype = 2, color = 'red') +
  xlab('Number of nearest neighbors, k') +
  ylab('AUC')
  
knn_pred = knn(train_onehot, test_onehot, (train_onehot$default==1), 
               k=k_opt, prob = TRUE)

prop_vote = attr(knn_pred,'prob')
prob_pos = ifelse(knn_pred==FALSE, 1-prop_vote, prop_vote)
test$pred_knn = prob_pos
(perf_test_knn = performance(truth = test_new$default, pred = prob_pos))


# Tabulation of classifiers' performance ----------------------------------

kable(rbind(perf_test_logreg, perf_test_lasso, perf_test_gam, perf_test_knn))
# ROC
library(WVPlots)
test$binary_default = (test$default==1)
ROCPlotList(
  frame = test,
  xvar_names = c("pred_logreg", "pred_lasso", "pred_gam", "pred_knn"),
  truthVar = "binary_default", truthTarget = TRUE,
  title = "ROC plots for all classifiers"
)

# Threshold of 0.2
perf_test_logreg = performance(truth = test$default, pred = test$pred_logreg, 
                               threshold = 0.2)
perf_test_lasso = performance(test$default, test$pred_lasso, threshold = 0.2)
perf_test_gam = performance(test$default, test$pred_gam, threshold = 0.2)
perf_test_knn = performance(test$default, test$pred_knn, threshold = 0.2)
kable(rbind(perf_test_logreg, perf_test_lasso, perf_test_gam, perf_test_knn), 
      caption = "test performances for different models (thresholds=0.2)")