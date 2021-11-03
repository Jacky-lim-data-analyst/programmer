# Reference: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques 
# for the predictive accuracy of probability of default of credit card clients. Expert 
# Systems with Applications, 36(2), 2473-2480.  

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

summary(train$default)/dim(train)[1]
summary(test$default)/dim(test)[1]

# Data visualization for continuous predictors------------------------

features_cont = setdiff(colnames(train), c(features_cat, response))

library(cdata)
library(ggplot2)
# Change from wide to tall format
data_long = unpivot_to_blocks(train, nameForNewKeyColumn = "variables",
                              nameForNewValueColumn = "values", 
                              columnsToTakeFrom = features_cont)

data_long %>% ggplot(aes(x=values)) +
  geom_histogram(bins = 10, fill="gray") +
  facet_wrap(~variables, ncol = 4, scales = "free")

# Boxplots
data_long = unpivot_to_blocks(train, nameForNewKeyColumn = "variables",
                              nameForNewValueColumn = "values", 
                              columnsToTakeFrom = features_cont[1:2])  
# Limit balance and age
ggplot(data_long,aes(x=default,y=values)) +
  geom_boxplot(color="blue",fill="blue",alpha=0.2,notch=TRUE,
               outlier.color="red",outlier.fill = "red",outlier.size = 2) +
  facet_wrap(~variables,ncol = 2,scales = "free")

# Correlation matrix
library(reshape2)
corr_mat = cor(train[,features_cont])
corr_mat[upper.tri(corr_mat)]=NA
melted_cormat = melt(corr_mat, na.rm = TRUE)

ggplot(data = melted_cormat, aes(x=Var1, y = Var2, fill=value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust=1, size =10, hjust=1)) +
  coord_fixed() +
  geom_text(aes(Var1, Var2, label = round(value,2)), color = "black", size = 3)


# Data visualization for categorical predictors ---------------------------

# Contingency table
table_cont = train %>% count(EDUCATION, default)

table_sum = train %>% count(EDUCATION, default) %>% group_by(EDUCATION) %>% 
  summarise(sum_col=sum(n))
row_sum = train %>% count(EDUCATION, default) %>% group_by(default) %>% 
  summarise(sum_row=sum(n))

# colormap (heatmap)
table_cont %>% left_join(table_sum, by = "EDUCATION") %>% 
  mutate(percent = n/sum_col) %>% 
  ggplot(aes(x = EDUCATION, y = default)) +
  geom_tile(aes(fill = percent)) +
  theme(axis.text.x = element_text(angle = 45,hjust=1)) +
  geom_text(aes(label = round(percent,2), hjust=1, vjust=1, colour = "white"),
            show.legend = FALSE) +
  geom_text(aes(label = n, hjust=1, vjust=0, colour = "red"), show.legend = FALSE)

# contingency table
table(train$EDUCATION, train$default)

# Chi-square statistics
train_cat_res = train[, !(colnames(train) %in% features_cont)]
#chisq.test(train_cat_res[,2], train_cat_res$default)

t=c()
idx=c()
for (i in (1:(ncol(train_cat_res)-1))) {
  t[i]=chisq.test(train_cat_res[,i],train_cat_res$default)$p.value
  # u[i]=fisher.test(train[,i],train$credit_risk)$p.value
  if (!is.list(tryCatch( { result <- chisq.test(train_cat_res[,i],train_cat_res$default) }
                         , warning = function(w) { print("TRUE") }))) {
    idx=c(idx,i)
  }
}

idx_sig=which(t<=0.05)

idx_int=!(idx_sig %in% idx)
colnames(train_cat_res)[idx_sig[idx_int]]
print(paste("Variables not independent with response: ", 
            colnames(train_cat_res)[idx_sig[idx_int]]))

# Grouped bar plots
m = dim(train)[1]
table_cont %>% left_join(table_sum, by = "EDUCATION") %>% 
  left_join(row_sum, by = "default") %>% 
  mutate(expected_n = (sum_col*sum_row)/m) %>% 
  unpivot_to_blocks(nameForNewKeyColumn = "variables",
                    nameForNewValueColumn = "values",
                    columnsToTakeFrom = c("n", "expected_n")) %>% 
  ggplot(aes(x = default, y = values, group = variables, fill = variables)) +
  geom_bar(stat = "identity", width = 0.5, position = "dodge") +
  facet_wrap(.~EDUCATION,ncol = 3, scales = "free_y") +
  geom_text(aes(label = round(values,2)), vjust = 1.6, size = 3) +
  theme_bw()

