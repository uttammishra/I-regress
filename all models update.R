#set your working directory
setwd("C:/Users/uttam/Downloads")

rm(list=ls())
library(e1071)
library(RTextTools)
library(stringi)
library(SnowballC)
library(slam)
library(tm)
library(dplyr)
library(plyr)
options(scipen=999)

#load sample df
df_sp<-read.csv('sample_combine_clean.csv',header=T,stringsAsFactors=F)
df_sp$Text<-stri_trim(df_sp$Text)
df_sp<-df_sp[!(is.na(df_sp$Text)), ]

out_sp<- strsplit(df_sp$Text, " ")

names(out_sp) <- paste0("doc", 1:length(out_sp))

#extract words
lev<- sort(unique(unlist(out_sp)))


#calulate the frequency and provide a matirx
dat_sp <- do.call(rbind, lapply(out_sp, function(x, lev) tabulate(factor(x, levels = lev, ordered = TRUE), nbins = length(lev)), lev= lev))
colnames(dat_sp) <- sort(lev)

# for informativity, you need one run as:
container <- create_container(dat_sp,as.numeric(as.factor(df_sp[,11])),trainSize=1:800, testSize=801:962,virgin=FALSE)
# for persuavisve
# container <- create_container(dat_sp,as.numeric(as.factor(df_sp[,6])),trainSize=1:800, testSize=801:962,virgin=FALSE)
# for transformative:
# container <- create_container(dat_sp,as.numeric(as.factor(df_sp[,7])),trainSize=1:800, testSize=801:962,virgin=FALSE)

# for binary informativity, you need one run as:
# container <- create_container(dat_sp,as.numeric(as.factor(df_sp[,11])),trainSize=1:800, testSize=801:962,virgin=FALSE)
# for binary persuavisve
# container <- create_container(dat_sp,as.numeric(as.factor(df_sp[,12])),trainSize=1:800, testSize=801:962,virgin=FALSE)
# for binary transformative:
# container <- create_container(dat_sp,as.numeric(as.factor(df_sp[,13])),trainSize=1:800, testSize=801:962,virgin=FALSE)


# since the server is too small. Error: cannot allocate vector of size 871.9 Mb
#models <- train_models(container, algorithms=c("MAXENT","SVM","GLMNET","SLDA","TREE","BAGGING","BOOSTING","RF"))
#results <- classify_models(container, models)
# let run model one by one.

models0 <- train_models(container, algorithms = c("MAXENT"))
results0 <- classify_models(container, models0)
models1 <- train_models(container, algorithms = c("SVM"))
results1 <- classify_models(container, models1)
models2 <- train_models(container, algorithms = c("GLMNET"))
results2 <- classify_models(container, models2)
models3 <- train_models(container, algorithms = c("SLDA"))
results3 <- classify_models(container, models3)
models4 <- train_models(container, algorithms = c("TREE"))
results4 <- classify_models(container, models4)
models5 <- train_models(container, algorithms = c("BAGGING"))
results5 <- classify_models(container, models5)
models6 <- train_models(container, algorithms = c("BOOSTING"))
results6 <- classify_models(container, models6)
models7 <- train_models(container, algorithms = c("RF"))
results7 <- classify_models(container, models7)

results<- cbind(results0, result1, result2, result3, result4, result5, result6,result7)

# transfer the factor to number
indx <- sapply(results, is.factor)
results[indx]<-lapply(results[indx], function(x) as.numeric(levels(x)[x])-1)
# calculate the  precison accuracy precision, recall, fmeasure of Binary infomrative
s<-data.frame()

for (j in 1:8){
  i=j*2-1
  acc<-recall_accuracy(as.numeric(df_sp[801:962,11]), results[,i])
  
  tp<-sum(results[,i]==1 & df_sp[801:962,11]==1 & results[,i]==df_sp[801:962,11])
  tn<-sum(results[,i]==0 & df_sp[801:962,11]==0 & results[,i]==df_sp[801:962,11])
  fp<-sum(results[,i]==1 & df_sp[801:962,11]==0 )
  fn<-sum(results[,i]==0 & df_sp[801:962,11]==1 )
  #   retrieved <- sum(as.numeric(as.character(results[,i]))
  #   precision <-sum ((as.numeric(results[,i])& df_sp[801:962,12]))/retrieved
  #   recall <- sum ((as.numeric(results[,i])& df_sp[801:962,12])) / sum( df_sp[801:962,12])
  #  
  precision<-(162-tp)/((162-tp)+(162-fp))
  recall<-(162-tp)/((162-tp)+(162-fn))
  fmeasure <- 2 * precision * recall / (precision + recall)
  m<-c(acc, precision, recall, fmeasure)
  s<- rbind(s,m)
}

