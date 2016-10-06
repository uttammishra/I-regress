#set your working directory
setwd("C:/Users/uttam/Downloads")
#setwd("/scratch/mgacesa/fl")

install.packages("dplyr")
install.packages("plyr")
rm(list=ls())
library(e1071)
library(RTextTools)
library(topicmodels)
library(stringi)
library(SnowballC)
library(slam)
library(tm)
library(ldatuning)
library(dplyr)
library(plyr)
options(scipen=999)

#load sample df
df_sp<-read.csv('sample_combine_clean.csv',header=T,stringsAsFactors=F)
#df_sp<-read.csv('sample_combine_clean_catagory.csv',header=T,stringsAsFactors=F)
df_sp$Text<-stri_trim(df_sp$Text)
df_sp<-df_sp[!(is.na(df_sp$Text)), ]

out_sp<- strsplit(df_sp$Text, " ")

names(out_sp) <- paste0("doc", 1:length(out_sp))

#extract words
lev<- sort(unique(unlist(out_sp)))


#calulate the frequency and provide a matirx
dat_sp <- do.call(rbind, lapply(out_sp, function(x, lev) tabulate(factor(x, levels = lev, ordered = TRUE), nbins = length(lev)), lev= lev))
colnames(dat_sp) <- sort(lev)

# 
container <- create_container(dat_sp,as.numeric(as.factor(df_sp[,5])),trainSize=1:800, testSize=801:962,virgin=FALSE)
models <- train_models(container, algorithms=c("MAXENT","SVM","GLMNET","SLDA","TREE","BAGGING","BOOSTING","RF"))
#models <- train_models(container, algorithms = c("MAXENT", "BAGGING"))
results <- classify_models(container, models)

# calculate the precison accuracy precision, recall, fmeasur
s<-data.frame()
#y <- results[,4]
#y<-df_sp[c(9,10,11)]

for (j in 1:8)
  {
  i=j*2-1
  acc<-recall_accuracy(as.numeric(df_sp[801:962,11]), results[,i])
  
  tp<-count((results[,i]==1) & (df_sp[801:962,11]==1) & (results[,i]==df_sp[801:962,11]))
  tn<-count(results[,i]==0 & df_sp[801:962,11]==0 & results[,i]==df_sp[801:962,11])
  fp<-count(results[,i]==1 & df_sp[801:962,11]==0 )
  fn<-count(results[,i]==0 & df_sp[801:962,11]==1 )
     retrieved <- sum(as.numeric(as.character(results[,i])))
     precision <-sum ((as.numeric(results[,i])& df_sp[801:962,11]))/retrieved
     recall <- sum ((as.numeric(results[,i])& df_sp[801:962,11])) / sum( df_sp[801:962,11])
  #  
  #precision<-(162-tp[1,2])/((162-tp[1,2])+(162-fp[1,2]))
  #recall<-(162-tp[1,2])/((162-tp[1,2])+(162-fn[1,2]))
  #fmeasure <- 2 * precision * recall / (precision + recall)
  m<-c(acc, precision, recall) #, fmeasure)
  s<- rbind(s,m)
}

save.image()

