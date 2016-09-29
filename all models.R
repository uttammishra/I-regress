
setwd("C:/Users/uttam/Downloads")
rm(list=ls())

#install required packages
install.packages("e1071")
install.packages("RTextTools")
install.packages("topicmodels")
install.packages("stringi")
install.packages("SnowballC")
install.packages("slam")
install.packages("tm")
install.packages("ldatuning")



require(e1071)
require(RTextTools)
require(topicmodels)
require(stringi)
require(SnowballC)
require(slam)
require(tm)
require(ldatuning)
options(scipen=999)l

#load sample df
df_sp<-read.csv('sample_combine_clean.csv',header=T,stringsAsFactors=F)
#df_sp<-read.csv('experiment_part1_filtered.csv',header=T,stringsAsFactors=F)
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
container <- create_container(dat_sp,as.numeric(df_sp$p.infor),trainSize=1:100, testSize=101:862,virgin=FALSE)
models <- train_models(container, algorithms=c("MAXENT","SVM","TREE", "BAGGING","BOOSTING")) 
model <- train_models(container, algorithms = "MAXENT")
#,"GLMNET", "SLDA","TREE","BAGGING","BOOSTING","RF"))
#as.factor(df_sp[,6]
results <- classify_models(container, model)

analysis <- cross_validate(container, 4,"BAGGING")

summary(results)

analytics<-create_analytics(container,results)

View(analytics)

write.csv(analytics@document_summary, "DocumentSummary.csv")

