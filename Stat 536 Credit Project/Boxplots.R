setwd("C:/Users/cason/Desktop/Classes/Assignments/Stat 536/Homework 6")

credit <- read.csv("creditcard.csv", header = TRUE)

library(ggplot2)
library(reshape2)
library(plyr)

cols <- which(colnames(credit) %in% c("Class","V4","V12","V22"))

df1_long <- melt(credit[,cols], id.vars=c("Class"))

df1_long$Class <- revalue(factor(df1_long$Class), c("0"="Safe","1"="Fraud"))

ggplot(df1_long, aes(y = value, fill = Class)) + 
  geom_boxplot() + 
  facet_wrap(~variable) +
  ylab(NULL)

