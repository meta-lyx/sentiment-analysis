loadedNamespaces()
install.packages("caret")
install.packages("tidyverse")
install.packages("tm")
install.packages("tokenizers")
install.packages("tidytext")
library(caret)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(tidytext)
library(tokenizers)
library(tm)
df <- read.csv('annotated_and_facebook_train_neg_1601_pos_1620_neu_1100_val__actual_validation.csv')

#counting number of characters
df <- df %>% mutate(text_length = nchar(df$Text))

#tokenized text as an added new column
df_token <- df %>% mutate(tokenized_text = tokenize_words(df$Text))

#create stopwords dataframe
ens_stopwords = data.frame(word = stopwords("en"))
df_token_wo_stopwords <- df_token %>% 
  unnest_tokens(word, Text, token = 'ngrams', n = 2) %>%
  anti_join(ens_stopwords)


df_token_wo_stopwords 

#create the bi-gram word distribution table, not grouped by sentiment label
frequency_df = df_token_wo_stopwords %>%
  group_by(TemperatureLabel) %>%
  count(word) %>% 
  arrange(desc(n))

#split based on column value (temperature label)
data_list <- split(frequency_df, f = frequency_df$TemperatureLabel)
do.call(rbind, data_list)

#split into three separate data frames
negative_df <- as.data.frame(data_list[1])

#visualize the first negative reviews and see which bi-grams are
#the most popular
negative_df_graph <-  negative_df %>%
  order(desc(n)) %>%
  slice(1:10) %>%
  ggplot() + geom_bar(aes(negative.word, negative.n), stat = "identity", fill = "#de5833") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams of Negative Reviews",
       subtitle = "using Tidytext in R",
       caption = "Data Source: Training data")
negative_df_graph

positive_df <- as.data.frame(data_list[2])
neutral_df <- as.data.frame(data_list[3])



df_text_sentiment <- df %>%
  group_by(TemperatureLabel) %>%
  summarise(n = n(), mean = mean(text_length), sd = sd(text_length)) %>%
  mutate(se = sd/sqrt(n))


error_graph <- ggplot(df_text_sentiment) +
  geom_bar(aes(x = TemperatureLabel, y = median)
           , stat="identity", fill="skyblue", alpha=0.7) + 
  geom_errorbar(aes(x = TemperatureLabel, ymin= median-sd, ymax=median-sd), width=0.4, colour="orange", alpha=0.5, size=1.3)
#error_graph

label_negative <- df %>%
  filter(TemperatureLabel == 'negative')

label_negative_boxplot <- label_negative %>%
  ggplot(., aes(x = TemperatureLabel, y = text_length, fill = 
                  TemperatureLabel)) +
  geom_boxplot() + labs(x = "TemperatureLabel", fill = "temperatureLabel") +
  theme_minimal() + 
  theme(axis.text.x=element_text (angle =45, hjust =1))

label_negative_boxplot

boxplot(text_length ~ TemperatureLabel, data = df)

stripchart(df$text_length ~ df$TemperatureLabel, vertical = TRUE, method = "jitter",
           pch = 19, add = TRUE, col = 1:length(levels(df$TemperatureLabel)))

#how many percentage of words are above x level of text_length
#doing a multi-nominal logistic regression

df$TemperatureLabel <- as.factor(df$TemperatureLabel)
levels(df$TemperatureLabel)



index <- createDataPartition(df$TemperatureLabel,
                             p = 0.7, list = FALSE)
train <- df[index,]
test <- df[-index,]

train$TemperatureLabel <- relevel(train$TemperatureLabel, ref = "negative")

require(nnet)
multinom_model <- multinom(TemperatureLabel ~ text_length,data = df)

summary(multinom_model)

multinom(formula = TemperatureLabel ~ text_length, data = df)

exp(coef(multinom_model))

head(round(fitted(multinom_model),2))

train$TemperatureLabelPredicted <- predict(multinom_model, newdata = train,
                                           'class')
tab <- table(train$TemperatureLabel, train$TemperatureLabelPredicted)

round((sum(diag(tab))/sum(tab))*100,2)

test$TemperatureLabelPredicted <- predict(multinom_model, newdata = test, "class")

tab <- table(test$TemperatureLabel, test$TemperatureLabelPredicted)
tab
