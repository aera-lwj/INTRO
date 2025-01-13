# Load packages
library(caret)
library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)


# Read the dataset
spotify_data <- read.csv("dataset.csv")

# Deal with missing data
colSums(is.na(spotify_data))
spotify_data <- na.omit(spotify_data)

# Filter K-Pop and mandopop data and select the top 500 songs by popularity for each genre
kpop_data <- spotify_data %>%
  filter(track_genre == "k-pop") %>%
  arrange(desc(popularity)) %>%
  slice(1:500)

mandopop_data <- spotify_data %>%
  filter(track_genre == "mandopop") %>%
  arrange(desc(popularity)) %>%
  slice(1:500)

# Combine the data
kpop_mandopop_data <- bind_rows(kpop_data, mandopop_data)

# Create a boxplot to visualize the distribution of audio features by genre
kpop_mandopop_data %>%
  pivot_longer(cols = c(danceability, energy, valence, loudness),
               names_to = "audio_feature",
               values_to = "value") %>%
  ggplot(aes(x = track_genre, y = value, fill = track_genre)) +
  geom_boxplot() +
  facet_wrap(~audio_feature, scales = "free_y") +
  labs(title = "Distribution of Audio Features by Genre",
       x = "Genre",
       y = "Feature Value")

# Create a Heatmap to display the correlation of audio features
correlation_matrix <- cor(kpop_mandopop_data %>% 
                            select(danceability, energy, valence, loudness))
correlation_matrix [upper.tri(correlation_matrix )] <- NA
correlation_melt <- melt(correlation_matrix)

ggplot(correlation_melt, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "darkred", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name="Correlation", na.value = "white") +
  theme_minimal() +
  labs(title = "Correlation Heatmap of Audio Features",
       x = NULL,
       y = NULL) +
  theme(axis.text.x = element_text(hjust = 1)) +
  geom_text(aes(label = ifelse(!is.na(value), round(value, 2), "")), color = "black", size =3)


# Scatter plot: Danceability vs Energy
ggplot(kpop_mandopop_data, aes(x = danceability, y = energy, color = track_genre)) +
  geom_point(alpha = 0.7) +
  labs(title = "K-Pop vs. Mandopop: Danceability vs Energy",
       x = "Danceability", y = "Energy", color = "Genre") +
  theme_minimal()


# Convert track_genre to a binary variable, 1 for K-Pop, 0 for Mandopop
kpop_mandopop_data <- kpop_mandopop_data %>%
  mutate(genre_binary = ifelse(track_genre == "k-pop", 1, 0))


# Split the data: 80% train_data, 20% test_data
set.seed(42)
train_indices <- createDataPartition(kpop_mandopop_data$genre_binary, p = 0.8, list = FALSE)
train_data <- kpop_mandopop_data[train_indices, ]
test_data <- kpop_mandopop_data[-train_indices, ]


# Logistic regression model: danceability and energy
logit_model <- glm(
  formula = genre_binary ~ danceability + energy,
  data = train_data,
  family = binomial(link = "logit")
)
summary(logit_model)


# Predict and evaluate model
predictions <- test_data %>%
  mutate(
    predicted_prob = predict(logit_model, newdata = test_data, type = "response"), 
    predicted_class = ifelse(predicted_prob > 0.5, 1, 0) 
  )

confusion_matrix <- table(
  Actual = predictions$genre_binary,
  Predicted = predictions$predicted_class
)


# Model metrics
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")



# Logistic regression with interaction term
kpop_mandopop_data_inter <- kpop_mandopop_data %>%
  mutate(interaction_term = danceability * energy)

train_data_inter <- train_data %>% mutate(interaction_term = danceability * energy)
test_data_inter <- test_data %>% mutate(interaction_term = danceability * energy)

# Update the logistic regression model
logit_model_interaction <- glm(
  formula = genre_binary ~ danceability + energy + interaction_term,
  data = train_data_inter,
  family = binomial(link = "logit")
)
summary(logit_model_interaction)

# Predict and evaluate interaction model
predictions_inter <- test_data_inter %>%
  mutate(
    predicted_prob = predict(logit_model_interaction, newdata = test_data_inter, type = "response"), 
    predicted_class = ifelse(predicted_prob > 0.5, 1, 0) 
  )

confusion_matrix_inter <- table(
  Actual = predictions_inter$genre_binary,
  Predicted = predictions_inter$predicted_class
)

# Interaction model metrics
accuracy_inter <- sum(diag(confusion_matrix_inter)) / sum(confusion_matrix_inter)
precision_inter <- confusion_matrix_inter[2, 2] / sum(confusion_matrix_inter[, 2])
cat("Interaction Model - Accuracy:", accuracy_inter, "\n")
cat("Interaction Model - Precision:", precision_inter, "\n")
