## Chen Jia, Jose Orozco Becerra, Marloes Evers, Sathya Jagannatha
## u863194, u512585, u264541, u580435

## Load packages ---------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("randomForest")) install.packages("randomForest")
library(gridExtra)
library(randomForest)

# 1. PREPROCESSING AND EDA -----------------------------------------------------

## Set working directory -------------------------------------------------------
#setwd()

## Load dataset 1 --------------------------------------------------------------
Players <- read.csv("input/Players.csv", stringsAsFactors = FALSE)

## Check Missing data ----------------------------------------------------------
counts_missing <- colSums(is.na(Players))
counts_missing
percentage_missing <- colMeans(is.na(Players))

## Delete missing data ---------------------------------------------------------
Players <- na.omit(Players)

## Delete useless feautres -----------------------------------------------------
Players <- Players %>%
  mutate(FG. = NULL) %>%
  mutate(X3P. = NULL) %>%
  mutate(FT. = NULL) %>%
  mutate(REB = NULL) %>%
  mutate(EFF = NULL) %>%
  mutate(AST.TOV = NULL) %>%
  mutate(STL.TOV = NULL) %>%
  mutate(Birth_Place = NULL) %>%
  mutate(Birthdate = NULL) %>%
  mutate(Collage = NULL) %>%
  mutate(Experience = NULL) %>%
  mutate(Team = NULL) %>%
  mutate(BMI = NULL) 

## Factorizing target feature --------------------------------------------------
Players$Pos <- as.factor(Players$Pos)

## Check Missing data after removement -----------------------------------------
counts_missing_complete <- colSums(is.na(Players))
percentage_missing_complete <- colMeans(is.na(Players))

## Checking outliers -----------------------------------------------------------
Players %>%
  group_by(as.factor(Pos))

Points_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = PTS, y = Pos)) +
  labs(x = 'Points') +
  scale_y_discrete("Positions")

FGM_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = FGM, y = Pos)) +
  labs(x = 'Field goals made') +
  scale_y_discrete("Positions")

FGA_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = FGA, y = Pos)) +
  labs(x = 'Field goals attempted') +
  scale_y_discrete("Positions")

X3PM_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = X3PM, y = Pos)) +
  labs(x = '3 points made') +
  scale_y_discrete("Positions")

X3PA_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = X3PA, y = Pos)) +
  labs(x = '3 points attempted') +
  scale_y_discrete("Positions")

FTM_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = FTM, y = Pos)) +
  labs(x = 'Free throws made') +
  scale_y_discrete("Positions")

FTA_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = FTA, y = Pos)) +
  labs(x = 'Free throws attempted') +
  scale_y_discrete("Positions")

OREB_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = OREB, y = Pos)) +
  labs(x = 'Offensive rebounds') +
  scale_y_discrete("Positions")

DREB_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = DREB, y = Pos)) +
  labs(x = 'Deffensive rebounds attempted') +
  scale_y_discrete("Positions")

AST_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = AST, y = Pos)) +
  labs(x = 'Assist') +
  scale_y_discrete("Positions")

STL_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = STL, y = Pos)) +
  labs(x = 'Stealth') +
  scale_y_discrete("Positions")

BLK_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = BLK, y = Pos)) +
  labs(x = 'Blocks') +
  scale_y_discrete("Positions")

TOV_outliers <- Players%>%
  ggplot()+
  geom_boxplot(data = Players, aes(x = TOV, y = Pos)) +
  labs(x = 'Turnover') +
  scale_y_discrete("Positions")

PF_outliers <- Players%>%
  ggplot() +
  geom_boxplot(data = Players, aes(x = PF, y = Pos)) +
  labs(x = 'personal foul') +
  scale_y_discrete("Positions")

HW_outliers <- ggplot()+
  geom_point(data = Players, aes(x = Height, y = Weight))

Weight_outliers <- ggplot()+
  geom_boxplot(data = Players, aes(x = Weight, y = Pos)) +
  labs(x = "Weight") +
  scale_y_discrete("Positions")
Weight_outliers

## Remove outliers -------------------------------------------------------------
Points_outliers <- Players %>%
  filter(PTS == max(Players$PTS, na.rm = TRUE))

X3PM_outliers <- Players %>%
  filter(X3PM == max(Players$X3PM, na.rm = TRUE))

Weight_outliers <- Players %>%
  filter(Weight == max(Players$Weight, na.rm = TRUE))

Players <- Players %>%
  anti_join(Points_outliers) %>%
  anti_join(X3PM_outliers) %>%
  anti_join(Weight_outliers)

## EDA -------------------------------------------------------------------------
summary <- summary(Players)
summary

feature_cor <- Players %>%
  mutate(Name = NULL) %>%
  mutate(Pos = NULL) %>%
  cor()

ggplot()+
  geom_histogram(data = Players, aes(x = Height), stat="bin")

ggplot()+
  geom_histogram(data = Players, aes(x = Weight), stat="bin")

ggplot()+
  geom_histogram(data = Players, aes(x = Weight), 
                 stat="bin", color = "white", fill="steelblue") +
  theme_minimal() +
  labs(x = 'Weight', y = 'Count') +
  facet_grid(Pos~.,)

ggplot()+
  geom_histogram(data = Players, aes(x = Height), 
                 stat="bin", color = "white", fill="steelblue") +
  theme_minimal() +
  labs(x = 'Height', y = 'Count') +
  facet_grid(Pos~.,)


# 2. KNN -----------------------------------------------------------------------

## Copy of the original dataset ------------------------------------------------
Players1 = Players

## Setting a seed --------------------------------------------------------------
set.seed(1)

## Partitioning the dataset for training and testing ---------------------------
trn_index = createDataPartition(y = Players1$Pos, p = 0.70, list = FALSE)
trn_pos = Players1[trn_index, ]
tst_pos = Players1[-trn_index, ]

## Setting a seed and training the model (KNN) ---------------------------------
# With scaling and centering
set.seed(1)
knn_pos = train(Pos ~ PTS + FGM + FGA + X3PM + X3PA + 
                  FTM + FTA + OREB + DREB + AST + STL + 
                  BLK + TOV + PF + Age + Height + Weight,
                method = "knn", data = trn_pos,
                trControl = trainControl(method = 'cv', number = 5),
                preProcess = c("center","scale"),
                tuneLength = 20)

knn_pos

# Without scaling and centering
knn_pos_2 = train(Pos ~ PTS + FGM + FGA + X3PM + X3PA + 
                  FTM + FTA + OREB + DREB + AST + STL + 
                  BLK + TOV + PF + Age + Height + Weight,
                method = "knn", data = trn_pos,
                trControl = trainControl(method = 'cv', number = 5),
                tuneLength = 20)

## Finding the best k value for KNN --------------------------------------------
knn_pos$bestTune

## Plot accuracy vs K values
plot(knn_pos)

## Writing a custom plot function for KNN models -------------------------------
plot_knn_results <- function(fit_knn) {
  ggplot(fit_knn$results, aes(x = k, y = Accuracy)) +
    geom_bar(stat = "identity", color = "white", fill="steelblue")+
    theme_minimal() +
    scale_x_discrete("value of k", limits = fit_knn$results$k) +
    scale_y_continuous("accuracy")
}

## Creating the KNN plot -------------------------------------------------------
plot_knn_results(knn_pos)

## KNN: Prediction and Evaluation ----------------------------------------------
knnPredict <- predict(knn_pos, newdata = tst_pos )

confusionMatrix(knnPredict, tst_pos$Pos ) 
mean(knnPredict == tst_pos$Pos)  

# 3. LOGISTIC REGRESSION -------------------------------------------------------

## Creating a copy of the dataset ----------------------------------------------
Players2 <- Players

## Checking the mean of the PTS variable and plotting it -----------------------
mean(Players2$PTS)

ggplot(data = Players2, aes(x = PTS))+
  geom_histogram(stat="bin", color = "white", fill="steelblue")+
  theme_minimal() +
  labs(x = 'Points', y = 'Count')

## Creating a binary variable for points (PTS) ---------------------------------
Players2$PTS[Players2$PTS < mean(Players2$PTS)]<- 0
Players2$PTS[Players2$PTS >= mean(Players2$PTS)]<- 1
Players2$PTS <- as.factor(Players2$PTS)

## Setting a seed --------------------------------------------------------------
set.seed(1)

## Partitioning the dataset for training and testing ---------------------------
trn_index2 = createDataPartition(y = Players2$PTS, p = 0.70, list = FALSE)
trn_pts = Players2[trn_index2, ]
tst_pts = Players2[-trn_index2, ]

## Setting a seed and training the model (Logistic Regression) -----------------
set.seed(1)
logistic_pts = train(PTS ~ FGM + FGA + X3PM + X3PA + 
                       FTM + FTA + OREB + DREB + AST + STL + 
                       BLK + TOV + PF + Age + Height + Weight , 
                     method = "glm", family = binomial(link = "logit"),
                     data = trn_pts,
                     trControl = trainControl(method = 'cv', number = 5))

logistic_pts

## Summarizing the model -------------------------------------------------------
summary(logistic_pts)
exp(coef(logistic_pts$finalModel))

## Logistic Regression: Prediction and Evaluation ------------------------------
logistic_predict <-predict(logistic_pts, newdata = tst_pts)
confusionMatrix(tst_pts$PTS, data = logistic_predict)

# 4. RANDOM FOREST (OWN CHOICE) ------------------------------------------------

## Creating a copy of the dataset ----------------------------------------------
Players3 <- Players

## Setting a seed --------------------------------------------------------------
set.seed(2020)

## Splitting data into train and testset ---------------------------------------
n <- nrow(Players3)
n_train <- round(0.7 * n) 
train_indices <- sample(1:n, n_train)
trn_rf <- Players3[train_indices, ]  
tst_rf <- Players3[-train_indices, ]

## Setting a seed and training the model (Random Forest) -----------------------
set.seed(2020)
rf_pos <- randomForest(Pos ~ PTS + FGM + FGA + X3PM + X3PA +
                        FTM + FTA + OREB + DREB + AST + STL + 
                        BLK + TOV + PF + Age + Height + Weight,
                      data = trn_rf)

rf_pos

## Random Forest: Prediction and Evaluation ------------------------------------
tst_rf$pred <- predict(rf_pos, tst_rf)
tst_rf$pred <- as.factor(tst_rf$pred)
confusionMatrix(tst_rf$pred, tst_rf$Pos)

# RANDOM FOREST BINARY CLASSIFICATION ------------------------------------------

## Creating a copy of the dataset ----------------------------------------------
Players4 <- Players

## Creating a binary variable for points (PTS) ---------------------------------
Players4$PTS[Players4$PTS < mean(Players4$PTS)]<- 0
Players4$PTS[Players4$PTS >= mean(Players4$PTS)]<- 1
Players4$PTS <- as.factor(Players4$PTS)

## Setting a seed --------------------------------------------------------------
set.seed(2020)

## Partitioning the dataset for training and testing ---------------------------
trn_index_rf = createDataPartition(y = Players4$PTS, p = 0.70, list = FALSE)
trn_pts_rf = Players4[trn_index_rf, ]
tst_pts_rf = Players4[-trn_index_rf, ]

## Setting a seed and training the model (Random Forest, binary) ---------------
set.seed(2020)
rf_pts <- randomForest(PTS ~ FGM + FGA + X3PM + X3PA + 
                         FTM + FTA + OREB + DREB + AST + STL + 
                         BLK + TOV + PF + Age + Height + Weight,
                       data = trn_pts_rf)

rf_pts

## Random Forest, binary: Prediction and Evaluation
tst_pts_rf$pred <- predict(rf_pts, tst_pts_rf)
tst_pts_rf$pred <- as.factor(tst_pts_rf$pred)
confusionMatrix(tst_pts_rf$pred, tst_pts_rf$PTS)
