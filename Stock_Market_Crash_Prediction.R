library(dplyr) 
library(haven)
library(ggplot2) 
library(caret) 
library(pROC) 
library(randomForest)

## Read in data
mret <- read_sas("mret7023.sas7bdat")
as.data.frame(mret)

## Add year column
mret <- mret %>%
  mutate(year = as.numeric(format(DATE, "%Y")))
as.data.frame(mret)

## Last 5 Years (2019, 2020, 2021, 2022, 2023)
mret <- mret %>%
  filter(year >= 2019)

## Add CRASH variable
mret <- mret %>%
  mutate(
    CRASH = if_else(RET < -0.08, 1, 0)
  )

## Group by PERMNO and arrange by DATE
mret <- mret %>%
  group_by(PERMNO) %>%
  arrange(DATE)

## Create variables and then lag all predictors
mret <- mret %>%
  mutate(
    mcap = abs(PRC) * SHROUT,
    volatility = abs(RET),
    turnover = if_else(SHROUT > 0, VOL / SHROUT, 0),
    
    # 1-month-lagged predictors
    RET1 = lag(RET, n = 1),
    mcap1 = lag(mcap, n = 1),
    volatility1 = lag(volatility, n = 1),
    turnover1 = lag(turnover, n = 1)
  )

## Remove rows where RET1 is NA and ungroup
mret <- mret %>%
  filter(!is.na(RET1)) %>%
  ungroup()

## Select variables
mret <- mret %>%
  select(CRASH, RET1, mcap1, volatility1, turnover1) %>%
  mutate(CRASH = as.factor(CRASH)) %>%
  na.omit()

## Identify variables with limited variation
nzv <- nearZeroVar(mret, freqCut = 70/30, uniqueCut = 10)
print(nzv)

## Data Splitting
set.seed(42)
inTrain <- createDataPartition(mret$CRASH, p = .70, list = FALSE)
print(inTrain[1:10])
Train <- mret[ inTrain,]
Test  <- mret[-inTrain,]

## Train the GLM (Logistic Regression) Model
glm_model <- train(
  CRASH ~ RET1 + mcap1 + volatility1 + turnover1, 
  data = Train, 
  method = "glm",
  family = binomial(link = 'logit')
)

# --- Print the Model Summary ---
print("--- GLM Model Training ---")
print(glm_model)

## Test the GLM Model
glm_probs <- predict(glm_model, Test, type = "prob")

glm_probs_crash <- glm_probs$"1"
par(pty = "s") 

roc(Test$CRASH, glm_probs_crash, plot = TRUE, legacy.axes = TRUE, 
    col = "#377eb8", print.auc = TRUE)


## Train the Random Forest (RF) Model
rf_model <- train(
  CRASH ~ RET1 + mcap1 + volatility1 + turnover1, 
  data = Train, 
  method = "rf"
)

# --- Print the Model Summary ---
print("--- Random Forest Model Training ---")
print(rf_model)

## Test the Random Forest (RF) Model
print("--- Predicting with RF model ---")
rf_probs <- predict(rf_model, Test, type = "prob")
rf_probs_crash <- rf_probs$"1"

par(pty = "s") 

plot(roc(Test$CRASH, glm_probs_crash), col = "#377eb8", legacy.axes = TRUE, 
     print.auc = TRUE)

rf_roc <- roc(Test$CRASH, rf_probs_crash, plot = TRUE, add = TRUE, 
              col = "#4daf4a", print.auc = TRUE, print.auc.y = 0.4)

legend("bottomright", legend = c("GLM Model", "RF Model"), 
       col = c("#377eb8", "#4daf4a"), lwd = 3)


## Train the Decision Tree (rpart) Model
rpart_model <- train(CRASH ~ RET1 + mcap1 + volatility1 + turnover1, 
  data = Train, method = "rpart")

# --- Print the Model Summary ---
print("--- Decision Tree Model Training Complete ---")
print(rpart_model)


## Test the Decision Tree (rpart) Model
rpart_probs <- predict(rpart_model, Test, type = "prob")
rpart_probs_crash <- rpart_probs$"1"

par(pty = "s")

plot(roc(Test$CRASH, glm_probs_crash),col = "#377eb8", legacy.axes = TRUE, 
     print.auc = TRUE)

plot(rf_roc, add = TRUE, col = "#4daf4a", print.auc = TRUE, print.auc.y = 0.4)

rpart_roc <- roc(Test$CRASH, rpart_probs_crash, plot = TRUE, add = TRUE, 
  col = "#e41a1c", print.auc = TRUE, print.auc.y = 0.3)

legend("bottomright", legend = c("GLM Model", "RF Model", "Tree Model"), 
       col = c("#377eb8", "#4daf4a", "#e41a1c"), lwd = 3)


## Print the final AUCs
print("--- Printing AUC Scores ---")
glm_roc <- roc(Test$CRASH, glm_probs_crash)
print(paste("GLM Model AUC:", glm_roc$auc))

print(paste("RF Model AUC:", rf_roc$auc))

print(paste("Tree Model AUC:", rpart_roc$auc))


