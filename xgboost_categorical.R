library(RODBC)
library(tidyr)
library(dplyr)
library(tidyverse)
library(survival)
library(car)
library(flexsurv)
library(KMsurv)
library(e1071)
library(rms)
library(MASS)
library(survminer)
library(glmnet)
library(MLmetrics)


analysis_df_latest_location <- "C:/Analyses/premier_customers_prediction/all_sequential/envs/analysis_df_latest"

churn_logit_latest_location <- "C:/Analyses/premier_customers_prediction/all_sequential/envs/churn_logit_latest"
churn_logit_old_location <- "C:/Analyses/premier_customers_prediction/all_sequential/envs/churn_logit_old"


alpha_levels <- c(0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1)
premier_threshold <- 100

latest_analysis_df_filename <- paste(
  analysis_df_latest_location,
  list.files(analysis_df_latest_location, pattern = 'Rdata')[[1]],
  sep = "/"
)

load(latest_analysis_df_filename)


# Threshold metric functions

fpr_from_class_threshold <- function(class_threshold, model_response, y_values){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  CM <- ConfusionMatrix(model_class,y_values)
  fpr <- CM[1,2]/(CM[1,1]+CM[1,2])
  fpr
}

tpr_from_class_threshold <- function(class_threshold, model_response, y_values){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  CM <- ConfusionMatrix(model_class,y_values)
  tpr <- CM[2,2]/(CM[2,1]+CM[2,2])
  tpr
}


f1_from_class_threshold <- function(class_threshold, model_response, y_values){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  CM <- ConfusionMatrix(model_class,y_values)
  recall <- CM[2,2]/(CM[2,1]+CM[2,2])
  precision <- CM[2,2]/(CM[1,2]+CM[2,2])
  f1 <- (2*precision*recall)/(precision+recall)
  f1
}

f1_baseline_from_class_threshold <- function(class_threshold, model_response, y_values){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  true_1 <- mean(y_values)
  true_0 <- 1 - true_1
  pred_1 <- mean(model_class)
  pred_0 <- 1- pred_1
  true_0_pred_0 <- true_0 * pred_0
  true_0_pred_1 <- true_0 * pred_1
  true_1_pred_0 <- true_1 * pred_0
  true_1_pred_1 <- true_1 * pred_1
  baseline_recall <- true_1_pred_1/true_1
  baseline_precision <- true_1_pred_1/pred_1
  baseline_f1 <- (2*baseline_precision*baseline_recall)/(baseline_precision+baseline_recall)
  baseline_f1
}

gmean_from_class_threshold <- function(class_threshold, model_response, y_values){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  CM <- ConfusionMatrix(model_class,y_values)
  recall <- CM[2,2]/(CM[2,1]+CM[2,2])
  specificity <- CM[1,1]/(CM[1,1]+CM[1,2])
  gmean <- sqrt(recall*specificity)
  gmean
}

gmean_baseline_from_class_threshold <- function(class_threshold, model_response, y_values){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  true_1 <- mean(y_values)
  true_0 <- 1 - true_1
  pred_1 <- mean(model_class)
  pred_0 <- 1- pred_1
  true_0_pred_0 <- true_0 * pred_0
  true_0_pred_1 <- true_0 * pred_1
  true_1_pred_0 <- true_1 * pred_0
  true_1_pred_1 <- true_1 * pred_1
  baseline_recall <- true_1_pred_1/true_1
  baseline_specificity <- true_0_pred_0/pred_0
  baseline_gmean <- sqrt(baseline_recall*baseline_specificity)
  baseline_gmean
}


recall_from_class_threshold <- function(class_threshold, model_response, y_values){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  CM <- ConfusionMatrix(model_class,y_values)
  recall <- CM[2,2]/(CM[2,1]+CM[2,2])
  recall
}


cost_from_class_threshold <- function(class_threshold, model_response, y_values, fp_cost){
  model_class <- ifelse(model_response < class_threshold, 0,1)
  CM <- ConfusionMatrix(model_class,y_values)
  cost <- CM[1,2]*fp_cost + (1-fp_cost)*CM[2,1]
  cost
}



#####Logit Churn Model


regression_df_train_full <- regression_df %>%
  left_join(dep_vars_df)


train_full_customers <- regression_df_train_full %>%
  dplyr::select(parent_id.1) %>%
  unique() %>%
  as.matrix()

training_percentage <- 0.8


train_customers <- train_full_customers[sample(c(1:nrow(train_full_customers)),
                                               round(training_percentage*nrow(train_full_customers))),]


test_customers <- train_full_customers[!(train_full_customers %in% train_customers)]



regression_df_train <- regression_df_train_full %>%
  filter(parent_id.1 %in% train_customers)

regression_df_test <- regression_df_train_full %>%
  filter(parent_id.1 %in% test_customers)






train_full_logistic_df_y <- regression_df_train_full %>%
  dplyr::select(churn) %>%
  unlist() %>%
  as.vector()

train_full_logistic_df_x <- regression_df_train_full %>%
  dplyr::select(-one_of(colnames(dep_vars_df))) %>%
  as.matrix() 


train_logistic_df_y <- regression_df_train %>%
  dplyr::select(churn) %>%
  unlist() %>%
  as.vector()

train_logistic_df_x <- regression_df_train %>%
  dplyr::select(-one_of(colnames(dep_vars_df))) %>%
  as.matrix() 



test_logistic_df_y <- regression_df_test %>%
  dplyr::select(churn) %>%
  unlist() %>%
  as.vector()

test_logistic_df_x <- regression_df_test %>%
  dplyr::select(-one_of(colnames(dep_vars_df))) %>%
  as.matrix()





nfold <- 10

train_full_foldid <- replicate(1, n = floor(length(train_full_logistic_df_y)/nfold))
for (j in 2:nfold) {
  current_fold <- replicate(j, n = floor(length(train_full_logistic_df_y)/nfold))
  train_full_foldid <- c(train_full_foldid,current_fold)
}

if(length(train_full_foldid) == length(train_full_logistic_df_y)) {train_full_foldid <- sample(train_full_foldid)} else {
  current_fold <- sample(1:nfold, size = length(train_full_logistic_df_y)-length(train_full_foldid), replace = TRUE)
  train_full_foldid <- sample(c(train_full_foldid, current_fold))
}


train_foldid <- replicate(1, n = floor(length(train_logistic_df_y)/nfold))
for (j in 2:nfold) {
  current_fold <- replicate(j, n = floor(length(train_logistic_df_y)/nfold))
  train_foldid <- c(train_foldid,current_fold)
}

if(length(train_foldid) == length(train_logistic_df_y)) {train_foldid <- sample(train_foldid)} else {
  current_fold <- sample(1:nfold, size = length(train_logistic_df_y)-length(train_foldid), replace = TRUE)
  train_foldid <- sample(c(train_foldid, current_fold))
}

test_foldid <- replicate(1, n = floor(length(test_logistic_df_y)/nfold))
for (j in 2:nfold) {
  current_fold <- replicate(j, n = floor(length(test_logistic_df_y)/nfold))
  test_foldid <- c(test_foldid,current_fold)
}

if(length(test_foldid) == length(test_logistic_df_y)) {test_foldid <- sample(test_foldid)} else {
  current_fold <- sample(1:nfold, size = length(test_logistic_df_y)-length(test_foldid), replace = TRUE)
  test_foldid <- sample(c(test_foldid, current_fold))
}





train_dev_baseline <- vector('numeric', length = length(alpha_levels))
train_cvfits <- list()
train_lambda_type <- vector('numeric', length = length(alpha_levels))
train_cv_deviance <- vector('numeric', length = length(alpha_levels))
train_deviance_ratio <- vector('numeric', length = length(alpha_levels))

train_full_dev_baseline <- vector('numeric', length = length(alpha_levels))
train_full_cvfits <- list()
train_full_lambda_type <- vector('numeric', length = length(alpha_levels))
train_full_cv_deviance <- vector('numeric', length = length(alpha_levels))
train_full_deviance_ratio <- vector('numeric', length = length(alpha_levels))

for (i in 1:length(alpha_levels)) {
  print(i)
  
  train_full_cvfit <- cv.glmnet(x = train_full_logistic_df_x, 
                                y = train_full_logistic_df_y, 
                                family = "binomial", 
                                type.measure = "deviance", 
                                nfolds = nfold,
                                alpha = alpha_levels[[i]],
                                foldid = train_full_foldid)
  
  train_cvfit <- cv.glmnet(x = train_logistic_df_x, 
                           y = train_logistic_df_y, 
                           family = "binomial", 
                           type.measure = "deviance", 
                           nfolds = nfold,
                           alpha = alpha_levels[[i]],
                           foldid = train_foldid)
  
  
  train_full_cvfits[[i]] <- train_full_cvfit
  train_cvfits[[i]] <- train_cvfit
  
  
  train_full_glmnetfit <- train_full_cvfit$glmnet.fit
  train_glmnetfit <- train_cvfit$glmnet.fit
  
  
  train_full_dev_baseline[i] <- train_full_glmnetfit$nulldev/length(train_full_logistic_df_y)
  train_dev_baseline[i] <- train_glmnetfit$nulldev/length(train_logistic_df_y)
  
  train_full_cvfit_values <- print(train_full_cvfit)
  train_cvfit_values <- print(train_cvfit)
  
  
  train_full_cv_deviance_lambdamin <- train_full_cvfit_values %>% 
    filter(Lambda == train_full_cvfit$lambda.min) %>% 
    dplyr::select(Measure) %>% 
    min()
  train_cv_deviance_lambdamin <- train_cvfit_values %>% 
    filter(Lambda == train_cvfit$lambda.min) %>% 
    dplyr::select(Measure) %>% 
    min()
  
  
  train_full_cv_deviance_lambda1se <- train_full_cvfit_values %>% 
    filter(Lambda == train_full_cvfit$lambda.1se) %>% 
    dplyr::select(Measure) %>% 
    min()
  train_cv_deviance_lambda1se <- train_cvfit_values %>% 
    filter(Lambda == train_cvfit$lambda.1se) %>% 
    dplyr::select(Measure) %>% 
    min()
  
  
  
  
  if (train_full_cv_deviance_lambdamin < train_full_cv_deviance_lambda1se) {
    train_full_lambda_type[i] <- 'lambda.min'
    train_full_cv_deviance[i] <- train_full_cv_deviance_lambdamin
  } else {
    train_full_lambda_type[i] <- 'lambda.1se'
    train_full_cv_deviance[i] <- train_full_cv_deviance_lambda1se
  }
  if (train_cv_deviance_lambdamin < train_cv_deviance_lambda1se) {
    train_lambda_type[i] <- 'lambda.min'
    train_cv_deviance[i] <- train_cv_deviance_lambdamin
  } else {
    train_lambda_type[i] <- 'lambda.1se'
    train_cv_deviance[i] <- train_cv_deviance_lambda1se
  }
  
  train_full_deviance_ratio[i] <- train_full_cv_deviance[i]/train_full_dev_baseline[i]
  train_deviance_ratio[i] <- train_cv_deviance[i]/train_dev_baseline[i]
}

train_full_optimal_fit <- train_full_cvfits[[which.min(train_full_deviance_ratio)]]
train_optimal_fit <- train_cvfits[[which.min(train_deviance_ratio)]]

train_full_optimal_fit_values <- print(train_full_optimal_fit)
train_optimal_fit_values <- print(train_optimal_fit)

train_full_optimal_lambda_type <- train_full_lambda_type[[which.min(train_full_deviance_ratio)]]
train_optimal_lambda_type <- train_lambda_type[[which.min(train_deviance_ratio)]]

train_full_optimal_lambda <- ifelse(train_full_optimal_lambda_type == 'lambda.min', 
                                    train_full_optimal_fit$lambda.min, 
                                    train_full_optimal_fit$lambda.1se)

train_optimal_lambda <- ifelse(train_optimal_lambda_type == 'lambda.min', 
                               train_optimal_fit$lambda.min, 
                               train_optimal_fit$lambda.1se)

train_full_optimal_alpha <- alpha_levels[[which.min(train_full_deviance_ratio)]]
train_optimal_alpha <- alpha_levels[[which.min(train_deviance_ratio)]]

if(train_full_optimal_lambda_type == 'lambda.min') {
  train_full_optimal_nonzero <- train_full_optimal_fit_values %>% 
    filter(Lambda == train_full_optimal_fit$lambda.min) %>% 
    dplyr::select(Nonzero) %>% 
    min()
} else {
  train_full_optimal_nonzero <- train_full_optimal_fit_values %>% 
    filter(Lambda == train_full_optimal_fit$lambda.1se) %>% 
    dplyr::select(Nonzero) %>% 
    min()
}

if(train_optimal_lambda_type == 'lambda.min') {
  train_optimal_nonzero <- train_optimal_fit_values %>% 
    filter(Lambda == train_optimal_fit$lambda.min) %>% 
    dplyr::select(Nonzero) %>% 
    min()
} else {
  train_optimal_nonzero <- train_optimal_fit_values %>% 
    filter(Lambda == train_optimal_fit$lambda.1se) %>% 
    dplyr::select(Nonzero) %>% 
    min()
}


if(train_full_optimal_lambda_type == 'lambda.min') {
  train_full_optimal_logloss_model <- LogLoss(predict(train_full_optimal_fit,
                                                      newx = train_full_logistic_df_x,
                                                      s=train_full_cvfit$lambda.min,
                                                      type = 'response'), train_full_logistic_df_y)
} else {
  train_full_optimal_logloss_model <- LogLoss(predict(train_full_optimal_fit,
                                                      newx = train_full_logistic_df_x,
                                                      s=train_full_cvfit$lambda.1se,
                                                      type = 'response'), train_full_logistic_df_y)
}

if(train_optimal_lambda_type == 'lambda.min') {
  train_optimal_logloss_model <- LogLoss(predict(train_optimal_fit,
                                                 newx = train_logistic_df_x,
                                                 s=train_cvfit$lambda.min,
                                                 type = 'response'), train_logistic_df_y)
} else {
  train_optimal_logloss_model <- LogLoss(predict(train_optimal_fit,
                                                 newx = train_logistic_df_x,
                                                 s=train_cvfit$lambda.1se,
                                                 type = 'response'), train_logistic_df_y)
}

if(train_optimal_lambda_type == 'lambda.min') {
  test_optimal_logloss_model <- LogLoss(predict(train_optimal_fit,
                                                newx = test_logistic_df_x,
                                                s=train_cvfit$lambda.min,
                                                type = 'response'), test_logistic_df_y)
} else {
  test_optimal_logloss_model <- LogLoss(predict(train_optimal_fit,
                                                newx = test_logistic_df_x,
                                                s=train_cvfit$lambda.1se,
                                                type = 'response'), train_logistic_df_y)
}

train_full_optimal_cv_deviance <- train_full_cv_deviance[[which.min(train_full_deviance_ratio)]]
train_optimal_cv_deviance <- train_cv_deviance[[which.min(train_deviance_ratio)]]

train_full_baseline_predictions <- rep.int(mean(train_full_logistic_df_y),length(train_full_logistic_df_y))
train_baseline_predictions <- rep.int(mean(train_logistic_df_y),length(train_logistic_df_y))
test_baseline_predictions <- rep.int(mean(test_logistic_df_y),length(test_logistic_df_y))


train_full_logloss_baseline <- LogLoss(train_full_baseline_predictions, train_full_logistic_df_y)
train_logloss_baseline <- LogLoss(train_baseline_predictions, train_logistic_df_y)
test_logloss_baseline <- LogLoss(test_baseline_predictions, test_logistic_df_y)




train_full_optimal_coefficients <- coefficients(train_full_optimal_fit, 
                                                s=ifelse(train_full_optimal_lambda_type == 'lambda.min', 
                                                         'lambda.min',
                                                         'lambda.1se')) %>% 
  as.matrix()

train_optimal_coefficients <- coefficients(train_optimal_fit,
                                           s=ifelse(train_optimal_lambda_type == 'lambda.min',
                                                    'lambda.min',
                                                    'lambda.1se')) %>% 
  as.matrix()


train_full_model_response <- predict(train_full_optimal_fit, newx = train_full_logistic_df_x,
                                     s=ifelse(train_full_lambda_type[[which.min(train_full_deviance_ratio)]] == 'lambda.min', 
                                              'lambda.min', 
                                              'lambda.1se'),
                                     type = 'response')

train_model_response <- predict(train_optimal_fit, newx = train_logistic_df_x,
                                s=ifelse(train_lambda_type[[which.min(train_deviance_ratio)]] == 'lambda.min', 
                                         'lambda.min', 
                                         'lambda.1se'),
                                type = 'response')

test_model_response <- predict(train_optimal_fit, newx = test_logistic_df_x,
                               s=ifelse(train_lambda_type[[which.min(train_deviance_ratio)]] == 'lambda.min', 
                                        'lambda.min', 
                                        'lambda.1se'),
                               type = 'response')



train_full_customers <- regression_df_train_full %>% dplyr::select(parent_id.1)
train_customers <- regression_df_train %>% dplyr::select(parent_id.1)
test_customers <- regression_df_test %>% dplyr::select(parent_id.1)



train_full_min_response <- min(train_full_model_response)
train_min_response <- min(train_model_response)

train_full_max_response <- max(train_full_model_response)
train_max_response <- max(train_model_response)


train_full_model_response_ordered <- train_full_model_response[train_full_model_response!=train_full_min_response & 
                                                                 train_full_model_response!=train_full_max_response] %>%
  unique() %>%
  sort(decreasing=TRUE)

train_model_response_ordered <- train_model_response[train_model_response!=train_min_response &
                                                       train_model_response!=train_max_response] %>%
  unique() %>%
  sort(decreasing=TRUE)

#Threshold metric vectors

train_full_fpr_score <- vector("numeric", length(train_full_model_response_ordered))
train_full_tpr_score <- vector("numeric", length(train_full_model_response_ordered))
train_full_distance_from_top_left_ROC <- vector("numeric", length(train_full_model_response_ordered))

train_full_f1_score <- vector("numeric", length(train_full_model_response_ordered))
train_full_f1_baseline <- vector("numeric", length(train_full_model_response_ordered))
train_full_f1_ratio <- vector("numeric", length(train_full_model_response_ordered))

train_full_gmean_score <- vector("numeric", length(train_full_model_response_ordered))
train_full_gmean_baseline <- vector("numeric", length(train_full_model_response_ordered))
train_full_gmean_ratio <- vector("numeric", length(train_full_model_response_ordered))

train_fpr_score <- vector("numeric", length(train_model_response_ordered))
train_tpr_score <- vector("numeric", length(train_model_response_ordered))
train_distance_from_top_left_ROC <- vector("numeric", length(train_model_response_ordered))

train_f1_score <- vector("numeric", length(train_model_response_ordered))
train_f1_baseline <- vector("numeric", length(train_model_response_ordered))
train_f1_ratio <- vector("numeric", length(train_model_response_ordered))

train_gmean_score <- vector("numeric", length(train_model_response_ordered))
train_gmean_baseline <- vector("numeric", length(train_model_response_ordered))
train_gmean_ratio <- vector("numeric", length(train_model_response_ordered))

# Generate threshold metrics dataset


for (j in 1:length(train_full_model_response_ordered)) {
  print(j)
  train_full_fpr_score[[j]] <- fpr_from_class_threshold(train_full_model_response_ordered[[j]],
                                                        train_full_model_response,
                                                        train_full_logistic_df_y)
  
  train_full_tpr_score[[j]] <- tpr_from_class_threshold(train_full_model_response_ordered[[j]],
                                                        train_full_model_response, 
                                                        train_full_logistic_df_y)
  
  train_full_distance_from_top_left_ROC[[j]] <- sqrt((0-train_full_fpr_score[[j]])^2 + (1-train_full_tpr_score[[j]])^2)
  
  
  
  train_full_f1_score[[j]] <- f1_from_class_threshold(train_full_model_response_ordered[[j]], 
                                                      train_full_model_response, 
                                                      train_full_logistic_df_y)
  
  train_full_f1_baseline[[j]] <- f1_baseline_from_class_threshold(train_full_model_response_ordered[[j]], 
                                                                  train_full_model_response, 
                                                                  train_full_logistic_df_y)
  
  train_full_f1_ratio[[j]] <- train_full_f1_score[[j]]/train_full_f1_baseline[[j]]
  
  train_full_gmean_score[[j]] <- gmean_from_class_threshold(train_full_model_response_ordered[[j]], 
                                                            train_full_model_response, 
                                                            train_full_logistic_df_y)
  
  train_full_gmean_baseline[[j]] <- gmean_baseline_from_class_threshold(train_full_model_response_ordered[[j]],
                                                                        train_full_model_response,
                                                                        train_full_logistic_df_y)
  
  train_full_gmean_ratio[[j]] <- train_full_gmean_score[[j]]/train_full_gmean_baseline[[j]]
  
}

for (j in 1:length(train_model_response_ordered)) {
  print(j)
  train_fpr_score[[j]] <- fpr_from_class_threshold(train_model_response_ordered[[j]],
                                                   train_model_response,
                                                   train_logistic_df_y)
  
  train_tpr_score[[j]] <- tpr_from_class_threshold(train_model_response_ordered[[j]],
                                                   train_model_response, 
                                                   train_logistic_df_y)
  
  train_distance_from_top_left_ROC[[j]] <- sqrt((0-train_fpr_score[[j]])^2 + (1-train_tpr_score[[j]])^2)
  
  
  
  train_f1_score[[j]] <- f1_from_class_threshold(train_model_response_ordered[[j]], 
                                                 train_model_response, 
                                                 train_logistic_df_y)
  
  train_f1_baseline[[j]] <- f1_baseline_from_class_threshold(train_model_response_ordered[[j]], 
                                                             train_model_response, 
                                                             train_logistic_df_y)
  
  train_f1_ratio[[j]] <- train_f1_score[[j]]/train_f1_baseline[[j]]
  
  train_gmean_score[[j]] <- gmean_from_class_threshold(train_model_response_ordered[[j]], 
                                                       train_model_response, 
                                                       train_logistic_df_y)
  
  train_gmean_baseline[[j]] <- gmean_baseline_from_class_threshold(train_model_response_ordered[[j]],
                                                                   train_model_response,
                                                                   train_logistic_df_y)
  
  train_gmean_ratio[[j]] <- train_gmean_score[[j]]/train_gmean_baseline[[j]]
  
}

#Optimal Thresholds

train_full_optimal_class_threshold_distance_from_top_left_ROC <- 
  train_full_model_response_ordered[[which.min(train_full_distance_from_top_left_ROC)]]
train_full_optimal_class_threshold_f1_score <- train_full_model_response_ordered[[which.max(train_full_f1_score)]]
train_full_optimal_class_threshold_f1_ratio <- train_full_model_response_ordered[[which.max(train_full_f1_ratio)]]
train_full_optimal_class_threshold_gmean_score <- train_full_model_response_ordered[[which.max(train_full_gmean_score)]]
train_full_optimal_class_threshold_gmean_ratio <- train_full_model_response_ordered[[which.max(train_full_gmean_ratio)]]

train_optimal_class_threshold_distance_from_top_left_ROC <- 
  train_model_response_ordered[[which.min(train_distance_from_top_left_ROC)]]
train_optimal_class_threshold_f1_score <- train_model_response_ordered[[which.max(train_f1_score)]]
train_optimal_class_threshold_f1_ratio <- train_model_response_ordered[[which.max(train_f1_ratio)]]
train_optimal_class_threshold_gmean_score <- train_model_response_ordered[[which.max(train_gmean_score)]]
train_optimal_class_threshold_gmean_ratio <- train_model_response_ordered[[which.max(train_gmean_ratio)]]

train_full_optimal_class_threshold <- median(c(
  train_full_optimal_class_threshold_distance_from_top_left_ROC,
  train_full_optimal_class_threshold_f1_score,
  train_full_optimal_class_threshold_f1_ratio,
  train_full_optimal_class_threshold_gmean_score,
  train_full_optimal_class_threshold_gmean_ratio
)
)

train_optimal_class_threshold <- median(c(
  train_optimal_class_threshold_distance_from_top_left_ROC,
  train_optimal_class_threshold_f1_score,
  train_optimal_class_threshold_f1_ratio,
  train_optimal_class_threshold_gmean_score,
  train_optimal_class_threshold_gmean_ratio
)
)

train_full_predictions <-
  cbind(train_full_customers, 
        class = regression_df_train_full$churn,
        train_full_model_response, 
        train_full_optimal_class_threshold = replicate(train_full_optimal_class_threshold, n=length(train_full_model_response))) %>%
  mutate(train_full_optimal_class_prediction = ifelse(train_full_model_response>=train_full_optimal_class_threshold,1,0))
colnames(train_full_predictions)[[3]] <- 'prediction'

train_predictions <-
  cbind(train_customers, 
        class = regression_df_train$churn,
        train_model_response, 
        train_optimal_class_threshold = replicate(train_optimal_class_threshold, n=length(train_model_response))) %>%
  mutate(train_optimal_class_prediction = ifelse(train_model_response>=train_optimal_class_threshold,1,0))
colnames(train_predictions)[[3]] <- 'prediction'

test_predictions <-
  cbind(test_customers, 
        class = regression_df_test$churn,
        test_model_response, 
        train_optimal_class_threshold = replicate(train_optimal_class_threshold, n=length(test_model_response))) %>%
  mutate(test_optimal_class_prediction = ifelse(test_model_response>=train_optimal_class_threshold,1,0))
colnames(test_predictions)[[3]] <- 'prediction'



#### Output
ConfusionMatrix(train_predictions$train_optimal_class_prediction,
                train_predictions$class)

ConfusionMatrix(test_predictions$test_optimal_class_prediction,
                test_predictions$class)


ConfusionMatrix(train_full_predictions$train_full_optimal_class_prediction,
                train_full_predictions$class)




train_optimal_lambda_type
train_optimal_lambda
train_optimal_alpha
train_optimal_nonzero
train_optimal_logloss_model
train_logloss_baseline
train_optimal_cv_deviance
train_dev_baseline %>% first()

test_optimal_logloss_model
test_logloss_baseline

train_full_optimal_lambda_type
train_full_optimal_lambda
train_full_optimal_alpha
train_full_optimal_nonzero
train_full_optimal_logloss_model
train_full_logloss_baseline
train_full_optimal_cv_deviance
train_full_dev_baseline %>% first()

train_full_optimal_coefficients
train_optimal_coefficients

getwd()

write.csv(train_full_optimal_coefficients, 'train_full_optimal_coefficients.csv')
write.csv(train_optimal_coefficients, 'train_optimal_coefficients.csv')

#####Save Env

churn_logit_name <- paste(
  churn_logit_latest_location,
  "/churn_logit_",
  gsub("-","_", Sys.Date()),
  "_uk.Rdata",
  sep = ""
)

current_latest_file <- list.files(churn_logit_latest_location, pattern = 'Rdata')

if(length(current_latest_file) == 0){
  save(list=c("train_optimal_class_threshold", 
              "train_optimal_fit"),
       file = churn_logit_name)
} else {
  latest_churn_logit_filename <- paste(
    churn_logit_latest_location,
    list.files(churn_logit_latest_location, pattern = 'Rdata')[[1]],
    sep = "/"
  )
  file.move(latest_churn_logit_filename, churn_logit_old_location)
  save(list=c("train_optimal_class_threshold", 
              "train_optimal_fit"),
       file = churn_logit_name)
}


##save.image("C:/Analyses/premier_customers_prediction/final_output.RData")
##
##unique(df_transactions_copy$PAYMENT_NETWORK_GROUP)
##
###### train_full LOOCV with parameters from cv.glmnet() on train data
##
##loocv_validation_predictions <- vector('numeric', length = nrow(train_full_logistic_df_x))
##
##for (i in 1:nrow(train_full_logistic_df_x)) {
##  print(i)
##  fit <- glmnet(x = train_full_logistic_df_x[-i,], 
##                y = train_full_logistic_df_y[-i], 
##                family = "binomial",
##                alpha = train_optimal_alpha,
##                lambda = train_optimal_lambda)
##  predictions <-  predict(fit,
##                          newx = as.matrix(train_full_logistic_df_x),
##                          s=train_optimal_lambda,
##                          type = 'response')
##  loocv_validation_predictions[i] <- predictions[i]
##  
##}
##
##
##loocv_validation_class_prediction <- ifelse(loocv_validation_predictions>=train_optimal_class_threshold,1,0)
##
##ConfusionMatrix(loocv_validation_class_prediction,
##                train_full_logistic_df_y)
##
##LogLoss(loocv_validation_predictions, train_full_logistic_df_y)
##
##
###### LOOCV Test set and k-fold cv training and validation
##
##kfold_loocv_optimal_fit <- list()
##kfold_loocv_optimal_alpha <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_loocv_optimal_lambda_type <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_loocv_optimal_lambda <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_loocv_optimal_nonzero <- vector("numeric", length = length(train_full_logistic_df_y))
##
##kfold_loocv_train_optimal_logloss_model <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_loocv_train_logloss_baseline <- vector("numeric", length = length(train_full_logistic_df_y))
##
##kfold_loocv_train_optimal_cv_deviance <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_loocv_train_dev_baseline <- vector("numeric", length = length(train_full_logistic_df_y))
##
##kfold_loocv_optimal_threshold <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_loocv_test_predictions <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_loocv_test_class <- vector("numeric", length = length(train_full_logistic_df_y))
##
##start_time <- Sys.time()
##
##for (i in 1:length(train_full_logistic_df_y)) {
##  print(i)
##  kfold_loocv_train_y <- train_full_logistic_df_y[-i]
##  kfold_loocv_train_x <- train_full_logistic_df_x[-i,]
##  kfold_loocv_foldid <- replicate(1, n = floor(length(kfold_loocv_train_y)/nfold))
##  for (j in 2:nfold) {
##    current_fold <- replicate(j, n = floor(length(kfold_loocv_train_y)/nfold))
##    kfold_loocv_foldid <- c(kfold_loocv_foldid,current_fold)
##  }
##  if(length(kfold_loocv_foldid) == length(kfold_loocv_train_y)) {kfold_loocv_foldid <- sample(kfold_loocv_foldid)} else {
##    current_fold <- sample(1:nfold, size = length(kfold_loocv_train_y)-length(kfold_loocv_foldid), replace = TRUE)
##    kfold_loocv_foldid <- sample(c(kfold_loocv_foldid, current_fold))
##  }
##  
##  kfold_loocv_cvfits <- list()
##  kfold_loocv_lambda_type <- vector('numeric', length = length(alpha_levels))
##  kfold_loocv_cv_deviance <- vector('numeric', length = length(alpha_levels))
##  
##  
##  for (k in 1:length(alpha_levels)) {
##    
##    
##    kfold_loocv_cvfit <- cv.glmnet(x = kfold_loocv_train_x, 
##                                   y = kfold_loocv_train_y, 
##                                   family = "binomial", 
##                                   type.measure = "deviance", 
##                                   nfolds = nfold,
##                                   alpha = alpha_levels[[k]],
##                                   foldid = kfold_loocv_foldid)
##    
##    kfold_loocv_cvfits[[k]] <- kfold_loocv_cvfit
##    
##    kfold_loocv_cvfit_values <- print(kfold_loocv_cvfit)
##    
##    
##    kfold_loocv_cv_deviance_lambdamin <- kfold_loocv_cvfit_values %>% 
##      filter(Lambda == kfold_loocv_cvfit$lambda.min) %>% 
##      dplyr::select(Measure) %>% 
##      min()
##    
##    
##    
##    kfold_loocv_cv_deviance_lambda1se <- kfold_loocv_cvfit_values %>% 
##      filter(Lambda == kfold_loocv_cvfit$lambda.1se) %>% 
##      dplyr::select(Measure) %>% 
##      min()
##    
##    
##    if (kfold_loocv_cv_deviance_lambdamin < kfold_loocv_cv_deviance_lambda1se) {
##      kfold_loocv_lambda_type[k] <- 'lambda.min'
##      kfold_loocv_cv_deviance[k] <- kfold_loocv_cv_deviance_lambdamin
##    } else {
##      kfold_loocv_lambda_type[k] <- 'lambda.1se'
##      kfold_loocv_cv_deviance[k] <- kfold_loocv_cv_deviance_lambda1se
##    }
##  }
##  
##  kfold_loocv_optimal_fit[[i]] <- kfold_loocv_cvfits[[which.min(kfold_loocv_cv_deviance)]]
##  kfold_loocv_optimal_alpha[i] <- alpha_levels[[which.min(kfold_loocv_cv_deviance)]]
##  kfold_loocv_optimal_lambda_type[i] <- kfold_loocv_lambda_type[[which.min(kfold_loocv_cv_deviance)]]
##  kfold_loocv_optimal_lambda[i] <- ifelse(kfold_loocv_optimal_lambda_type[i] == 'lambda.min',
##                                          kfold_loocv_optimal_fit[[i]]$lambda.min,
##                                          kfold_loocv_optimal_fit[[i]]$lambda.1se)
##  kfold_loocv_optimal_fit_values <- print(kfold_loocv_optimal_fit[[i]])
##  if(kfold_loocv_optimal_lambda_type[i] == 'lambda.min') {
##    kfold_loocv_optimal_nonzero[i] <- kfold_loocv_optimal_fit_values %>% 
##      filter(Lambda == kfold_loocv_optimal_fit[[i]]$lambda.min) %>% 
##      dplyr::select(Nonzero) %>% 
##      min()
##  } else {
##    kfold_loocv_optimal_nonzero[i]  <- kfold_loocv_optimal_fit_values %>% 
##      filter(Lambda == kfold_loocv_optimal_fit[[i]]$lambda.1se) %>% 
##      dplyr::select(Nonzero) %>% 
##      min()
##  }
##  
##  
##  if(kfold_loocv_optimal_lambda_type[i] == 'lambda.min') {
##    kfold_loocv_train_optimal_logloss_model[i] <- LogLoss(predict(kfold_loocv_optimal_fit[[i]],
##                                                                  newx = kfold_loocv_train_x,
##                                                                  s=kfold_loocv_optimal_fit[[i]]$lambda.min,
##                                                                  type = 'response'), kfold_loocv_train_y)
##  } else {
##    kfold_loocv_train_optimal_logloss_model[i] <- LogLoss(predict(kfold_loocv_optimal_fit[[i]],
##                                                                  newx = kfold_loocv_train_x,
##                                                                  s=kfold_loocv_optimal_fit[[i]]$lambda.1se,
##                                                                  type = 'response'), kfold_loocv_train_y)
##  }
##  
##  kfold_loocv_baseline_predictions <- rep.int(mean(kfold_loocv_train_y),length(kfold_loocv_train_y))
##  kfold_loocv_train_logloss_baseline[i] <- LogLoss(kfold_loocv_baseline_predictions, kfold_loocv_train_y)
##  
##  kfold_loocv_train_optimal_cv_deviance[i] <- kfold_loocv_cv_deviance[[which.min(kfold_loocv_cv_deviance)]]
##  
##  kfold_loocv_optimal_glmnetfit <- kfold_loocv_optimal_fit[[i]]$glmnet.fit
##  kfold_loocv_train_dev_baseline[i] <- kfold_loocv_optimal_glmnetfit$nulldev/length(kfold_loocv_train_y)
##  
##  
##  
##  
##  
##  kfold_loocv_model_response <- predict(kfold_loocv_optimal_fit[[i]], newx = kfold_loocv_train_x,
##                                        s=ifelse(kfold_loocv_optimal_lambda_type[i] == 'lambda.min', 
##                                                 'lambda.min', 
##                                                 'lambda.1se'),
##                                        type = 'response')
##  
##  
##  
##  kfold_loocv_min_response <- min(kfold_loocv_model_response)
##  kfold_loocv_max_response <- max(kfold_loocv_model_response)
##  
##  kfold_loocv_model_response_ordered <- kfold_loocv_model_response[kfold_loocv_model_response!=kfold_loocv_min_response & 
##                                                                     kfold_loocv_model_response!=kfold_loocv_max_response] %>%
##    unique() %>%
##    sort(decreasing=TRUE)
##  
##  kfold_loocv_fpr_score <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  kfold_loocv_tpr_score <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  kfold_loocv_distance_from_top_left_ROC <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  
##  kfold_loocv_f1_score <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  kfold_loocv_f1_baseline <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  kfold_loocv_f1_ratio <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  
##  kfold_loocv_gmean_score <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  kfold_loocv_gmean_baseline <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  kfold_loocv_gmean_ratio <- vector("numeric", length(kfold_loocv_model_response_ordered))
##  
##  for (l in 1:length(kfold_loocv_model_response_ordered)) {
##    
##    kfold_loocv_fpr_score[[l]] <- fpr_from_class_threshold(kfold_loocv_model_response_ordered[[l]],
##                                                           kfold_loocv_model_response,
##                                                           kfold_loocv_train_y)
##    
##    kfold_loocv_tpr_score[[l]] <- tpr_from_class_threshold(kfold_loocv_model_response_ordered[[l]],
##                                                           kfold_loocv_model_response, 
##                                                           kfold_loocv_train_y)
##    
##    kfold_loocv_distance_from_top_left_ROC[[l]] <- sqrt((0-kfold_loocv_fpr_score[[l]])^2 + (1-kfold_loocv_tpr_score[[l]])^2)
##    
##    
##    
##    kfold_loocv_f1_score[[l]] <- f1_from_class_threshold(kfold_loocv_model_response_ordered[[l]], 
##                                                         kfold_loocv_model_response, 
##                                                         kfold_loocv_train_y)
##    
##    kfold_loocv_f1_baseline[[l]] <- f1_baseline_from_class_threshold(kfold_loocv_model_response_ordered[[l]], 
##                                                                     kfold_loocv_model_response, 
##                                                                     kfold_loocv_train_y)
##    
##    kfold_loocv_f1_ratio[[l]] <- kfold_loocv_f1_score[[l]]/kfold_loocv_f1_baseline[[l]]
##    
##    kfold_loocv_gmean_score[[l]] <- gmean_from_class_threshold(kfold_loocv_model_response_ordered[[l]], 
##                                                               kfold_loocv_model_response, 
##                                                               kfold_loocv_train_y)
##    
##    kfold_loocv_gmean_baseline[[l]] <- gmean_baseline_from_class_threshold(kfold_loocv_model_response_ordered[[l]],
##                                                                           kfold_loocv_model_response,
##                                                                           kfold_loocv_train_y)
##    
##    kfold_loocv_gmean_ratio[[l]] <- kfold_loocv_gmean_score[[l]]/kfold_loocv_gmean_baseline[[l]]
##    
##  }
##  kfold_loocv_optimal_class_threshold_distance_from_top_left_ROC <- 
##    kfold_loocv_model_response_ordered[[which.min(kfold_loocv_distance_from_top_left_ROC)]]
##  kfold_loocv_optimal_class_threshold_f1_score <- kfold_loocv_model_response_ordered[[which.max(kfold_loocv_f1_score)]]
##  kfold_loocv_optimal_class_threshold_f1_ratio <- kfold_loocv_model_response_ordered[[which.max(kfold_loocv_f1_ratio)]]
##  kfold_loocv_optimal_class_threshold_gmean_score <- kfold_loocv_model_response_ordered[[which.max(kfold_loocv_gmean_score)]]
##  kfold_loocv_optimal_class_threshold_gmean_ratio <- kfold_loocv_model_response_ordered[[which.max(kfold_loocv_gmean_ratio)]]
##  
##  kfold_loocv_optimal_threshold[i] <- median(c(
##    kfold_loocv_optimal_class_threshold_distance_from_top_left_ROC,
##    kfold_loocv_optimal_class_threshold_f1_score,
##    kfold_loocv_optimal_class_threshold_f1_ratio,
##    kfold_loocv_optimal_class_threshold_gmean_score,
##    kfold_loocv_optimal_class_threshold_gmean_ratio
##  ))
##  
##  kfold_loocv_full_predictions <- predict(kfold_loocv_optimal_fit[[i]], newx = train_full_logistic_df_x,
##                                          s=ifelse(kfold_loocv_optimal_lambda_type[i] == 'lambda.min', 
##                                                   'lambda.min', 
##                                                   'lambda.1se'),
##                                          type = 'response')
##  
##  kfold_loocv_test_predictions[i] <- kfold_loocv_full_predictions[[i]]
##  
##  kfold_loocv_test_class[i] <- ifelse(kfold_loocv_test_predictions[i]>=kfold_loocv_optimal_threshold[i],1,0)
##  
##}
##
##
##end_time <- Sys.time()
##
##
##
##
##
#### kfold_loocv Results
##
##ConfusionMatrix(kfold_loocv_test_class,
##                train_full_logistic_df_y)
##
##
##kfold_loocv_results <- cbind(train_full_customers,
##                             train_full_logistic_df_y,
##                             kfold_loocv_optimal_alpha,
##                             kfold_loocv_optimal_lambda_type,
##                             kfold_loocv_optimal_lambda,
##                             kfold_loocv_optimal_nonzero,
##                             kfold_loocv_train_optimal_logloss_model,
##                             kfold_loocv_train_logloss_baseline,
##                             kfold_loocv_train_optimal_cv_deviance,
##                             kfold_loocv_train_dev_baseline,
##                             kfold_loocv_optimal_threshold,
##                             kfold_loocv_test_predictions,
##                             kfold_loocv_test_class)
##
##
##write.csv(kfold_loocv_results, "kfold_loocv_results2.csv")                             
##
##
##
##cbind(predict(train_full_optimal_fit, newx = train_full_logistic_df_x,
##              s= 'lambda.min',
##              type = 'response'),
##      predict(train_full_optimal_fit, newx = train_full_logistic_df_x,
##              s= 'lambda.min',
##              type = 'class')) %>% view() 
##
##
### kfold training/validation and kfold test
##
##set.seed(42)
##
##kfold_kfold_foldid_outer <- replicate(1, n = floor(length(train_full_logistic_df_y)/nfold))
##for (j in 2:nfold) {
##  current_fold <- replicate(j, n = floor(length(train_full_logistic_df_y)/nfold))
##  kfold_kfold_foldid_outer <- c(kfold_kfold_foldid_outer,current_fold)
##}
##if(length(kfold_kfold_foldid_outer) == length(train_full_logistic_df_y)) {kfold_kfold_foldid_outer <- sample(kfold_kfold_foldid_outer)} else {
##  current_fold <- sample(1:nfold, size = length(train_full_logistic_df_y)-length(kfold_kfold_foldid_outer), replace = TRUE)
##  kfold_kfold_foldid_outer <- sample(c(kfold_kfold_foldid_outer, current_fold))
##}
##
##
##kfold_kfold_optimal_fit <- list()
##kfold_kfold_optimal_alpha <- vector("numeric", length = nfold)
##kfold_kfold_optimal_lambda_type <- vector("numeric", length = nfold)
##kfold_kfold_optimal_lambda <- vector("numeric", length = nfold)
##kfold_kfold_optimal_nonzero <- vector("numeric", length = nfold)
##
##kfold_kfold_train_optimal_logloss_model <- vector("numeric", length = nfold)
##kfold_kfold_train_logloss_baseline <- vector("numeric", length = nfold)
##
##kfold_kfold_train_optimal_cv_deviance <- vector("numeric", length = nfold)
##kfold_kfold_train_dev_baseline <- vector("numeric", length = nfold)
##
##kfold_kfold_optimal_threshold <- vector("numeric", length = nfold)
##
##kfold_kfold_test_predictions <- vector("numeric", length = length(train_full_logistic_df_y))
##kfold_kfold_test_class <- vector("numeric", length = length(train_full_logistic_df_y))
##
##start_time <- Sys.time()
##
##
##
##for (i in 1:nfold) {
##  print(i)
##  kfold_kfold_train_y <- train_full_logistic_df_y[kfold_kfold_foldid_outer != i]
##  kfold_kfold_train_x <- train_full_logistic_df_x[kfold_kfold_foldid_outer != i,]
##  kfold_kfold_foldid <- replicate(1, n = floor(length(kfold_kfold_train_y)/nfold))
##  for (j in 2:nfold) {
##    current_fold <- replicate(j, n = floor(length(kfold_kfold_train_y)/nfold))
##    kfold_kfold_foldid <- c(kfold_kfold_foldid,current_fold)
##  }
##  if(length(kfold_kfold_foldid) == length(kfold_kfold_train_y)) {kfold_kfold_foldid <- sample(kfold_kfold_foldid)} else {
##    current_fold <- sample(1:nfold, size = length(kfold_kfold_train_y)-length(kfold_kfold_foldid), replace = TRUE)
##    kfold_kfold_foldid <- sample(c(kfold_kfold_foldid, current_fold))
##  }
##  
##  kfold_kfold_cvfits <- list()
##  kfold_kfold_lambda_type <- vector('numeric', length = length(alpha_levels))
##  kfold_kfold_cv_deviance <- vector('numeric', length = length(alpha_levels))
##  
##  
##  for (k in 1:length(alpha_levels)) {
##    
##    
##    kfold_kfold_cvfit <- cv.glmnet(x = kfold_kfold_train_x, 
##                                   y = kfold_kfold_train_y, 
##                                   family = "binomial", 
##                                   type.measure = "deviance", 
##                                   nfolds = nfold,
##                                   alpha = alpha_levels[[k]],
##                                   foldid = kfold_kfold_foldid)
##    
##    kfold_kfold_cvfits[[k]] <- kfold_kfold_cvfit
##    
##    kfold_kfold_cvfit_values <- print(kfold_kfold_cvfit)
##    
##    
##    kfold_kfold_cv_deviance_lambdamin <- kfold_kfold_cvfit_values %>% 
##      filter(Lambda == kfold_kfold_cvfit$lambda.min) %>% 
##      dplyr::select(Measure) %>% 
##      min()
##    
##    
##    
##    kfold_kfold_cv_deviance_lambda1se <- kfold_kfold_cvfit_values %>% 
##      filter(Lambda == kfold_kfold_cvfit$lambda.1se) %>% 
##      dplyr::select(Measure) %>% 
##      min()
##    
##    
##    if (kfold_kfold_cv_deviance_lambdamin < kfold_kfold_cv_deviance_lambda1se) {
##      kfold_kfold_lambda_type[k] <- 'lambda.min'
##      kfold_kfold_cv_deviance[k] <- kfold_kfold_cv_deviance_lambdamin
##    } else {
##      kfold_kfold_lambda_type[k] <- 'lambda.1se'
##      kfold_kfold_cv_deviance[k] <- kfold_kfold_cv_deviance_lambda1se
##    }
##  }
##  
##  kfold_kfold_optimal_fit[[i]] <- kfold_kfold_cvfits[[which.min(kfold_kfold_cv_deviance)]]
##  kfold_kfold_optimal_alpha[i] <- alpha_levels[[which.min(kfold_kfold_cv_deviance)]]
##  kfold_kfold_optimal_lambda_type[i] <- kfold_kfold_lambda_type[[which.min(kfold_kfold_cv_deviance)]]
##  kfold_kfold_optimal_lambda[i] <- ifelse(kfold_kfold_optimal_lambda_type[i] == 'lambda.min',
##                                          kfold_kfold_optimal_fit[[i]]$lambda.min,
##                                          kfold_kfold_optimal_fit[[i]]$lambda.1se)
##  kfold_kfold_optimal_fit_values <- print(kfold_kfold_optimal_fit[[i]])
##  if(kfold_kfold_optimal_lambda_type[i] == 'lambda.min') {
##    kfold_kfold_optimal_nonzero[i] <- kfold_kfold_optimal_fit_values %>% 
##      filter(Lambda == kfold_kfold_optimal_fit[[i]]$lambda.min) %>% 
##      dplyr::select(Nonzero) %>% 
##      min()
##  } else {
##    kfold_kfold_optimal_nonzero[i]  <- kfold_kfold_optimal_fit_values %>% 
##      filter(Lambda == kfold_kfold_optimal_fit[[i]]$lambda.1se) %>% 
##      dplyr::select(Nonzero) %>% 
##      min()
##  }
##  
##  
##  if(kfold_kfold_optimal_lambda_type[i] == 'lambda.min') {
##    kfold_kfold_train_optimal_logloss_model[i] <- LogLoss(predict(kfold_kfold_optimal_fit[[i]],
##                                                                  newx = kfold_kfold_train_x,
##                                                                  s=kfold_kfold_optimal_fit[[i]]$lambda.min,
##                                                                  type = 'response'), kfold_kfold_train_y)
##  } else {
##    kfold_kfold_train_optimal_logloss_model[i] <- LogLoss(predict(kfold_kfold_optimal_fit[[i]],
##                                                                  newx = kfold_kfold_train_x,
##                                                                  s=kfold_kfold_optimal_fit[[i]]$lambda.1se,
##                                                                  type = 'response'), kfold_kfold_train_y)
##  }
##  
##  kfold_kfold_baseline_predictions <- rep.int(mean(kfold_kfold_train_y),length(kfold_kfold_train_y))
##  kfold_kfold_train_logloss_baseline[i] <- LogLoss(kfold_kfold_baseline_predictions, kfold_kfold_train_y)
##  
##  kfold_kfold_train_optimal_cv_deviance[i] <- kfold_kfold_cv_deviance[[which.min(kfold_kfold_cv_deviance)]]
##  
##  kfold_kfold_optimal_glmnetfit <- kfold_kfold_optimal_fit[[i]]$glmnet.fit
##  kfold_kfold_train_dev_baseline[i] <- kfold_kfold_optimal_glmnetfit$nulldev/length(kfold_kfold_train_y)
##  
##  
##  
##  
##  
##  kfold_kfold_model_response <- predict(kfold_kfold_optimal_fit[[i]], newx = kfold_kfold_train_x,
##                                        s=ifelse(kfold_kfold_optimal_lambda_type[i] == 'lambda.min', 
##                                                 'lambda.min', 
##                                                 'lambda.1se'),
##                                        type = 'response')
##  
##  
##  
##  kfold_kfold_min_response <- min(kfold_kfold_model_response)
##  kfold_kfold_max_response <- max(kfold_kfold_model_response)
##  
##  kfold_kfold_model_response_ordered <- kfold_kfold_model_response[kfold_kfold_model_response!=kfold_kfold_min_response & 
##                                                                     kfold_kfold_model_response!=kfold_kfold_max_response] %>%
##    unique() %>%
##    sort(decreasing=TRUE)
##  
##  kfold_kfold_fpr_score <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  kfold_kfold_tpr_score <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  kfold_kfold_distance_from_top_left_ROC <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  
##  kfold_kfold_f1_score <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  kfold_kfold_f1_baseline <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  kfold_kfold_f1_ratio <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  
##  kfold_kfold_gmean_score <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  kfold_kfold_gmean_baseline <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  kfold_kfold_gmean_ratio <- vector("numeric", length(kfold_kfold_model_response_ordered))
##  
##  for (l in 1:length(kfold_kfold_model_response_ordered)) {
##    
##    kfold_kfold_fpr_score[[l]] <- fpr_from_class_threshold(kfold_kfold_model_response_ordered[[l]],
##                                                           kfold_kfold_model_response,
##                                                           kfold_kfold_train_y)
##    
##    kfold_kfold_tpr_score[[l]] <- tpr_from_class_threshold(kfold_kfold_model_response_ordered[[l]],
##                                                           kfold_kfold_model_response, 
##                                                           kfold_kfold_train_y)
##    
##    kfold_kfold_distance_from_top_left_ROC[[l]] <- sqrt((0-kfold_kfold_fpr_score[[l]])^2 + (1-kfold_kfold_tpr_score[[l]])^2)
##    
##    
##    
##    kfold_kfold_f1_score[[l]] <- f1_from_class_threshold(kfold_kfold_model_response_ordered[[l]], 
##                                                         kfold_kfold_model_response, 
##                                                         kfold_kfold_train_y)
##    
##    kfold_kfold_f1_baseline[[l]] <- f1_baseline_from_class_threshold(kfold_kfold_model_response_ordered[[l]], 
##                                                                     kfold_kfold_model_response, 
##                                                                     kfold_kfold_train_y)
##    
##    kfold_kfold_f1_ratio[[l]] <- kfold_kfold_f1_score[[l]]/kfold_kfold_f1_baseline[[l]]
##    
##    kfold_kfold_gmean_score[[l]] <- gmean_from_class_threshold(kfold_kfold_model_response_ordered[[l]], 
##                                                               kfold_kfold_model_response, 
##                                                               kfold_kfold_train_y)
##    
##    kfold_kfold_gmean_baseline[[l]] <- gmean_baseline_from_class_threshold(kfold_kfold_model_response_ordered[[l]],
##                                                                           kfold_kfold_model_response,
##                                                                           kfold_kfold_train_y)
##    
##    kfold_kfold_gmean_ratio[[l]] <- kfold_kfold_gmean_score[[l]]/kfold_kfold_gmean_baseline[[l]]
##    
##  }
##  kfold_kfold_optimal_class_threshold_distance_from_top_left_ROC <- 
##    kfold_kfold_model_response_ordered[[which.min(kfold_kfold_distance_from_top_left_ROC)]]
##  kfold_kfold_optimal_class_threshold_f1_score <- kfold_kfold_model_response_ordered[[which.max(kfold_kfold_f1_score)]]
##  kfold_kfold_optimal_class_threshold_f1_ratio <- kfold_kfold_model_response_ordered[[which.max(kfold_kfold_f1_ratio)]]
##  kfold_kfold_optimal_class_threshold_gmean_score <- kfold_kfold_model_response_ordered[[which.max(kfold_kfold_gmean_score)]]
##  kfold_kfold_optimal_class_threshold_gmean_ratio <- kfold_kfold_model_response_ordered[[which.max(kfold_kfold_gmean_ratio)]]
##  
##  kfold_kfold_optimal_threshold[i] <- median(c(
##    kfold_kfold_optimal_class_threshold_distance_from_top_left_ROC,
##    kfold_kfold_optimal_class_threshold_f1_score,
##    kfold_kfold_optimal_class_threshold_f1_ratio,
##    kfold_kfold_optimal_class_threshold_gmean_score,
##    kfold_kfold_optimal_class_threshold_gmean_ratio
##  ))
##  
##  kfold_kfold_test_predictions_current <- predict(kfold_kfold_optimal_fit[[i]], newx = train_full_logistic_df_x[kfold_kfold_foldid_outer == i,],
##                                                  s=ifelse(kfold_kfold_optimal_lambda_type[i] == 'lambda.min', 
##                                                           'lambda.min', 
##                                                           'lambda.1se'),
##                                                  type = 'response')
##  
##  kfold_kfold_test_predictions[kfold_kfold_foldid_outer == i] <- kfold_kfold_test_predictions_current
##  
##  kfold_kfold_test_class[kfold_kfold_foldid_outer == i] <- ifelse(kfold_kfold_test_predictions_current>=
##                                                                    kfold_kfold_optimal_threshold[i],1,0)
##  
##}
##
##
##end_time <- Sys.time()
##
##kfold_kfold_results <- cbind(
##  kfold_kfold_optimal_alpha,
##  kfold_kfold_optimal_lambda_type,
##  kfold_kfold_optimal_lambda,
##  kfold_kfold_optimal_nonzero,
##  kfold_kfold_train_optimal_logloss_model,
##  kfold_kfold_train_logloss_baseline,
##  kfold_kfold_train_optimal_cv_deviance,
##  kfold_kfold_train_dev_baseline,
##  kfold_kfold_optimal_threshold
##)
##
##
##write.csv(kfold_kfold_results, "kfold_kfold_results.csv")  
##
##ConfusionMatrix(kfold_kfold_test_class,
##                train_full_logistic_df_y)
##
##?coefplot
##
##write.csv(cbind(train_full_customers,
##                kfold_kfold_foldid_outer, 
##                kfold_kfold_test_predictions,
##                kfold_kfold_test_class,
##                train_full_logistic_df_y,
##                train_full_logistic_df_x), "probabilities.csv")
##
##
##
#### All threshold results of optimal model
##
##set.seed(42)
##
##fp_cost_levels <- c(1:100)/100
##
##
#### Step 1: models, sets and predictions
##
##set.seed(42)
##
##kfold_kfold_foldid_outer <- replicate(1, n = floor(length(train_full_logistic_df_y)/nfold))
##for (j in 2:nfold) {
##  current_fold <- replicate(j, n = floor(length(train_full_logistic_df_y)/nfold))
##  kfold_kfold_foldid_outer <- c(kfold_kfold_foldid_outer,current_fold)
##}
##if(length(kfold_kfold_foldid_outer) == length(train_full_logistic_df_y)) {kfold_kfold_foldid_outer <- sample(kfold_kfold_foldid_outer)} else {
##  current_fold <- sample(1:nfold, size = length(train_full_logistic_df_y)-length(kfold_kfold_foldid_outer), replace = TRUE)
##  kfold_kfold_foldid_outer <- sample(c(kfold_kfold_foldid_outer, current_fold))
##}
##
##
##kfold_kfold_optimal_fit <- list()
##kfold_kfold_optimal_alpha <- vector("numeric", length = nfold)
##kfold_kfold_optimal_lambda_type <- vector("numeric", length = nfold)
##kfold_kfold_optimal_lambda <- vector("numeric", length = nfold)
##kfold_kfold_optimal_nonzero <- vector("numeric", length = nfold)
##
##kfold_kfold_train_optimal_logloss_model <- vector("numeric", length = nfold)
##kfold_kfold_train_logloss_baseline <- vector("numeric", length = nfold)
##
##kfold_kfold_train_optimal_cv_deviance <- vector("numeric", length = nfold)
##kfold_kfold_train_dev_baseline <- vector("numeric", length = nfold)
##
##
####kfold_kfold_test_predictions <- vector("numeric", length = length(train_full_logistic_df_y))
##
##kfold_kfold_model_responses <- list()
##kfold_kfold_test_predictions_list <- list()
##kfold_kfold_train_ys <- list()
##
##kfold_kfold_true_class <- list()
##
##
##
##
##start_time <- Sys.time()
##
##
##
##for (i in 1:nfold) {
##  print(i)
##  kfold_kfold_true_class[[i]] <- train_full_logistic_df_y[kfold_kfold_foldid_outer == i]
##  kfold_kfold_train_y <- train_full_logistic_df_y[kfold_kfold_foldid_outer != i]
##  kfold_kfold_train_ys[[i]] <- kfold_kfold_train_y
##  kfold_kfold_train_x <- train_full_logistic_df_x[kfold_kfold_foldid_outer != i,]
##  kfold_kfold_foldid <- replicate(1, n = floor(length(kfold_kfold_train_y)/nfold))
##  for (j in 2:nfold) {
##    current_fold <- replicate(j, n = floor(length(kfold_kfold_train_y)/nfold))
##    kfold_kfold_foldid <- c(kfold_kfold_foldid,current_fold)
##  }
##  if(length(kfold_kfold_foldid) == length(kfold_kfold_train_y)) {kfold_kfold_foldid <- sample(kfold_kfold_foldid)} else {
##    current_fold <- sample(1:nfold, size = length(kfold_kfold_train_y)-length(kfold_kfold_foldid), replace = TRUE)
##    kfold_kfold_foldid <- sample(c(kfold_kfold_foldid, current_fold))
##  }
##  
##  
##  kfold_kfold_cvfits <- list()
##  kfold_kfold_lambda_type <- vector('numeric', length = length(alpha_levels))
##  kfold_kfold_cv_deviance <- vector('numeric', length = length(alpha_levels))
##  
##  
##  for (k in 1:length(alpha_levels)) {
##    
##    
##    kfold_kfold_cvfit <- cv.glmnet(x = kfold_kfold_train_x, 
##                                   y = kfold_kfold_train_y, 
##                                   family = "binomial", 
##                                   type.measure = "deviance", 
##                                   nfolds = nfold,
##                                   alpha = alpha_levels[[k]],
##                                   foldid = kfold_kfold_foldid)
##    
##    kfold_kfold_cvfits[[k]] <- kfold_kfold_cvfit
##    
##    kfold_kfold_cvfit_values <- print(kfold_kfold_cvfit)
##    
##    
##    kfold_kfold_cv_deviance_lambdamin <- kfold_kfold_cvfit_values %>% 
##      filter(Lambda == kfold_kfold_cvfit$lambda.min) %>% 
##      dplyr::select(Measure) %>% 
##      min()
##    
##    
##    
##    kfold_kfold_cv_deviance_lambda1se <- kfold_kfold_cvfit_values %>% 
##      filter(Lambda == kfold_kfold_cvfit$lambda.1se) %>% 
##      dplyr::select(Measure) %>% 
##      min()
##    
##    
##    if (kfold_kfold_cv_deviance_lambdamin < kfold_kfold_cv_deviance_lambda1se) {
##      kfold_kfold_lambda_type[k] <- 'lambda.min'
##      kfold_kfold_cv_deviance[k] <- kfold_kfold_cv_deviance_lambdamin
##    } else {
##      kfold_kfold_lambda_type[k] <- 'lambda.1se'
##      kfold_kfold_cv_deviance[k] <- kfold_kfold_cv_deviance_lambda1se
##    }
##  }
##  
##  kfold_kfold_optimal_fit[[i]] <- kfold_kfold_cvfits[[which.min(kfold_kfold_cv_deviance)]]
##  kfold_kfold_optimal_alpha[i] <- alpha_levels[[which.min(kfold_kfold_cv_deviance)]]
##  kfold_kfold_optimal_lambda_type[i] <- kfold_kfold_lambda_type[[which.min(kfold_kfold_cv_deviance)]]
##  kfold_kfold_optimal_lambda[i] <- ifelse(kfold_kfold_optimal_lambda_type[i] == 'lambda.min',
##                                          kfold_kfold_optimal_fit[[i]]$lambda.min,
##                                          kfold_kfold_optimal_fit[[i]]$lambda.1se)
##  kfold_kfold_optimal_fit_values <- print(kfold_kfold_optimal_fit[[i]])
##  if(kfold_kfold_optimal_lambda_type[i] == 'lambda.min') {
##    kfold_kfold_optimal_nonzero[i] <- kfold_kfold_optimal_fit_values %>% 
##      filter(Lambda == kfold_kfold_optimal_fit[[i]]$lambda.min) %>% 
##      dplyr::select(Nonzero) %>% 
##      min()
##  } else {
##    kfold_kfold_optimal_nonzero[i]  <- kfold_kfold_optimal_fit_values %>% 
##      filter(Lambda == kfold_kfold_optimal_fit[[i]]$lambda.1se) %>% 
##      dplyr::select(Nonzero) %>% 
##      min()
##  }
##  
##  
##  if(kfold_kfold_optimal_lambda_type[i] == 'lambda.min') {
##    kfold_kfold_train_optimal_logloss_model[i] <- LogLoss(predict(kfold_kfold_optimal_fit[[i]],
##                                                                  newx = kfold_kfold_train_x,
##                                                                  s=kfold_kfold_optimal_fit[[i]]$lambda.min,
##                                                                  type = 'response'), kfold_kfold_train_y)
##  } else {
##    kfold_kfold_train_optimal_logloss_model[i] <- LogLoss(predict(kfold_kfold_optimal_fit[[i]],
##                                                                  newx = kfold_kfold_train_x,
##                                                                  s=kfold_kfold_optimal_fit[[i]]$lambda.1se,
##                                                                  type = 'response'), kfold_kfold_train_y)
##  }
##  
##  kfold_kfold_baseline_predictions <- rep.int(mean(kfold_kfold_train_y),length(kfold_kfold_train_y))
##  kfold_kfold_train_logloss_baseline[i] <- LogLoss(kfold_kfold_baseline_predictions, kfold_kfold_train_y)
##  
##  kfold_kfold_train_optimal_cv_deviance[i] <- kfold_kfold_cv_deviance[[which.min(kfold_kfold_cv_deviance)]]
##  
##  kfold_kfold_optimal_glmnetfit <- kfold_kfold_optimal_fit[[i]]$glmnet.fit
##  kfold_kfold_train_dev_baseline[i] <- kfold_kfold_optimal_glmnetfit$nulldev/length(kfold_kfold_train_y)
##  
##  
##  
##  
##  
##  kfold_kfold_model_response <- predict(kfold_kfold_optimal_fit[[i]], newx = kfold_kfold_train_x,
##                                        s=ifelse(kfold_kfold_optimal_lambda_type[i] == 'lambda.min', 
##                                                 'lambda.min', 
##                                                 'lambda.1se'),
##                                        type = 'response')
##  
##  
##  kfold_kfold_model_responses[[i]] <- kfold_kfold_model_response
##  
##  
##  
##  
##  
##  kfold_kfold_test_predictions_list[[i]] <- predict(kfold_kfold_optimal_fit[[i]], newx = train_full_logistic_df_x[kfold_kfold_foldid_outer == i,],
##                                                    s=ifelse(kfold_kfold_optimal_lambda_type[i] == 'lambda.min', 
##                                                             'lambda.min', 
##                                                             'lambda.1se'),
##                                                    type = 'response')
##  
##}
##
##
### Step 2: Resuls for all thresholds
##
##kfold_kfold_true_class <- do.call(rbind, kfold_kfold_true_class)
##true_negatives <- vector('numeric', length = length(fp_cost_levels))
##false_positives <- vector('numeric', length = length(fp_cost_levels))
##false_negatives <- vector('numeric', length = length(fp_cost_levels))
##true_positives <- vector('numeric', length = length(fp_cost_levels))
##
##
##for (i in 1:length(fp_cost_levels)) {
##  print(i)
##  kfold_kfold_test_class <- list()
##  for (k in 1:nfold) {
##    kfold_kfold_test_predictions <- kfold_kfold_test_predictions_list[[k]]
##    kfold_kfold_model_response <- kfold_kfold_model_responses[[k]]
##    
##    
##    kfold_kfold_min_response <- min(kfold_kfold_model_response)
##    kfold_kfold_max_response <- max(kfold_kfold_model_response)
##    
##    kfold_kfold_model_response_ordered <- kfold_kfold_model_response[kfold_kfold_model_response!=kfold_kfold_min_response & 
##                                                                       kfold_kfold_model_response!=kfold_kfold_max_response] %>%
##      unique() %>%
##      sort(decreasing=TRUE)
##    
##    kfold_kfold_cost_score <- vector("numeric", length(kfold_kfold_model_response_ordered))
##    
##    for (l in 1:length(kfold_kfold_model_response_ordered)) {
##      
##      kfold_kfold_cost_score[[l]] <- cost_from_class_threshold(kfold_kfold_model_response_ordered[[l]],
##                                                               kfold_kfold_model_response,
##                                                               kfold_kfold_train_y,
##                                                               fp_cost_levels[[i]])
##    }
##    
##    kfold_kfold_optimal_threshold <- 
##      kfold_kfold_model_response_ordered[[which.min(kfold_kfold_cost_score)]]
##    
##    
##    kfold_kfold_test_class[[k]] <- ifelse(kfold_kfold_test_predictions>=
##                                            kfold_kfold_optimal_threshold,1,0)
##  }
##  kfold_kfold_test_class <- do.call(rbind, kfold_kfold_test_class)
##  confusion_matrix <- ConfusionMatrix(kfold_kfold_test_class,
##                                      kfold_kfold_true_class)
##  true_negatives[[i]] <- confusion_matrix[1,1]
##  false_positives[i] <- confusion_matrix[1,2]
##  false_negatives[i] <- confusion_matrix[2,1]
##  true_positives[i] <- confusion_matrix[2,2]
##}
##
##setwd("E:/experiment.repositories/series/crit_sit_attrition/logit_models/Phase_1/with_2019_no_qa")
##write.csv(cbind(fp_cost_levels, 
##                true_negatives,
##                false_positives,
##                false_negatives,
##                true_positives), "all_cost_results.csv")
##
##
##
##
##
##getwd()
##write.csv(train_full_ly_predictions, "train_full_ly_predictions.csv")