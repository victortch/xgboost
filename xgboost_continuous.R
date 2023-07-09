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
library(xgboost)
library(DiagrammeR)



analysis_df_latest_location <- "C:/Analyses/premier_customers_prediction/all_sequential/envs/analysis_df_latest"

value_models_latest_location <- "C:/Analyses/premier_customers_prediction/all_sequential/envs/value_models_latest"
value_models_old_location <- "C:/Analyses/premier_customers_prediction/all_sequential/envs/value_models_old"


alpha_levels <- c(0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1)
folds <- 10
set.seed(424242)

max_depths <- c(2:20)
nrounds <- c(2:20)
gammas <- c(0:8)/4


########Functions

rsq <- function(y, yhat) {
  mse <- assess.glmnet(yhat, newy = y, family = "gaussian")$mse[[1]]
  return(1 - mse  /var(y))
}


create_foldid <- function(len, nfold) {
  foldid <- replicate(1, n = floor(len/nfold))
  for (j in 2:nfold) {
    current_fold <- replicate(j, n = floor(len/nfold))
    foldid <- c(foldid,current_fold)
  }
  
  if(length(foldid) == len) {foldid <- sample(foldid)} else {
    current_fold <- sample(1:nfold, size = len - length(foldid), replace = TRUE)
    foldid <- sample(c(foldid, current_fold))
  }
  return(foldid)
}







#########################


latest_analysis_df_filename <- paste(
  analysis_df_latest_location,
  list.files(analysis_df_latest_location, pattern = 'Rdata')[[1]],
  sep = "/"
)

load(latest_analysis_df_filename)



#####Linear value model


regression_df_train_full <- regression_df %>%
  left_join(dep_vars_df)

train_full_linear_df_y <- regression_df_train_full %>%
  dplyr::select(actual_revenue) %>%
  unlist() %>%
  as.vector()

train_full_linear_df_x <- regression_df_train_full %>%
  dplyr::select(-one_of(colnames(dep_vars_df))) %>%
  as.matrix() 

train_full_foldid <- create_foldid(length(train_full_linear_df_y), folds)

#######################Fit lm for outlier detection

lm_df <- cbind(train_full_linear_df_y, train_full_linear_df_x) %>% as.data.frame()
outlier_lm <- lm(train_full_linear_df_y ~. , data = lm_df)

############DFFits

p <- length(outlier_lm$coefficients)
n <- nrow(outlier_lm$model)
dffits_crit = 3 * sqrt((p + 1) / (n - p - 1))
outlier_lm_dffits <- dffits(outlier_lm)
dffit_index <- outlier_lm_dffits > dffits_crit

#df <- data.frame(obs = names(outlier_lm_dffits),
#                 dffits = outlier_lm_dffits)
#ggplot(df, aes(y = dffits, x = obs)) +
#  geom_point() +
#  geom_hline(yintercept = c(dffits_crit, -dffits_crit), linetype="dashed") +
#  labs(title = "DFFITS",
#       x = "Observation Number",
#       y = "DFFITS")


###########Cooks Distance

cooks_crit = 0.5
outlier_lm_cooks <- cooks.distance(outlier_lm)
df <- data.frame(obs = names(outlier_lm_cooks),
                 cooks = outlier_lm_cooks)
cooks_index <- outlier_lm_cooks > cooks_crit

#ggplot(df, aes(y = cooks, x = obs)) +
#  geom_point() +
#  geom_hline(yintercept = cooks_crit, linetype="dashed") +
#  labs(title = "Cook's Distance",
#       x = "Observation Number",
#       y = "Cook's")
#

###########Filter out outliers

influential_index <- dffit_index | cooks_index
influential_index[is.na(influential_index)] <- TRUE

regression_df_train_full <- regression_df_train_full[!influential_index,]

train_full_linear_df_y <- regression_df_train_full %>%
  dplyr::select(actual_revenue) %>%
  unlist() %>%
  as.vector()

train_full_linear_df_x <- regression_df_train_full %>%
  dplyr::select(-one_of(colnames(dep_vars_df))) %>%
  as.matrix() 

train_full_foldid <- create_foldid(length(train_full_linear_df_y), folds)



################Split Dataset to train and test

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


train_linear_df_y <- regression_df_train %>%
  dplyr::select(actual_revenue) %>%
  unlist() %>%
  as.vector()

train_linear_df_x <- regression_df_train %>%
  dplyr::select(-one_of(colnames(dep_vars_df))) %>%
  as.matrix() 



test_linear_df_y <- regression_df_test %>%
  dplyr::select(actual_revenue) %>%
  unlist() %>%
  as.vector()

test_linear_df_x <- regression_df_test %>%
  dplyr::select(-one_of(colnames(dep_vars_df))) %>%
  as.matrix()







train_foldid <- create_foldid(length(train_linear_df_y), folds)
test_foldid <- create_foldid(length(test_linear_df_y), folds)









#######################Train first model


train_cvfits <- list()
train_full_cvfits <- list()

train_rsq <- vector('numeric', length = length(alpha_levels))
train_val_rsq <- vector('numeric', length = length(alpha_levels))
test_rsq <- vector('numeric', length = length(alpha_levels))

train_full_rsq <- vector('numeric', length = length(alpha_levels))
train_full_val_rsq <- vector('numeric', length = length(alpha_levels))



for (i in 1:length(alpha_levels)) {
  print(i)
  
  train_full_cvfit <- cv.glmnet(x = train_full_linear_df_x, 
                                y = train_full_linear_df_y, 
                                family = "gaussian", 
                                nfolds = nfold,
                                alpha = alpha_levels[[i]],
                                foldid = train_full_foldid,
                                keep = TRUE)
  
  train_cvfit <- cv.glmnet(x = train_linear_df_x, 
                           y = train_linear_df_y, 
                           family = "gaussian", 
                           nfolds = nfold,
                           alpha = alpha_levels[[i]],
                           foldid = train_foldid,
                           keep = TRUE)
  
  
  minlambda_index <- train_cvfit$index[[1]]
  
  train_full_cvfits[[i]] <- train_full_cvfit
  train_cvfits[[i]] <- train_cvfit
  
  train_preds <- predict(train_cvfit, newx = train_linear_df_x, s = 'lambda.min')
  train_val_preds <- train_cvfit$fit.preval[,minlambda_index]
  test_preds <- predict(train_cvfit, newx = test_linear_df_x, s = 'lambda.min')
  train_full_preds <- predict(train_full_cvfit, newx = train_full_linear_df_x, s = 'lambda.min')
  train_full_val_preds <- train_full_cvfit$fit.preval[,minlambda_index]
  
  
  
  train_rsq[i] <- rsq(train_linear_df_y, train_preds)
  train_val_rsq[i] <- rsq(train_linear_df_y, train_val_preds)
  test_rsq[i] <- rsq(test_linear_df_y, test_preds)
  train_full_rsq[i] <- rsq(train_full_linear_df_y, train_full_preds)
  train_full_val_rsq[i] <- rsq(train_full_linear_df_y, train_full_val_preds)
}

optimal_train_index <- which.max(train_val_rsq)
optimal_train_full_index <- which.max(train_full_val_rsq)

glmnet_optimal_model <- train_cvfits[[optimal_train_index]]
optimal_train_alpha <- alpha_levels[[optimal_train_index]]
optimal_train_full_model <- train_full_cvfits[[optimal_train_full_index]]

optimal_train_rsq <- train_rsq[[optimal_train_index]]
optimal_train_val_rsq <-train_val_rsq[[optimal_train_index]]
optimal_test_rsq <- test_rsq[[optimal_train_index]]

optimal_train_full_rsq <- train_full_rsq[[optimal_train_index]]
optimal_train_full_val_rsq <- train_full_val_rsq[[optimal_train_index]]



perc_outliers <- sum(influential_index)/nrow(dep_vars_df)


####### Greedy xgboost

list_xgbd_matrices <- function(df_x, df_y, foldid) {
  xgbd_matrix_list <- list(list(), list(), list())
  folds <- max(foldid)
  for (j in 1:folds) {
    filter_vector <- foldid != j
    train_df_x.j <- df_x[filter_vector,]
    train_df_y.j <- df_y[filter_vector]
    validation_df_x.j <- df_x[!filter_vector,]
    validation_df_y.j <- df_y[!filter_vector]
    train_xgb_df.j <- xgb.DMatrix(data = train_df_x.j, label= train_df_y.j)
    validation_xgb_df.j <- xgb.DMatrix(data = validation_df_x.j, label= validation_df_y.j)
    xgbd_matrix_list[[1]][[j]] <- train_xgb_df.j
    xgbd_matrix_list[[2]][[j]] <- validation_xgb_df.j
    xgbd_matrix_list[[3]][[j]] <- validation_df_y.j
  }
  return(xgbd_matrix_list)
}


xgb_kfold_val_rsq <- function(max.depth.i, 
                              nround.i, 
                              gamma.i, 
                              xgbd_matrix_list,
                              booster = "gbtree",
                              objective = "reg:squarederror"
                              ) {
  folds <- length(xgbd_matrix_list[[1]])
  val_preds <- list()
  for (j in 1:folds) {
    train_xgb_df.j <- xgbd_matrix_list[[1]][[j]]
    validation_xgb_df.j <- xgbd_matrix_list[[2]][[j]]
    validation_df_y.j <- xgbd_matrix_list[[3]][[j]]
    xgbfit.j <- xgboost(data = train_xgb_df.j, 
                        max.depth = max.depth.i,
                        nround = nround.i, 
                        early_stopping_rounds = 3, 
                        gamma = gamma.i,
                        verbose = 0)
    
    val_preds[[j]] <- cbind(validation_df_y.j, predict(xgbfit.j, validation_xgb_df.j))
  }
  val_preds <- do.call("rbind", val_preds) %>% as.matrix()
  validation_rsq <- rsq(val_preds[,1], val_preds[,2])
  return(validation_rsq)
}


xgb_kfold_val_rsq_overide <- function(prev_step_var,
                                      prev_step_condition,
                                      prev_rsq,
                                      max.depth, 
                                      nround, 
                                      gamma, 
                                      xgbd_matrix_list) {
  if(prev_step_var == prev_step_condition) {
    rsq <- prev_rsq
  } else {
    rsq <- xgb_kfold_val_rsq(max.depth,
                             nround,
                             gamma,
                             xgbd_matrix_list)
  }
  return(rsq)
}



xgb_tune_greedy <- function(xgbd_matrix_list,
                            max_depth_start = 6, 
                            nrounds_start = 6,
                            gamma_start = 0,
                            increment_gamma = 0.5) {
  #Random initialization
  max_depth <- max_depth_start
  nround <- nrounds_start
  gamma <- gamma_start
  
  current_rsq <- xgb_kfold_val_rsq(max_depth,
                                   nround,
                                   gamma,
                                   xgbd_matrix_list)
  
  previous_step <- "NULL"
  baseline_rsq <- 0
  print(paste(
    "max_depth: ", max_depth, "; nround: ", nround, "; gamma: ", gamma, sep = ""
  ))
  repeat {
    previous_rsq <- baseline_rsq
    baseline_rsq <- current_rsq
    
    print(paste(previous_step, baseline_rsq, sep = " "))
    
    
    max_depth_plus_rsq <- xgb_kfold_val_rsq_overide(previous_step,
                                                    "max_depth_minus",
                                                    previous_rsq,
                                                    max_depth + 1,
                                                    nround,
                                                    gamma,
                                                    xgbd_matrix_list)
    
    
    if(max_depth >= 2) {
      max_depth_minus_rsq <- xgb_kfold_val_rsq_overide(previous_step,
                                                       "max_depth_plus",
                                                       previous_rsq,
                                                       max_depth - 1,
                                                       nround,
                                                       gamma,
                                                       xgbd_matrix_list)
    } else {max_depth_minus_rsq <- 0}
    


    nround_plus_rsq <- xgb_kfold_val_rsq_overide(previous_step,
                                                 "nround_minus",
                                                 previous_rsq,
                                                 max_depth,
                                                 nround + 1,
                                                 gamma,
                                                 xgbd_matrix_list)

    if(nround >= 2) {
      nround_minus_rsq <- xgb_kfold_val_rsq_overide(previous_step,
                                                    "nround_plus",
                                                    previous_rsq,
                                                    max_depth,
                                                    nround - 1,
                                                    gamma,
                                                    xgbd_matrix_list)
    } else {nround_minus_rsq <- 0}
    
    

    
    gamma_plus_rsq <- xgb_kfold_val_rsq_overide(previous_step,
                                                "gamma_minus",
                                                previous_rsq,
                                                max_depth,
                                                nround,
                                                gamma + increment_gamma,
                                                xgbd_matrix_list)
    
    
    if(gamma >= increment_gamma) {
      gamma_minus_rsq <- xgb_kfold_val_rsq_overide(previous_step,
                                                   "gamma_plus",
                                                   previous_rsq,
                                                   max_depth,
                                                   nround,
                                                   gamma - increment_gamma,
                                                   xgbd_matrix_list)
    } else {gamma_minus_rsq <- 0}

    
    current_rsq <- max(baseline_rsq, 
                       max_depth_plus_rsq,
                       max_depth_minus_rsq,
                       nround_plus_rsq,
                       nround_minus_rsq,
                       gamma_plus_rsq,
                       gamma_minus_rsq)
    
    if(current_rsq == baseline_rsq) {break}
      else if (current_rsq == max_depth_plus_rsq) {
      max_depth <- max_depth + 1
      previous_step <- "max_depth_plus"
    } else if (current_rsq == max_depth_minus_rsq) {
      max_depth <- max_depth - 1
      previous_step <- "max_depth_minus"
    } else if (current_rsq == nround_plus_rsq) {
      nround <- nround + 1
      previous_step <- "nround_plus"
    } else if (current_rsq == nround_minus_rsq) {
      nround <- nround - 1
      previous_step <- "nround_minus"
    } else if (current_rsq == gamma_plus_rsq) {
      gamma <- gamma + 1
      previous_step <- "gamma_plus"
    } else {
      gamma <- gamma - 1
      previous_step <- "gamma_minus"
    }
  }
  
  print(paste(
    "optimal_max_depth: ", max_depth, "; optimal_nround: ", nround, "; optimal_gamma: ", gamma, sep = ""
  ))
  
  return(c(baseline_rsq, max_depth, nround, gamma))
}




start_time <- Sys.time()
xgbd_matrices <- list_xgbd_matrices(train_linear_df_x, train_linear_df_y, train_foldid)
xgb_tunefit <- xgb_tune_greedy(xgbd_matrices)
end_time <- Sys.time()
end_time-start_time


train_xgb_df <- xgb.DMatrix(data = train_linear_df_x, label= train_linear_df_y)
test_xgb_df <- xgb.DMatrix(data = test_linear_df_x, label= test_linear_df_y)
xgbfit_optimal_model <- xgboost(data = train_xgb_df, 
                                max.depth = xgb_tunefit[[2]],
                                nround = xgb_tunefit[[3]], 
                                early_stopping_rounds = 3, 
                                gamma = xgb_tunefit[[4]],
                                verbose = 0)

xgb_preds <- predict(xgbfit_optimal_model, test_xgb_df)
glmnet_preds <- predict(glmnet_optimal_model, newx = test_linear_df_x, s = 'lambda.min')



glmnet_rsq <- rsq(test_linear_df_y, glmnet_preds)
xgb_rsq <- rsq(test_linear_df_y, xgb_preds)


plot(test_linear_df_y, glmnet_preds)
plot(test_linear_df_y, xgb_preds)


xgb.plot.multi.trees(feature_names = names(train_xgb_df), 
                     model = xgbfit_optimal_model)
importance_matrix <- xgb.importance(names(train_xgb_df), model = xgbfit_optimal_model)
xgb.plot.importance(importance_matrix)

if(xgb_rsq > glmnet_rsq) {
  optimal_model <- xgbfit_optimal_model
} else {
  optimal_model <- glmnet_optimal_model
}


#####Save Env

value_models_name <- paste(
  value_models_latest_location,
  "/value_models_",
  gsub("-","_", Sys.Date()),
  "_uk.Rdata",
  sep = ""
)

current_latest_file <- list.files(value_models_latest_location, pattern = 'Rdata')

if(length(current_latest_file) == 0){
  save(list=c("optimal_model", 
              "xgbfit_optimal_model",
              "optimal_train_alpha",
              "xgb_tunefit",
              "train_xgb_df",
              "test_xgb_df",
              "train_linear_df_x",
              "train_linear_df_y",
              "test_linear_df_x",
              "test_linear_df_y"), 
       file = value_models_name)
} else {
  latest_value_models_filename <- paste(
    value_models_latest_location,
    list.files(value_models_latest_location, pattern = 'Rdata')[[1]],
    sep = "/"
  )
  file.move(latest_value_models_filename, value_models_old_location)
  save(list=c("optimal_model", 
              "xgbfit_optimal_model",
              "optimal_train_alpha",
              "xgb_tunefit",
              "train_xgb_df",
              "test_xgb_df",
              "train_linear_df_x",
              "train_linear_df_y",
              "test_linear_df_x",
              "test_linear_df_y"), 
       file = value_models_name)
}




