##### authors: dr. Laryushkin, dr Kritskaya 


####        PDS analysis git version            ####

#### First of all, please clean up the environment ####

rm(list=ls()) 
ls()

#### library ####

# if the libraries are not installed, please install all the necessary ones.
library(lessR)
library(ggplot2)
library(GGally)
library(cluster)
library(dplyr)
library(caTools)
library(factoextra)
library(PerformanceAnalytics)
library(psych)
library(corrplot)
library(randomForest)
library(pROC)
library(caret)
library(tidyverse)
library(tidymodels)
library(vip)
library(rpart.plot)

#### choose a directory ####

setwd("copy the full path to the directory here")

#### data preparation ####

df <- read.csv2('data_1.csv', 
      header = TRUE, sep  = ';', dec = '.') #  main dataframe

df_Cor <- df[,-c(1,2,18,22,28)] # dataframe for the correlation matrix




####  visualize correlation matrix  ####


df_Cor$Type_PDS <-  as.factor(df_Cor$Type_PDS)

CR <- cor(df_cor)

corrplot(CR,tl.cex = 0.5, hclust.method = c("complete"))

df_cor <- df[,c(3, 26,15)]

chart.Correlation(df_cor, histogram=TRUE, pch=19, cex.cor=50)

par(mar = rep(2, 4))

######  pie plot ######  



PDS_type <- table(df$Type_PDS)

qw <- data.frame(PDS_type = PDS_type)

PieChart(PDS_type, hole = 0.45, values = "%", data = qw,
         fill = c( "grey", "lightblue", "black"), main = "",
         values_size = 1.2, labels_cex =1.2)



####  decision trees #### 


df  <- df[,-c(1,2)]
df$Type_PDS <- as.factor(df$Type_PDS)


## get AUC and other metrics
# Remember to always set your seed. Any integer will work

set.seed(71)

  rf <-randomForest(Type_PDS ~.,data = df, 
                  importance=TRUE,ntree=500)
  print(rf)
  
#Evaluate variable importance
  
importance(rf)

  varImpPlot(rf)


churn_split <- initial_split(df_Cor, prop = 0.5)

  churn_training <- churn_split %>% training()

    churn_test <- churn_split %>% testing()

churn_folds <- vfold_cv(churn_training, v = 3)
    churn_recipe <- recipe(Type_PDS  ~ ., data = churn_training) %>% 
     step_YeoJohnson(all_numeric(), -all_outcomes()) %>% 
       step_normalize(all_numeric(), -all_outcomes()) %>% 
         step_dummy(all_nominal(), -all_outcomes())

churn_recipe %>% 
  prep() %>% 
   bake(new_data = churn_training)

tree_model <- decision_tree(cost_complexity = tune(),
               tree_depth = tune(),
                  min_n = tune()) %>% 
set_engine('rpart') %>% 
    set_mode('classification')


tree_workflow <- workflow() %>% 
  add_model(tree_model) %>% 
    add_recipe(churn_recipe)


tree_grid <- grid_regular(cost_complexity(),
            tree_depth(),
              min_n(), 
                levels = 3)

tree_grid

  tree_tuning <- tree_workflow %>% 
    tune_grid(resamples = churn_folds,
            grid = tree_grid)


tree_tuning %>% show_best('roc_auc')


best_tree <- tree_tuning %>% 
  select_best(metric = 'roc_auc')

# View the best tree parameters
best_tree

final_tree_workflow <- tree_workflow %>% 
  finalize_workflow(best_tree)

tree_wf_fit <- final_tree_workflow %>% 
  fit(data = churn_training)

tree_fit <- tree_wf_fit %>% 
  pull_workflow_fit()

vip(tree_fit)

rpart.plot(tree_fit$fit, roundint = FALSE)


tree_last_fit <- final_tree_workflow %>% 
  last_fit(churn_split)

tree_last_fit %>% collect_metrics()


tree_last_fit %>% collect_predictions() %>% 
  roc_curve(truth  = canceled_service, estimate = .pred_yes) %>% 
    autoplot()





####  without parameters #####

df_Cor1 <- df_Cor[,-c(2:11,13,14)]


df_Cor1$Type_PDS <- as.factor(df_Cor1$Type_PDS)


churn_split <- initial_split(df_Cor1, prop = 0.45)

churn_training <- churn_split %>% training()

churn_test <- churn_split %>% testing()

churn_folds <- vfold_cv(churn_training, v = 3)

churn_recipe <- recipe(Type_PDS  ~ ., data = churn_training) %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>% 
    step_normalize(all_numeric(), -all_outcomes()) %>% 
      step_dummy(all_nominal(), -all_outcomes())

churn_recipe %>% 
  prep() %>% 
    bake(new_data = churn_training)

tree_model <- decision_tree(cost_complexity = tune(),
                            tree_depth = tune(),
                            min_n = tune()) %>% 
  set_engine('rpart') %>% 
  set_mode('classification')


tree_workflow <- workflow() %>% 
  add_model(tree_model) %>% 
   add_recipe(churn_recipe)

tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          min_n(), 
                          levels = 3)

tree_grid

tree_tuning <- tree_workflow %>% 
  tune_grid(resamples = churn_folds,
            grid = tree_grid)

tree_tuning %>% show_best('roc_auc')

best_tree <- tree_tuning %>% 
  select_best(metric = 'roc_auc')

# View the best tree parameters
best_tree

final_tree_workflow <- tree_workflow %>% 
  finalize_workflow(best_tree)


tree_wf_fit <- final_tree_workflow %>% 
  fit(data = churn_training)

tree_fit <- tree_wf_fit %>% 
  pull_workflow_fit()

vip(tree_fit)

rpart.plot(tree_fit$fit, roundint = TRUE)

tree_last_fit <- final_tree_workflow %>% 
  last_fit(churn_split)

tree_last_fit %>% collect_metrics()


tree_last_fit %>% collect_predictions() %>% 
  roc_curve(truth  = canceled_service, estimate = .pred_yes) %>% 
  autoplot()







####  PCA  #####
#### data for PCA ####

df_PCA  <- df[,-c(1,2)]
  groups <- as.factor(df[,14])

###### PCA ##### 

res.pca <- prcomp(df_PCA, scale = TRUE)
  fviz_eig(res.pca)




fviz_pca_var(res.pca,
             
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)



#### data for PCA  reduce parameters #######
df_Cor2 <- df[,-c(1,2,4:8:13,15:21,23:26,28)]
##### PCA reduce parameters #######

res.pca <- prcomp(df_Cor2, scale = TRUE)
fviz_eig(res.pca)

fviz_pca_ind(res.pca,
             col.ind = groups, # color by groups
             palette = c('#00AFBB', '#E7B800', '#FC4E07'),
             addEllipses = TRUE, # Concentration ellipses
             ellipse.type = "confidence",
             legend.title = "PDS type",
             repel = TRUE
)



ggplot()+
  geom

############# correl #########
df_corr <- df[,c(3,9,14,23,27)]
df_corr$Type_PDS <- as.factor(df_corr$Type_PDS )

ggpairs(df_corr,
        
        aes(color = Type_PDS),                             # Separate data by levels of vs
        upper = list(continuous = wrap('cor', size = 3)),
        lower = list(combo = wrap("facethist", bins = 30)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of `mtcars` Grouped by Engine")















