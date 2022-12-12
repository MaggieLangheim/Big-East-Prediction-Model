---
title: "Big East Prediction Model"
author: "Maggie Langheim"
date: "2022-12-14"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

##### Before beginning, this analysis utilizes the following packages:

```{r}
library(ggplot2)
library(caret)
library(xgboost)
library(pROC)
library(OptimalCutpoints)
library(xgboostExplainer)
library(ggimage)
library(ggdark)
library(Metrics)
```

### Import Data
```{r}
bigEast <- read.csv("bigeastncaabasketball.csv")
teamLogos <- read.csv("teamLogos.csv")
teams21 <- bigEast[c(1:11),]
```

#### Let's start by partitioning the data into test and training data. 
#### The training data is composed of all teams to ever compete in the Big East from Fall 1979 to Spring 2021. 
#### The test data includes the 11 teams that competed in Big East Men's basketball during the 2021-22 season.
```{r}
train_data <- bigEast[bigEast$year < 2021,]
test_data <- bigEast[bigEast$year == 2021,]
```

#### Next, we look at the all time summary statistics to get an idea of what our data set looks like.
```{r}
summary(bigEast)
```


#### Finally it's time to run the XGBoost prediction model on our training and test data. Before running, we remove rankings, wins and losses (conference and nonconference), and rating satistics that have the possiblity of skewing the model.
```{r}
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, c(18:34)]), label = train_data$win_percentage)

dtest <- xgb.DMatrix(data = as.matrix(test_data[, c(18:34)]), label = test_data$win_percentage)
```

```{r}
set.seed(111111)

bst_1 <- xgboost(data = dtrain,
               nrounds = 100,
               verbose = 1,
                print_every_n = 20
 )
```

#### These are the variables of importance from the first XGBoost model. We can see that the top two most important variables are field goal percentage and total_rebounds. 
```{r}
# Extract importance
imp_mat <- xgb.importance(model = bst_1)
# Plot importance (top 10 variables)
xgb.plot.importance(imp_mat, top_n = 10)
```

#### Now we can see how well the orginal XGBoost predicts the test data, the 2021-22 basketball season.
```{r}
boost_preds <- predict(bst_1, dtest) 
```

```{r}
rmse(test_data$win_percentage, boost_preds)

plot_data <- cbind.data.frame(test_data$win_percentage, boost_preds)
names(plot_data) <- c("actual", "predicted")

ggplot(plot_data, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_smooth() +
  labs(subtitle = paste("RMSE: ",rmse(test_data$win_percentage, boost_preds), sep ="" ))
```


#### The next step is to tune the XGBoost model in order to better predict the 2021-22 season.
```{r}
max_depth_vals <- c(3, 5, 7, 10, 15) # Create vector of max depth values
min_child_weight <- c(1,3,5,7, 10, 15) # Create vector of min child values

# Expand grid of parameter values
cv_params <- expand.grid(max_depth_vals, min_child_weight)
names(cv_params) <- c("max_depth", "min_child_weight")
# Create results vector
rmse_vec  <- rep(NA, nrow(cv_params)) 
# Loop through results
for(i in 1:nrow(cv_params)){
  set.seed(111111)
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     
                     nfold = 5, # Use 5 fold cross-validation
                     
                     eta = 0.1, # Set learning rate
                     max.depth = cv_params$max_depth[i], # Set max depth
                     min_child_weight = cv_params$min_child_weight[i], # Set minimum number of samples in node to split
                     
                     
                     nrounds = 100, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
                     
  ) # Set evaluation metric to use
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}


# Join results in dataset
res_db <- cbind.data.frame(cv_params, rmse_vec)
names(res_db)[3] <- c("rmse") 
res_db$max_depth <- as.factor(res_db$max_depth) # Convert tree number to factor for plotting
res_db$min_child_weight <- as.factor(res_db$min_child_weight) # Convert node size to factor for plotting
# Print AUC heatmap
g_2 <- ggplot(res_db, aes(y = max_depth, x = min_child_weight, fill = rmse)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = "blue", # Choose low color
                       mid = "white", # Choose mid color
                       high = "red", # Choose high color
                       midpoint =mean(res_db$rmse), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Minimum Child Weight", y = "Max Depth", fill = "RMSE") # Set labels
g_2 # Generate plot


###### 2 - Gamma Tuning ######


gamma_vals <- c(0, 0.05, 0.1, 0.15, 0.2) # Create vector of gamma values

# Be Careful - This can take a very long time to run
set.seed(111111)
rmse_vec  <- rep(NA, length(gamma_vals))
for(i in 1:length(gamma_vals)){
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     
                     nfold = 5, # Use 5 fold cross-validation
                     
                     eta = 0.1, # Set learning rate
                     max.depth = 7, # Set max depth
                     min_child_weight = 10, # Set minimum number of samples in node to split
                     gamma = gamma_vals[i], # Set minimum loss reduction for split
                     
                     
                     
                     nrounds = 100, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
  ) # Set evaluation metric to use
  
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}

# Lets view our results to identify the value of gamma to use:

# Gamma results
# Join gamma to values
cbind.data.frame(gamma_vals, rmse_vec)


###### 3 - Subsample and Column sample Tuning ######

# Be Careful - This can take a very long time to run
subsample <- c(0.6, 0.7, 0.8, 0.9, 1) # Create vector of subsample values
colsample_by_tree <- c(0.6, 0.7, 0.8, 0.9, 1) # Create vector of col sample values

# Expand grid of tuning parameters
cv_params <- expand.grid(subsample, colsample_by_tree)
names(cv_params) <- c("subsample", "colsample_by_tree")
# Create vectors to store results
rmse_vec <- rep(NA, nrow(cv_params)) 
# Loop through parameter values
for(i in 1:nrow(cv_params)){
  set.seed(111111)
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     
                     nfold = 5, # Use 5 fold cross-validation
                     
                     eta = 0.1, # Set learning rate
                     max.depth = 7, # Set max depth
                     min_child_weight = 10, # Set minimum number of samples in node to split
                     gamma = 0, # Set minimum loss reduction for split
                     subsample = cv_params$subsample[i], # Set proportion of training data to use in tree
                     colsample_bytree = cv_params$colsample_by_tree[i], # Set number of variables to use in each tree
                     
                     nrounds = 150, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
  ) # Set evaluation metric to use
  
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}

# visualise tuning sample params

res_db <- cbind.data.frame(cv_params, rmse_vec)
names(res_db)[3] <- c("rmse") 
res_db$subsample <- as.factor(res_db$subsample) # Convert tree number to factor for plotting
res_db$colsample_by_tree <- as.factor(res_db$colsample_by_tree) # Convert node size to factor for plotting
g_4 <- ggplot(res_db, aes(y = colsample_by_tree, x = subsample, fill = rmse)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = "blue", # Choose low color
                       mid = "white", # Choose mid color
                       high = "red", # Choose high color
                       midpoint =mean(res_db$rmse), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Subsample", y = "Column Sample by Tree", fill = "RMSE") # Set labels
g_4 # Generate plot
```

```{r}
###### 4 - eta tuning ######

# Use xgb.cv to run cross-validation inside xgboost
set.seed(111111)
bst_mod_1 <- xgb.cv(data = dtrain, # Set training data
                    
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.3, # Set learning rate
                    max.depth = 7, # Set max depth
                    min_child_weight = 10, # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 0.9, # Set proportion of training data to use in tree
                    colsample_bytree =  0.9, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use


set.seed(111111)
bst_mod_2 <- xgb.cv(data = dtrain, # Set training data
                    
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.1, # Set learning rate
                    max.depth =  7, # Set max depth
                    min_child_weight = 10, # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 0.9 , # Set proportion of training data to use in tree
                    colsample_bytree = 0.9, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use

set.seed(111111)
bst_mod_3 <- xgb.cv(data = dtrain, # Set training data
                    
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.05, # Set learning rate
                    max.depth = 7, # Set max depth
                    min_child_weight = 10 , # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 0.9 , # Set proportion of training data to use in tree
                    colsample_bytree =  0.9, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use


set.seed(111111)
bst_mod_4 <- xgb.cv(data = dtrain, # Set training data
                    
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.01, # Set learning rate
                    max.depth = 7, # Set max depth
                    min_child_weight = 10, # Set minimum number of samples in node to split
                    gamma = 0.1, # Set minimum loss reduction for split
                    subsample = 0.9 , # Set proportion of training data to use in tree
                    colsample_bytree = 0.9, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use



set.seed(111111)
bst_mod_5 <- xgb.cv(data = dtrain, # Set training data
                    
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.005, # Set learning rate
                    max.depth = 7, # Set max depth
                    min_child_weight = 10, # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 0.9 , # Set proportion of training data to use in tree
                    colsample_bytree = 0.9, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
                    
) # Set evaluation metric to use
```

```{r}
# Extract results for model with eta = 0.3
pd1 <- cbind.data.frame(bst_mod_1$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.3, nrow(bst_mod_1$evaluation_log)))
names(pd1)[3] <- "eta"
# Extract results for model with eta = 0.1
pd2 <- cbind.data.frame(bst_mod_2$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.1, nrow(bst_mod_2$evaluation_log)))
names(pd2)[3] <- "eta"
# Extract results for model with eta = 0.05
pd3 <- cbind.data.frame(bst_mod_3$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.05, nrow(bst_mod_3$evaluation_log)))
names(pd3)[3] <- "eta"
# Extract results for model with eta = 0.01
pd4 <- cbind.data.frame(bst_mod_4$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.01, nrow(bst_mod_4$evaluation_log)))
names(pd4)[3] <- "eta"
# Extract results for model with eta = 0.005
pd5 <- cbind.data.frame(bst_mod_5$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.005, nrow(bst_mod_5$evaluation_log)))
names(pd5)[3] <- "eta"
# Join datasets
plot_data <- rbind.data.frame(pd1, pd2, pd3, pd4, pd5)
# Converty ETA to factor
plot_data$eta <- as.factor(plot_data$eta)
# Plot points
g_6 <- ggplot(plot_data, aes(x = iter, y = test_rmse_mean, color = eta))+
  geom_point(alpha = 0.5) +
  theme_bw() + # Set theme
  theme(panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Number of Trees", title = "RMSE v Number of Trees",
       y = "RMSE", color = "Learning \n Rate")  # Set labels
g_6

# Plot lines
g_7 <- ggplot(plot_data, aes(x = iter, y = test_rmse_mean, color = eta))+
  geom_smooth(alpha = 0.5) +
  theme_bw() + # Set theme
  theme(panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Number of Trees", title = "RMSE v Number of Trees",
       y = "RMSE", color = "Learning \n Rate")  # Set labels
g_7
```


### Now that it's tuned, we can rerun the XGBoost model!
```{r}
set.seed(111111)

bst_final <- xgboost(data = dtrain, # Set training data
                     eta = 0.005, # Set learning rate
                     max.depth =  15, # Set max depth
                     min_child_weight = 1, # Set minimum number of samples in node to split
                     gamma = 0.1, # Set minimum loss reduction for split
                     #g_4 
                     subsample =  1, # Set proportion of training data to use in tree
                     colsample_bytree = .6, # Set number of variables to use in each tree
                     
                     nrounds = 100, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
                     
) # Set evaluation metric to use

```

#### We can extract the variables of importance from the tuned XGBoost model. We can see that the top five most important variables are field goal percentage, effective field goal percentage, total rebounds, assists, and turnovers.
#### Even though effective field goal percentage is a weighted version of field goal percentage, meaning 3-point shots hold more value than 2-point shots, regular field goal percentage was given more importance in this model. 
#### Because of the unweighted field goal percentage being more important, effective field goal percentage was removed from the discussion all together.
```{r}
# Extract importance
imp_mat_final <- xgb.importance(model = bst_final)
# Plot importance (top 10 variables)
xgb.plot.importance(imp_mat_final, top_n = 10)
```


#### And now it's time to visualize!

#### The first viz is a basic scatter plot looking at winning percentage versus average points per game.
```{r}
# Create points vs. win % plot

ggplot(bigEast, 
       aes(x = points, win_percentage)) +
  geom_point(color = 'red') +
  geom_point(data = teams21, color = 'blue') +
  labs(y = "Winning %", # Add labels
       x = "Average PPG",
       title = "Winning Percentage by Average Points per Game",
       subtitle = "Big East All Time Stats") +
  dark_theme_bw() + # Set theme
  theme( # Modify plot elements
    axis.text = element_text(size = 10), # Change Axis text size
    axis.title.x = element_text(size = 12), # Change x axis title size
    axis.title.y = element_text(size = 12), # Change y axis title size 
    plot.title = element_text(size = 16), # Change plot title size
    plot.subtitle = element_text(size = 14), # Change plot subtitle size
    plot.caption = element_text(size = 10), # Change plot caption size
    panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank())
#+  xlim(0, .5)
```

#### This includes all Big East teams since 1979 with the 2021-2022 teams
highlighted in blue.
#### There is a positive correlation between these to variables.
#### The 2021-22 teams are fairly spread out; there were some teams that were very successful while others were not.


#### We can also look at points scored versus opponent points scored against. This is a basis representation of offense and defense. 
```{r}
# Merge logos and data
plot_dat <- merge(teams21, teamLogos, by.x = "school", by.y = "team")


# Create points plot
pointsOpponentPoints <- ggplot(plot_dat, # Set dataset
              aes(x = points, y = opponent_points)) + # Set aesthetics
  geom_point(alpha = 0.3) + # Set geom point
  geom_image(image = plot_dat$url, asp = 16/9) + # Add logos
  labs(y = "Average PPG Scored by Opponents", # Add labels
       x = "Average PPG",
       title = "Average PPG Scored by Opponents vs. Average PPG",
       subtitle = "Big East (2021-22)") +
  dark_theme_bw() + # Set theme
  theme( # Modify plot elements
    axis.text = element_text(size = 10), # Change Axis text size
    axis.title.x = element_text(size = 12), # Change x axis title size
    axis.title.y = element_text(size = 12), # Change y axis title size 
    plot.title = element_text(size = 16), # Change plot title size
    plot.subtitle = element_text(size = 14), # Change plot subtitle size
    plot.caption = element_text(size = 10), # Change plot caption size
    panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) # Remove grid

pointsOpponentPoints
```
#### Teams that were successful overall included the University of
Villanova and University of Connecticut. Both teams had stellar defenses which allowed them to keep their opponents scoring to a minimum. 
#### On the other hand, teams like Butler University and Saint
John’s University (NY) were not as successful. On average, Butler was letting their opponents score more points than then during a game. 
#### While St. John’s had a powerful offense and scored a lot of points, they ended up letting their opponents score close to the same amount.


#### Next, we look at the variables on importance: field goal percent against rebounds and assists against turnovers.
#### This is a good indicator of success as the more a team has the ball in their hands, the more shots they take and score—therefore boosting their winning percentage as points lead to wins as we saw earlier.
```{r}
# Create FG% vs. REB plot
fieldGoalPercentTotalRebounds <- ggplot(plot_dat, # Set dataset
              aes(x = field_goal_percentage, y = total_rebounds)) + # Set aesthetics
  geom_point(alpha = 0.3) + # Set geom point
  geom_image(image = plot_dat$url, asp = 16/9) + # Add logos
  labs(y = "Total Rebounds", # Add labels
       x = "FG %",
       title = "Average Field Goal Percentage vs. Average Total Rebounds per Game",
       subtitle = "Big East (2021-22)") +
  dark_theme_bw() + # Set theme
  theme( # Modify plot elements
    axis.text = element_text(size = 10), # Change Axis text size
    axis.title.x = element_text(size = 12), # Change x axis title size
    axis.title.y = element_text(size = 12), # Change y axis title size 
    plot.title = element_text(size = 16), # Change plot title size
    plot.subtitle = element_text(size = 14), # Change plot subtitle size
    plot.caption = element_text(size = 10), # Change plot caption size
    panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) # Remove grid

fieldGoalPercentTotalRebounds
```
#### In the Big East, UConn was successful at getting rebounds and turning them into points, whereas Villanova’s field goal success came from a different offensive aspect.


#### Lastly, we will look at turnovers against assists.
```{r}
# Create TO vs. AST plot
TurnoversAssists <- ggplot(plot_dat, # Set dataset
              aes(x = turnovers, y = assists)) + # Set aesthetics
  geom_point(alpha = 0.3) + # Set geom point
  geom_image(image = plot_dat$url, asp = 16/9) + # Add logos
  labs(y = "Assists", # Add labels
       x = "Turnovers",
       title = "Average Turnovers vs. Average Assists per Game",
       subtitle = "Big East (2021-22)") +
  dark_theme_bw() + # Set theme
  theme( # Modify plot elements
    axis.text = element_text(size = 10), # Change Axis text size
    axis.title.x = element_text(size = 12), # Change x axis title size
    axis.title.y = element_text(size = 12), # Change y axis title size 
    plot.title = element_text(size = 16), # Change plot title size
    plot.subtitle = element_text(size = 14), # Change plot subtitle size
    plot.caption = element_text(size = 10), # Change plot caption size
    panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) # Remove grid

TurnoversAssists

# Turn off dark mode
invert_geom_defaults()
```
#### The average amount of turnovers per game was around 12. 
#### Teams like Georgetown and Creighton were above average, turning the ball over frequently, and they did not make up for it with assists. 
#### Saint John’s had the most assists which is one thing that allowed them to score so many points, but they also had a significant number of turnovers which is a contributor to their points scored against. 
#### Villanova was well under the conference average of turnover, but also did not have a great number of assists. 


#### From this information, it can be inferred that Villanova’s game plan was different from the rest of the conference, and that may have been what led them to their success in the conference championship.