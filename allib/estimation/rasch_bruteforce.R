library(tidyverse)

df.modify <- function(freq.df, n.pos, n.neg){
  # From Python the rows for n_00...0 are missing
  # We add them with the following statement
  df <- freq.df %>% 
    add_row(count=n.pos, positive=1) %>%
    add_row(count=n.neg, positive=0) %>%
    mutate_at(vars(-c("count")), ~replace(., is.na(.), 0))
  return(df)
}

rasch.bf.comb <- function(freq.df, N,proportion=0.1, tolerance=1e-5){
  # Calculate the number of documents that are not read
  count.found <- sum(freq.df$count)
  N0 <- N - sum(freq.df$count) 
  # Calculate the start values for the n.pos and n.neg
  deviances <- list()
  models <- list()
  estimates <- list()
  model.min <- glm(count ~ ., "poisson", df.modify(freq.df, 0, N0))
  deviance.min <- Inf
  for(i in 0:N0){
    n.pos <- i
    n.neg <- N0 - i
    df <- df.modify(freq.df, n.pos, n.neg)
    model <- glm(count ~ ., "poisson", df)
    if(deviance.min > model$deviance){
      deviance.min <- model$deviance
      model.min <- model
    }
    estimate <- fitted(model) %>% 
                  unlist(use.names=F) %>% 
                  as.vector() %>% 
                  tail(2) %>% head(1)
    models[[(i+1)]] <- model
    deviances[[(i+1)]] <- model$deviance
    estimates[[(i+1)]] <- estimate
  }
  ret <- list(
    model.min = model.min,
    models = models,
    deviances = (deviances %>% unlist(use.names = F) %>% as.vector()),
    estimates = (estimates %>% unlist(use.names = F) %>% as.vector())
  )
  return(ret)
}

rasch.bf.table <- function(result){
  x <- (1:length(result$deviances)) - 1
  df <- data.frame(
    n.pos = x,
    deviance = result$deviances,
    estimate = result$estimates ,
    difference = result$estimates - x
  )
  return(df)
}




rasch.csv <- function(filename){
  # For reading the csv files with matrices 
  # when manually reading and testing.
  # Usage: df <- rasch.csv("matrix3_iteration_11.csv")
  df <- read_csv(filename, col_types = cols(X1= col_skip()))
  return(df)
}
rasch.em.horizon <- function(freq.df, # The dataframe (from python or rasch.csv)
                             N, # The dataset size 
                             proportion=0.1){ # Initial ratio of positive documents
  # This is the function that is called from Python
  # Gather the positive part of the table
  df.pos <- freq.df %>% filter(positive==1)
  # Calculate the number of positive documents
  count.found <- sum(df.pos$count)
  model <- rasch.em.comb(freq.df, N, proportion=proportion)
  fv <- fitted(model) %>% unlist(use.names=F) %>% as.vector()
  # Estimates for n00...0 are located at the last two members of the list
  estimates <-  tail(fv, 2)
  # Return the estimated number of positive documents
  df <- data.frame(estimate=count.found + estimates[1])
  return(df)
}