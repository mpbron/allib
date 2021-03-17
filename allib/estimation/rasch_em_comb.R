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

rasch.em.comb <- function(freq.df, N, proportion=0.1, tolerance=1e-5){
  # Calculate the number of documents that are not read
  N0 <- N - sum(freq.df$count) 
  # Calculate the start values for the n.pos and n.neg
  n.pos <- round(proportion*N0)
  n.neg <- round((1-proportion)*N0)
  
  # Add the missing rows for n_00...0
  df <- df.modify(freq.df, n.pos, n.neg)
  s <- c(
    rep(T, nrow(freq.df)), # Do not contain n_00...0
    rep(F, nrow(df) - nrow(freq.df)) #Do contain n_00...0
  )
  
  # Copy the data frame to new variable that can be manipulated
  df.em <- df 
  
  # Fit initial log linear model and calculate deviance
  mstep   <- glm(count ~ ., "poisson", df.em)
  devold  <- mstep$deviance
  tol <- devold
  
  while(tol > tolerance){
    # Calculate fitted frequencies
    mfit  <- fitted(mstep, "response")
    
    # Adjust the frequencies for n_00...0
    efit  <- df.em$count
    efit[!s] <- mfit[!s] * N0 / sum(mfit[!s]) 
    
    # Store new frequencies in data frame
    df.em$count <- efit
    
    # Fit log linear model and calculate deviance
    mstep <- glm(count ~ ., "poisson", df.em)
    devnew <- mstep$deviance
    
    # Determine if we have converged
    tol <- devold - mstep$deviance
    devold <- mstep$deviance
  }
  return(mstep)
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