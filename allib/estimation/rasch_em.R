library(tidyverse)

rasch.estimate <- function(df, count.expectation){
  df.t <- as_tibble(df)
  df.m <- df.t %>% 
            add_row(count=count.expectation) %>% 
            mutate_all(
              ~replace(., is.na(.), 0)
            )
  model <- glm(
    formula = count ~ ., 
    family = poisson(link = "log"), 
    data = df.m
  )
  # Get the coefficients from the model
  coefficients <- coef(model) %>% unlist(use.names = F)
  stderrs = sqrt(diag(vcov(model))) %>% unlist(use.names = F)
  
  # Get the intercept of the formula (the estimate is exp(intercept))
  intercept = coefficients[1]
  intercept.err = stderrs[1]
  
  # Calculate the number of missing papers
  estimate.missing = exp(intercept)
  ret <- list(
    estimate.missing=estimate.missing,
    deviance=model$deviance)
  return(ret)
}
create.design.matrix<- function(learners){
  learners.n <- dim(learners)
  
}

rasch.expectation <- function(df.pos, df.neg, dataset.size, missing.pos, missing.neg){
  count.read <- sum(df.pos$count) + sum(df.neg$count)
  count.unread <- dataset.size - count.read
  random.partition <- rmultinom(1, count.unread, c(missing.pos, missing.neg))
  return(list(missing.pos=random.partition[1], missing.neg=random.partition[2]))
}

rasch.maximization <- function(df.pos, df.neg, dataset.size, missing.pos, missing.neg){
  estimate.pos <- rasch.estimate(df.pos, missing.pos)
  estimate.neg <- rasch.estimate(df.neg, missing.neg)
  deviance <- rasch.estimate(df.pos, missing.)
  return(list(missing.pos=estimate.pos, missing.neg=estimate.neg))
}
calculate.ratio <- function(exp.r){
  ratio <- exp.r$missing.pos / (exp.r$missing.pos + exp.r$missing.neg)
  return(ratio) 
}

rasch.em <- function(df.pos, df.neg, dataset.size, missing.pos=1, missing.neg=1, maxit=1000, epsilon=1e-5){
  converged <- FALSE
  
  e.result <- list(missing.pos=missing.pos, missing.neg=missing.neg)
  m.result <- list(missing.pos=missing.pos, missing.neg=missing.neg)
  
  e.results <- list()
  m.results <- list()
  i.max <- 1
  
  for(i in 1:maxit){
    m.result.prev <- m.result
    e.result.prev <- e.result
    
    e.ratio.prev <- calculate.ratio(e.result.prev)
    m.ratio.prev <- calculate.ratio(m.result.prev)
    
    e.result <- rasch.expectation(
      df.pos, df.neg, dataset.size, 
      m.result$missing.pos, m.result$missing.neg)
    m.result <- rasch.maximization(
      df.pos, df.neg, dataset.size, 
      e.result$missing.pos, e.result$missing.neg)

    e.ratio <- calculate.ratio(e.result)
    m.ratio <- calculate.ratio(m.result)
    
    if(abs(e.ratio - m.ratio) < epsilon){
      converged <- TRUE
      break
    }
  
    e.results[[i]] <- e.ratio
    m.results[[i]] <- m.ratio
    i.max <- i
  }
  if(!converged){
    warning("Did not converge")
  }
  result.list <- list(
    df = data.frame(
      pos_estimate = m.result$missing.pos,
      neg_estimate = m.result$missing.neg),
    ratios.e = unlist(e.results),
    ratios.m = unlist(m.results)
  )
  return(result.list$df)
}

df.modify <- function(freq.df, n.pos, n.neg){
  # From Python the rows for n_00...0 are missing
  # We add them with the following statement
  df <- freq.df %>% 
    add_row(count=n.pos, positive=1) %>%
    add_row(count=n.neg, positive=0) %>%
    mutate_at(vars(-c("count")), ~replace(., is.na(.), 0))
  return(df)
}

rasch.em.comb <- function(freq.df, N, proportion=0.05, tolerance=1e-6){
  # Calculate the start values for the n.pos and n.neg
  n.pos   <- round(proportion*N)
  n.neg   <- round((1-proportion)*N)
  
  # Add the missing rows for n_00...0
  df      <- df.modify(freq.df, n.pos, n.neg)
  s       <- c(
    rep(T, nrow(freq.df)), # Do not contain n_00...0
    rep(F, nrow(df) - nrow(freq.df)) #Do contain n_00...0
  )
  N0      <- N - sum(df$count[s]) # The number of documents that are not read
  # Copy the data frame to new variable that can be manipulated
  df.em     <- df 
  
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
  df <- read_csv(filename, col_types = cols(X1= col_skip()))
  return(df)
}
rasch.em.horizon <- function(freq.df, N, proportion=0.05){
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