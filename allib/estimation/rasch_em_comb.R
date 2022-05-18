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
  N0 <- N - sum(freq.df$count) # N0  <- N - sum(d$Freq[s])
  # Calculate the start values for the n.pos and n.neg
  n.pos <- proportion*N0 
  n.neg <- (1-proportion)*N0
  
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
    mstep <- glm(count ~ ., "poisson", df.em, start=coef(mstep))
    devnew <- mstep$deviance
    
    # Determine if we have converged
    tol <- abs(devold - mstep$deviance)
    devold <- mstep$deviance
  }
  return(mstep)
}


rasch.em.comb.conv <- function(freq.df, N, proportion=0.1, tolerance=1e-5){
  # Calculate the number of documents that are not read
  N0 <- N - sum(freq.df$count) # N0  <- N - sum(d$Freq[s])
  # Calculate the start values for the n.pos and n.neg
  n.pos <- proportion*N0 
  n.neg <- (1-proportion)*N0
  
  # Add the missing rows for n_00...0
  df <- df.modify(freq.df, n.pos, n.neg)
  s <- df %>% 
    select(!ends_with("positive")) %>% 
    select(starts_with("learner")) %>% 
    rowwise() %>% 
    mutate(
      selection=(sum(c_across(starts_with("learner"))) > 0)) %>% 
    select("selection") %>% 
    unlist() %>% 
    as.vector()
  s.pos <- df$positive == 1 & !s
  s.neg <- df$positive == 0 & !s
  
  # Copy the data frame to new variable that can be manipulated
  df.em <- df 
  
  # Fit initial log linear model and calculate deviance
  mstep   <- glm(count ~ ., "poisson", df.em)
  mfit.old <- fitted(mstep, "response")
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
    tol <- abs(mfit[s.pos] - efit[s.pos])
    mfit.old <- mfit
  }
  return(mstep)
}

L2 <- function(b0, lambda = 0, D, counts, N) {
  tol <- 1
  tolerance <- 1e-8
  while (tol > tolerance) {
    mu   <- exp(D %*% b0)
    db   <- crossprod(D, counts) - crossprod(D, mu) - 2 * c(0, lambda * b0[-1])
    I    <- crossprod(D, diag(c(mu))) %*% D + 2 * diag(c(0, rep(lambda, length(b0) - 1)))
    invI <- tryCatch(solve(I), error = function(e) e)
    if(inherits(invI, "error")){
      lambda <- g(lambda)
      next
    }
    b <- b0 + invI %*% db
    tol  <- sum(abs(b0 - b))
    b0   <- b
  }
  return(b)
}

calc.dev <- function(y, yhat){
  2 * sum(y[y > 0] * log(y[y > 0] / yhat[y > 0]), na.rm = T) - 2 * sum(y - yhat, na.rm = T)
}
g <- function(x){
  ifelse(x == 0, 1e-6, 10 * x)
}



rasch.ridge.em.comb <- function(freq.df, N, proportion=0.1, tolerance=1e-5){
  N0 <- N - sum(freq.df$count) # N0  <- N - sum(d$Freq[s])
  # Calculate the start values for the n.pos and n.neg
  n.pos <- proportion*N0 
  n.neg <- (1-proportion)*N0
  
  # Add the missing rows for n_00...0
  df <- df.modify(freq.df, NA, NA) %>% 
    arrange(positive, rev(across(starts_with("learner")))) %>%
    add_column(intercept=1, .before="learner_0")
  df.mat <- df %>% select(-count)
  s <- df %>% 
    select(!ends_with("positive")) %>% 
    select(starts_with("learner")) %>% 
    rowwise() %>% 
    mutate(
      selection=(sum(c_across(starts_with("learner"))) > 0)) %>% 
    select("selection") %>% 
    unlist() %>% 
    as.vector()
  s.pos <- df$positive == 1 & !s
  s.neg <- df$positive == 0 & !s
  D <- as.matrix(df.mat)
  efit     <- df$count
  efit[s.pos] <- n.pos
  efit[s.neg] <- n.neg
  beta     <- L2(
    b0 = c(log(N), rep(0, ncol(D) - 1)), 
    D = D, N = N, counts = efit
  )
  devold   <- calc.dev(df$count, exp(D %*% beta))
  tol      <- 1
  while(tol > tolerance){
    mfit     <- exp(D %*% beta)
    efit[!s] <- mfit[!s] * N0 / sum(mfit[!s]) 
    beta     <- L2(b0 = beta, D = D, N = N, counts = efit)
    devnew   <- calc.dev(df$count, exp(D %*% beta))
    tol      <- devold - devnew
    devold   <- devnew
  }
  p <- as.list(mfit) %>% unlist()
  W <- diag(p)
  cvm <- solve(t(D) %*% W %*% D)
  SE <- sqrt(diag(cvm))
  ret <- list(
    coefficients=beta,
    fitted=mfit,
    deviance=devold,
    est.pos=mfit[s.pos,1],
    est.neg=mfit[s.neg,1],
    std.err = SE,#SE,
    cvm = cvm#cvm
  )
  return(ret)
}

Vcov.glm = function(object, dispersion = NULL, ...) {
  if (p <- len(coefficients)) {
    dispersion = 1
    p1 = seq_len(p)
    nm <- names(object$coefficients[object$qr$pivot[p1]])
    covmat = dispersion * chol2inv(object$qr$qr[p1, p1, drop = FALSE])
    dimnames(covmat) = list(nm, nm)
    return(covmat)
  } else return(numeric(0))
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

rasch.em.bootstrap.horizon <- function(freq.df, # The dataframe (from python or rasch.csv)
                             N, # The dataset size 
                             proportion=0.5, it=2000, confidence=0.95){ # Initial ratio of positive documents
  # This is the function that is called from Python
  # Gather the positive part of the table
  df.pos <- freq.df %>% filter(positive==1)
  # Calculate the number of positive documents
  count.found <- sum(df.pos$count)
  
  # Fit the initial model through EM
  model <- rasch.em.comb(freq.df, N, proportion=proportion)
  fv <- fitted(model) %>% unlist(use.names=F) %>% as.vector()
  model.estimates <-  tail(fv, 2)
  # Return the estimated number of positive documents
  model.estimate.pos <- count.found + model.estimates[1]
  # Calculate the confidence interval
  counts.bootstrap <- rmultinom(it, N, fv)
  df <- df.modify(freq.df, NA, NA)
  s <- c(
    rep(T, nrow(freq.df)), # Do not contain n_00...0
    rep(F, nrow(df) - nrow(freq.df)) #Do contain n_00...0
  )
  estimates <- list()
  for (col.idx in 1:dim(counts.bootstrap)[2]) {
    count.bootstrap <- counts.bootstrap[,col.idx]
    df.adjusted <- df
    df.adjusted$count <- count.bootstrap
    bootstrap.model <- glm(count ~ ., "poisson", df.adjusted)
    bootstrap.fv <- fitted(bootstrap.model) %>% unlist(use.names=F) %>% as.vector()
    # Estimates for n00...0 are located at the last two members of the list
    bootstrap.estimates <- tail(bootstrap.fv, 2)
    # Return the estimated number of positive documents
    estimates[[col.idx]] <- count.found + bootstrap.estimates[1]
  }
  results <- unlist(estimates, use.names = F)
  sorted <- sort(results)
  bounds <- unlist(quantile(sorted, c(1-confidence, confidence)), use.names=F)
  result.df <- data.frame(
    estimate = model.estimate.pos,
    lowerbound = bounds[1],
    upperbound = bounds[2])
  return(result.df)
}


rasch.em.ms <- function(f.em, freq.df, N){
  lowest <- f.em(freq.df, N, proportion=0.5)
  proportions = c(0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 0)
  for(pp in proportions){
    result <- f.em(freq.df, N, proportion=pp)
    if(lowest$deviance > result$deviance){
      lowest <- result
    }
  }
  return(lowest)
}


rasch.ridge.em.horizon <- function(freq.df, # The dataframe (from python or rasch.csv)
                             N, # The dataset size 
                             proportion=0.1){ # Initial ratio of positive documents
  # This is the function that is called from Python
  # Gather the positive part of the table
  df.pos <- freq.df %>% filter(positive==1)
  # Calculate the number of positive documents
  count.found <- sum(df.pos$count)
  results <- rasch.em.ms(rasch.ridge.em.comb, freq.df, N)
  # Return the estimated number of positive documents
  df <- data.frame(estimate=count.found + results$est.pos)
  return(df)
}
rasch.ridge.em.horizon.parametric <- function(freq.df, # The dataframe (from python or rasch.csv)
                                   N, # The dataset size 
                                   it=100, confidence=0.95){ # Initial ratio of positive documents
  
  # Store the original freq.df
  orig.df <- freq.df
  
  # Find the prediction value
  results <- rasch.ridge.em.horizon(freq.df, N)
  est.pos <- results[1,1]

  # This is the function that is called from Python
  # Gather the positive part of the table
  df.pos <- freq.df %>% filter(positive==1)
  df.neg <- freq.df %>% filter(positive==0)
  # Calculate the number of positive documents
  count.found <- sum(df.pos$count)
  
  counts.model <- append(freq.df$count, est.pos)
  counts.model.sum <- sum(counts.model)
  # Sample with replacement of size n from this multinomial distribution. 
  # Remove the observation that correspond with cell 00..0. 
  counts.bootstrap <- rmultinom(it, 
                                counts.model.sum, 
                                counts.model) %>% head(-1)
  # Calculate the n_00...0 for all the bootstrap counts
  estimates <- list()
  for (col.idx in 1:dim(counts.bootstrap)[2]) {
    count.bootstrap <- counts.bootstrap[,col.idx]
    df.adjusted <- orig.df
    df.adjusted$count <- count.bootstrap
    model.results <- rasch.ridge.em.horizon(df.adjusted, N)
    estimates[[col.idx]] <- model.results[1,1]
  }
  # Determine using the percentile method a 
  # 95% confidence interval
  results <- unlist(estimates, use.names = F)
  sorted <- sort(results)
  bounds <- unlist(quantile(sorted, c(1-confidence, confidence)), use.names=F)
  result.df <- data.frame(
    estimate = est.pos,
    lowerbound = bounds[1],
    upperbound = bounds[2]
  )
  return(result.df)
  
  # Return the estimated number of positive documents
  df <- data.frame(estimate=count.found + results$est.pos)
  return(df)
}




