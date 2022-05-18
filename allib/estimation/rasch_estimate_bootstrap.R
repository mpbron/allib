library(purrr)

rasch.single <- function(df, epsilon=0.1){
  # Copy the Frequency Table to a new variable
  count.df <- df
  # Add a small amount $\epsilon$ to the counts to migitate
  # explosion of estimates
  count.df$count <- count.df$count + epsilon
  # Estimate the Rasch model
  model <- glm(
    formula = count ~ ., 
    family = poisson(link = "log"), 
    data = count.df
  )
  # Get the coefficients from the model
  coefficients <- coef(model) %>% unlist(use.names = F)
  stderrs = sqrt(diag(vcov(model))) %>% unlist(use.names = F)
  
  # Get the intercept of the formula (the estimate is exp(intercept))
  intercept = coefficients[1]
  intercept.err = stderrs[1]
  
  # Calculate the estimate
  estimate.missing = exp(intercept)
  estimate.stderr = exp(intercept.err)
  
  # Construct a dataframe that contains the results 
  results <- data.frame(
    estimate = c(estimate.missing),
    stderr = c(estimate.stderr))
  return(results)
}

rasch.single.model <- function(df, epsilon=0.1){
  # Copy the Frequency Table to a new variable
  count.df <- df
  # Add a small amount $\epsilon$ to the counts to migitate
  # explosion of estimates
  count.df$count <- count.df$count + epsilon
  # Estimate the Rasch model
  model <- glm(
    formula = count ~ ., 
    family = poisson(link = "log"), 
    data = count.df
  )
  # Get the coefficients from the model
  coefficients <- coef(model) %>% unlist(use.names = F)
  stderrs = sqrt(diag(vcov(model))) %>% unlist(use.names = F)
  
  # Get the intercept of the formula (the estimate is exp(intercept))
  intercept = coefficients[1]
  intercept.err = stderrs[1]
  
  # Calculate the estimate
  estimate.missing = exp(intercept)
  estimate.stderr = exp(intercept.err)
  
  # Construct a dataframe that contains the results 
  results <- list(
    estimate = estimate.missing,
    stderr = estimate.stderr,
    model=model)
  return(results)
}


rasch.nonparametric <- function(df, it=2000, confidence=0.95, epsilon=0.1){
  count.found <- sum(df$count)
  counts.orig <- df$count + epsilon
  counts.bootstrap <- rmultinom(it, count.found, counts.orig)
  # Estimate n_00...0 using the Rasch model
  estimate.missing <- rasch.single(df, epsilon=epsilon)$estimate
  estimates <- list()
  for (col.idx in 1:dim(counts.bootstrap)[2]) {
    count.bootstrap <- counts.bootstrap[,col.idx]
    df.adjusted <- df
    df.adjusted$count <- count.bootstrap
    model.results <- rasch.single(df.adjusted)
    estimates[[col.idx]] <- model.results$estimate
  }
  # Determine using the percentile method a 
  # 95% confidence interval
  results <- unlist(estimates, use.names = F)
  sorted <- sort(results)
  bounds <- unlist(quantile(sorted, c(1-confidence, confidence)), use.names=F)
  result.df <- data.frame(
    estimate = c(count.found + estimate.missing),
    lowerbound = c(count.found + bounds[1]),
    upperbound = c(count.found + bounds[2])
  )
  return(result.df)
}

rasch.parametric <- function(df, it=2000, confidence=0.95, epsilon=0.1){
  # Copy the Frequency Table to a new variable
  count.df <- df
  count.found <- sum(count.df$count)
  # Estimate n_00...0 using the Rasch model
  main.model <- rasch.single.model(df, epsilon)$estimate
  # Gather the counts of all rows and add the estimation for n_00..0
  counts.model <- append(df$count, main.model)
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
    df.adjusted <- df
    df.adjusted$count <- count.bootstrap
    model.results <- rasch.single(df.adjusted)
    estimates[[col.idx]] <- model.results$estimate
  }
  # Determine using the percentile method a 
  # 95% confidence interval
  results <- unlist(estimates, use.names = F)
  sorted <- sort(results)
  bounds <- unlist(quantile(sorted, c(1-confidence, confidence)), use.names=F)
  result.df <- data.frame(
    estimate = c(count.found + main.model),
    lowerbound = c(count.found + bounds[1]),
    median = c(count.found + median(sorted)), 
    upperbound = c(count.found + bounds[2])
  )
  rownames(result.df) <- NULL
  return(result.df)
}