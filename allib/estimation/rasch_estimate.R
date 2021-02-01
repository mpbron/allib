get_rasch <- function(df){
  model <- glm(
    formula = count ~ ., 
    family = poisson(link = "log"), 
    data = df
  )
  coefficients = coef(model)
  stderrs = sqrt(diag(vcov(model)))
  intercept = coefficients[1]
  intercept.err = stderrs[1]
  estimate.missing = exp(intercept)
  estimate.stderr = exp(intercept.err)
  df <- data.frame(
    estimate = c(estimate.missing),
    stderr = c(estimate.stderr))
  return(df)
}