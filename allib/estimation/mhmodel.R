library(Rcapture)
suppressPackageStartupMessages(library(tidyverse))

get_abundance <- function(df) {
  dat <- closedpCI.0(df, t = Inf, m = "Mh", h = "LB")
  res <- dat$CI
  return(as.data.frame(res))
}
getval <- function(expr){
  tryCatch(expr, error=function(e){return(NA)})
}
get_abundance_orig_point <- function(df) {
  dat <- closedpCI.0(df, t = Inf, m = "Mh", h = "LB")
  point_est <- dat$results[1]
  infCI <- dat$CI[2]
  supCI <- dat$CI[3]
  infoCI <- dat$CI[4]
  if (is.na(supCI) || supCI >= point_est) {
    res <- list(abundance = point_est,
                infCL = infCI,
                supCL = supCI,
                infoCI = infoCI)
  } else {
    res <- list(abundance = point_est,
                infCL = point_est,
                supCL = point_est,
                infoCI = point_est)
  }
  return(as.data.frame(res))
}
get_abundance_eta <- function(df) {
  dat <- closedpCI.0(df, m = "Mh", h = "LB")
  res <- dat$CI
  return(as.data.frame(res))
}

get_abundance_ll <- function(df) {
  dat <- closedpCI.t(df, mX = "[12,13,14,23,5]")
  res <- dat$CI
  return(as.data.frame(res))
}

rcapture.df <- function(df, positive_selection = 1) {
  pos.df <- df %>% filter(positive == positive_selection)
  rcapture.df <- pos.df %>%
    select(!ends_with("positive")) %>%
    select(starts_with("learner") | matches("count"))
  return(rcapture.df)
}

rasch.csv <- function(filename) {
  # For reading the csv files with matrices
  # when manually reading and testing.
  # Usage: df <- rasch.csv("design_matrix_11.csv")
  df <- read_csv(filename, col_types = cols(X1 = col_skip()))
  return(df)
}

rasch_to_abundance <- function(df) {
  rcp.df <- rcapture.df(df)
  result <- closedpCI.0(rcp.df, dfreq = TRUE, t = Inf, m = "Mh", h = "LB")
  return(result)
}
