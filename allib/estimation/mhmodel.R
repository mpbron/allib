library(Rcapture)
suppressPackageStartupMessages(library(tidyverse))

get_abundance <- function(df){
    dat <- closedpCI.0(df, t = Inf, m = "Mh", h = "LB")
    res <- dat$CI
    return(as.data.frame(res))
}
get_abundance_eta <- function(df){
    dat <- closedpCI.0(df, m = "Mh", h = "LB")
    res <- dat$CI
    return(as.data.frame(res))
}

rcapture.df <- function(df, positive_selection=1){
    pos.df <- df %>% filter(positive == positive_selection)
    rcapture.df  <- pos.df %>%
      select(!ends_with("positive")) %>%
      select(starts_with("learner") | matches("count"))
    return(rcapture.df)
}

rasch.csv <- function(filename){
  # For reading the csv files with matrices 
  # when manually reading and testing.
  # Usage: df <- rasch.csv("design_matrix_11.csv")
  df <- read_csv(filename, col_types = cols(X1= col_skip()))
  return(df)
}

rasch_to_abundance <- function(df){
  rcp.df <- rcapture.df(df)
  result <- closedpCI.0(rcp.df,dfreq = TRUE, t = Inf, m = "Mh", h = "LB")
  return(result)
}

