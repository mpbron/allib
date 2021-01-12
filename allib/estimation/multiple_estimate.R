library(Rcapture)

get_abundance <- function(df){
    dat <- closedpMS.t(df)
    res <- dat$results
    return(as.data.frame(res))
}