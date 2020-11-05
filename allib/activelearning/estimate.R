library(Rcapture)

get_abundance <- function(df){
    dat <- closedp(df)
    res <- dat$results
    return(as.data.frame(res))
}