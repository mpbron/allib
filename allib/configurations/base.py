from .catalog import ALConfiguration, FEConfiguration
from .ensemble import naive_bayes_estimator, svm_estimator, tf_idf5000, rasch_estimator

AL_REPOSITORY = {
    ALConfiguration.NaiveBayesEstimator :  naive_bayes_estimator,
    ALConfiguration.SVMEstimator : svm_estimator,
    ALConfiguration.RaschEstimator: rasch_estimator,
}

FE_REPOSITORY = {
    FEConfiguration.TFIDF5000 : tf_idf5000
}