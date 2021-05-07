from .catalog import ALConfiguration, FEConfiguration
from .ensemble import naive_bayes_estimator, svm_estimator, tf_idf5000, rasch_estimator, al_config_entropy, al_config_ensemble_prob

AL_REPOSITORY = {
    ALConfiguration.NaiveBayesEstimator :  naive_bayes_estimator,
    ALConfiguration.SVMEstimator : svm_estimator,
    ALConfiguration.RaschEstimator: rasch_estimator,
    ALConfiguration.EntropySamplingNB: al_config_entropy,
    ALConfiguration.ProbabilityEnsemble: al_config_ensemble_prob,
}

FE_REPOSITORY = {
    FEConfiguration.TFIDF5000 : tf_idf5000
}