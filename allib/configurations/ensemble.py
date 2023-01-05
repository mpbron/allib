from typing import Any, Mapping
from ..module import ModuleCatalog as Cat


def add_identifier(config: Mapping[str, Any], identifier: str) -> Mapping[str, Any]:
    id_dict = {"identifier": identifier}
    new_dict = {**config, **id_dict}
    return new_dict


def set_batch_size(config: Mapping[str, Any], batch_size: int) -> Mapping[str, Any]:
    id_dict = {"batch_size": batch_size}
    new_dict = {**config, **id_dict}
    return new_dict


al_config_svm = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.LABELMAXIMIZER,
    "label": "Relevant",
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear",
            "probability": True,
            "class_weight": "balanced",
        },
        "task": Cat.ML.Task.BINARY,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
}

tf_idf_full = {
    "datatype": Cat.FE.DataType.TEXTINSTANCE,
    "vec_type": Cat.FE.VectorizerType.STACK,
    "vectorizers": [
        {
            "vec_type": Cat.FE.VectorizerType.SKLEARN,
            "sklearn_vec_type": Cat.FE.SklearnVecType.TFIDF_VECTORIZER,
            "sklearn_config": {},
        }
    ],
}

tf_idf5000 = {
    "datatype": Cat.FE.DataType.TEXTINSTANCE,
    "vec_type": Cat.FE.VectorizerType.STACK,
    "vectorizers": [
        {
            "vec_type": Cat.FE.VectorizerType.SKLEARN,
            "sklearn_vec_type": Cat.FE.SklearnVecType.TFIDF_VECTORIZER,
            "sklearn_config": {"max_features": 5000},
        }
    ],
}

tf_idf_autotar = {
    "datatype": Cat.FE.DataType.TEXTINSTANCE,
    "vec_type": Cat.FE.VectorizerType.STACK,
    "vectorizers": [
        {
            "vec_type": Cat.FE.VectorizerType.SKLEARN,
            "sklearn_vec_type": Cat.FE.SklearnVecType.TFIDF_VECTORIZER,
            "sklearn_config": {
                "stop_words": "english",
                "min_df": 2,
                "max_features": 3000,
            },
        }
    ],
}

al_il_config_lr = {
    "paradigm": Cat.AL.Paradigm.CUSTOM,
    "method": Cat.AL.CustomMethods.BINARYTAR,
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.LOGISTIC,
        "model_configuration": {
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 10000,
        },
        "task": Cat.ML.Task.BINARY_TAR,
        "feature_extraction": tf_idf_autotar,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
    "batch_size": 10,
}

al_config_lr = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.LABELMAXIMIZER,
    "label": "Relevant",
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.LOGISTIC,
        "model_configuration": {
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 10000,
        },
        "task": Cat.ML.Task.BINARY,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
}
DOUBLEBALANCER = {"type": Cat.BL.Type.DOUBLE, "config": {}}
IDENTITYBALANCER = {"type": Cat.BL.Type.IDENTITY, "config": {}}


def btar(machinelearning: Mapping[str, Any], batch_size: int = 10):
    config = {
        "paradigm": Cat.AL.Paradigm.CUSTOM,
        "method": Cat.AL.CustomMethods.BINARYTAR,
        "machinelearning": machinelearning,
        "batch_size": batch_size,
    }
    return config


def autotar(
    machinelearning: Mapping[str, Any], k_sample: int = 100, batch_size: int = 20
):
    config = {
        "paradigm": Cat.AL.Paradigm.CUSTOM,
        "method": Cat.AL.CustomMethods.AUTOTAR,
        "machinelearning": machinelearning,
        "k_sample": 100,
        "batch_size": 20,
    }
    return config


def tar_classifier(
    sklearn_model: Cat.ML.SklearnModel,
    model_configuration: Mapping[str, Any],
    feature_extraction: Mapping[str, Any],
    balancer: Mapping[str, Any] = IDENTITYBALANCER,
) -> Mapping[str, Any]:
    config = {
        "sklearn_model": sklearn_model,
        "model_configuration": model_configuration,
        "task": Cat.ML.Task.BINARY_TAR,
        "feature_extraction": feature_extraction,
        "balancer": balancer,
    }
    return config


al_il_config_nb = btar(
    machinelearning=tar_classifier(
        sklearn_model=Cat.ML.SklearnModel.NAIVE_BAYES,
        model_configuration={"alpha": 3.822},
        feature_extraction=tf_idf_autotar,
        balancer=DOUBLEBALANCER,
    ),
    batch_size=10,
)
al_il_config_lr = btar(
    machinelearning=tar_classifier(
        sklearn_model=Cat.ML.SklearnModel.LOGISTIC,
        model_configuration=dict(),
        feature_extraction=tf_idf_autotar,
        balancer=DOUBLEBALANCER,
    ),
    batch_size=10,
)
al_il_config_lgbm = btar(
    machinelearning=tar_classifier(
        sklearn_model=Cat.ML.SklearnModel.LGBM,
        model_configuration=dict(),
        feature_extraction=tf_idf_autotar,
        balancer=DOUBLEBALANCER,
    ),
    batch_size=10,
)
al_il_config_rf = btar(
    machinelearning=tar_classifier(
        sklearn_model=Cat.ML.SklearnModel.RANDOM_FOREST,
        model_configuration={
            "n_estimators": 100,
            "max_features": 10,
        },
        feature_extraction=tf_idf_autotar,
        balancer=DOUBLEBALANCER,
    ),
    batch_size=10,
)
al_config_nb = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.LABELMAXIMIZER,
    "label": "Relevant",
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
        "model_configuration": {
            "alpha": 3.822,
        },
        "task": Cat.ML.Task.BINARY,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
}
al_config_lgbm = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.LABELMAXIMIZER,
    "label": "Relevant",
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.LGBM,
        "model_configuration": {},
        "task": Cat.ML.Task.BINARY,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
}
al_config_svm_random = {
    "paradigm": Cat.AL.Paradigm.PROBABILITY_BASED_ENSEMBLE,
    "strategies": [
        {"query_type": Cat.AL.QueryType.MAX_ENTROPY},
        {"query_type": Cat.AL.QueryType.MOST_CONFIDENCE},
    ],
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear",
            "probability": True,
            "class_weight": "balanced",
        },
        "task": Cat.ML.Task.MULTILABEL,
        "balancer": {"type": Cat.BL.Type.IDENTITY, "config": {}},
        "mc_method": Cat.ML.MulticlassMethod.ONE_VS_REST,
    },
}

al_config_ensemble_prob = {
    "paradigm": Cat.AL.Paradigm.PROBABILITY_BASED_ENSEMBLE,
    "strategies": [
        {"query_type": Cat.AL.QueryType.MAX_ENTROPY},
        {"query_type": Cat.AL.QueryType.MOST_CONFIDENCE},
    ],
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear",
            "probability": True,
            "class_weight": "balanced",
        },
        "task": Cat.ML.Task.MULTILABEL,
        "balancer": {"type": Cat.BL.Type.IDENTITY, "config": {}},
        "mc_method": Cat.ML.MulticlassMethod.ONE_VS_REST,
    },
}

al_config_ensemble_labelprob = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED_ENSEMBLE,
    "strategy": Cat.AL.QueryType.LABELUNCERTAINTY_NEW,
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear",
            "probability": True,
            "class_weight": "balanced",
        },
        "task": Cat.ML.Task.MULTILABEL,
        "balancer": {"type": Cat.BL.Type.IDENTITY, "config": {}},
        "mc_method": Cat.ML.MulticlassMethod.ONE_VS_REST,
    },
}
al_config_ensemble_random = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED_ENSEMBLE,
    "strategy": Cat.AL.QueryType.RANDOM_ML,
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear",
            "probability": True,
            "class_weight": "balanced",
        },
        "task": Cat.ML.Task.MULTILABEL,
        "balancer": {"type": Cat.BL.Type.IDENTITY, "config": {}},
        "mc_method": Cat.ML.MulticlassMethod.ONE_VS_REST,
    },
}


al_config_rf = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.LABELMAXIMIZER,
    "label": "Relevant",
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.RANDOM_FOREST,
        "model_configuration": {
            "n_estimators": 100,
            "max_features": 10,
        },
        "task": Cat.ML.Task.BINARY,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
}
al_config_unc = {
    "paradigm": Cat.AL.Paradigm.LABEL_PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.LABELUNCERTAINTY,
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
        "model_configuration": {
            "alpha": 3.822,
        },
        "task": Cat.ML.Task.BINARY,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
}
al_config_svm_multilabel = {
    "paradigm": Cat.AL.Paradigm.PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.MARGIN_SAMPLING,
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear",
            "probability": True,
            "class_weight": "balanced",
        },
        "task": Cat.ML.Task.MULTILABEL,
        "balancer": {"type": Cat.BL.Type.IDENTITY, "config": {}},
        "mc_method": Cat.ML.MulticlassMethod.ONE_VS_REST,
    },
}

al_config_random = {
    "paradigm": Cat.AL.Paradigm.POOLBASED,
    "query_type": Cat.AL.QueryType.RANDOM_SAMPLING,
}
mixed_estimator = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        al_config_nb,
        al_config_svm,
        al_config_rf,
        al_config_lgbm,
    ],
}

naive_bayes_estimator = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes1"),
        add_identifier(al_config_nb, "NaiveBayes2"),
        add_identifier(al_config_nb, "NaiveBayes3"),
        add_identifier(al_config_nb, "NaiveBayes4"),
    ],
}
svm_estimator = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_svm, "SVM1"),
        add_identifier(al_config_svm, "SVM2"),
        add_identifier(al_config_svm, "SVM3"),
        add_identifier(al_config_svm, "SVM4"),
    ],
}

rasch_estimator = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes"),
        add_identifier(al_config_svm, "SVM"),
        add_identifier(al_config_rf, "RandomForest"),
    ],
}

rasch_nblrrf = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_lr, "LogisticRegression"),
    ],
}
rasch_nblrrfsvm = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_lr, "LogisticRegression"),
        add_identifier(al_config_svm, "SVM"),
    ],
}
rasch_nblrrflgbm = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_lr, "LogisticRegression"),
        add_identifier(al_config_lgbm, "LGBM"),
    ],
}
rasch_nblrrflgbmrand = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_lr, "LogisticRegression"),
        add_identifier(al_config_lgbm, "LGBM"),
        add_identifier(al_config_random, "Random"),
    ],
}

ilrasch_nblrrflgbmrand = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        set_batch_size(add_identifier(al_il_config_nb, "NaiveBayes"), 200),
        set_batch_size(add_identifier(al_il_config_rf, "RandomForest"), 200),
        set_batch_size(add_identifier(al_il_config_lr, "LogisticRegression"), 200),
        set_batch_size(add_identifier(al_il_config_lgbm, "LGBM"), 200),
        set_batch_size(add_identifier(al_config_random, "Random"), 200),
    ],
}

rasch_lr = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes"),
        add_identifier(al_config_svm, "SVM"),
        add_identifier(al_config_lr, "LogisticRegression"),
    ],
}
rasch_rf = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_rf, "RandomForest1"),
        add_identifier(al_config_rf, "RandomForest2"),
        add_identifier(al_config_rf, "RandomForest3"),
    ],
}

rasch_random_estimator = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes"),
        add_identifier(al_config_svm, "SVM"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_random, "Random3"),
    ],
}

# rasch_random_estimator = {
#     "paradigm": Cat.AL.Paradigm.ESTIMATOR,
#     "learners": [
#         add_identifier(al_config_random, "Random1"),
#         add_identifier(al_config_random, "Random2"),
#         add_identifier(al_config_random, "Random3"),
#     ]
# }


al_config_est3 = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes1"),
        add_identifier(al_config_svm, "SVM"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_nb, "NaiveBayes4"),
    ],
}
al_config_est4 = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes1"),
        add_identifier(al_config_svm, "SVM"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_lr, "LogisticRegression"),
    ],
}
al_config_est5 = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(al_config_nb, "NaiveBayes1"),
        add_identifier(al_config_svm, "SVM"),
        add_identifier(al_config_rf, "RandomForest"),
        add_identifier(al_config_svm, "SVM2"),
        add_identifier(al_config_svm, "SVM3"),
    ],
}
autotars = {
    "NaiveBayes": (Cat.ML.SklearnModel.NAIVE_BAYES, {"alpha": 3.822}, 100, 20),
    "Logistic": (
        Cat.ML.SklearnModel.LOGISTIC,
        {
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 10000,
        },
        100,
        20,
    ),
    "RF": (
        Cat.ML.SklearnModel.RANDOM_FOREST,
        {
            "n_estimators": 100,
            "max_features": 10,
        },
        100,
        20,
    ),
    "LGBM": (Cat.ML.SklearnModel.LGBM, dict(), 100, 20),
}
autotar_ensemble = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        add_identifier(
            autotar(
                tar_classifier(skl, skl_config, tf_idf_autotar, IDENTITYBALANCER), k, b
            ),
            name,
        )
        for name, (skl, skl_config, k, b) in autotars.items()
    ]
    + [add_identifier(al_config_random, "Random")],
}

al_config_ens = {
    "paradigm": Cat.AL.Paradigm.ENSEMBLE,
    "learners": [
        al_config_nb,
        al_config_svm,
        al_config_rf,
        al_config_unc,
    ],
    "probabilities": [0.35, 0.35, 0.2, 0.1],
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
        "model_configuration": {},
        "task": Cat.ML.Task.BINARY,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
    },
}

al_config_entropy = {
    "paradigm": Cat.AL.Paradigm.PROBABILITY_BASED,
    "query_type": Cat.AL.QueryType.MOST_CONFIDENCE,
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.SVC,
        "model_configuration": {
            "kernel": "linear",
            "probability": True,
            "class_weight": "balanced",
        },
        "task": Cat.ML.Task.MULTILABEL,
        "balancer": {"type": Cat.BL.Type.DOUBLE, "config": {}},
        "mc_method": Cat.ML.MulticlassMethod.ONE_VS_REST,
    },
}


env_config = {"environment_type": Cat.ENV.Type.MEMORY}
