import activelearning as al
import balancing as bl
import environment as env
import feature_extraction as fe
import machinelearning as ml

from .component import Component
from factory.factory import ObjectFactory

CONFIG = {
    Component.ACTIVELEARNER: {
        "al_paradigm": al.catalog.ALParadigm.POOLBASED,
        "query_type": al.catalog.QueryType.INTERLEAVE,
        "machinelearning": {
            "sklearn_model": ml.catalog.SklearnModel.RANDOM_FOREST,
            "model_configuration": {},
            "ml_task": ml.catalog.MachineLearningTask.MULTILABEL,
            "mc_method": ml.catalog.MulticlassMethod.ONE_VS_REST,
            "min_train_annotations": 30,
            Component.BALANCER: {
                "balancer_type"
            }
        }
    },
    Component.ENVIRONMENT: {
        "environment_type": env.catalog.EnvironmentType.MEMORY,

    },
    Component.FEATURE_EXTRACTION:{
        "datatype": fe.catalog.DataType.TEXTINSTANCE,
        "vec_type": fe.catalog.VectorizerType.STACK,
        "vectorizers": [
            {
                "vec_type": fe.catalog.VectorizerType.SKLEARN,
                "sklearn_vec_type": fe.catalog.SklearnVecType.TFIDF_VECTORIZER,
                "sklearn_config": {
                    "min_df": 1,
                    "max_df": 0.8,
                    "max_features": 2000,
                    "analyzer": "char_wb",
                    "ngram_range": (3, 4),
                    "sublinear_tf": True
                }
            },
            {
                "vec_type": fe.catalog.VectorizerType.DOC2VEC,
                "d2v_params": {
                    "epochs": 5,
                    "vector_size": 300,
                    "workers": 2,
                }
            }
        ]
    }
}
class MainFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(fe.FeatureExtractionFactory())
        self.attach(al.ActiveLearningFactory())
        self.attach(env.EnvironmentFactory())
