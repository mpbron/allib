from ..activelearning import ActiveLearningFactory
from ..component import Component
from ..environment import EnvironmentFactory
from ..factory import ObjectFactory
from ..feature_extraction import FeatureExtractionFactory
from .catalog import ModuleCatalog as Cat

CONFIG = {
    Component.ACTIVELEARNER: {
        "al_paradigm": Cat.AL.Paradigm.POOLBASED,
        "query_type": Cat.AL.QueryType.INTERLEAVE,
        "machinelearning": {
            "sklearn_model": Cat.ML.SklearnModel.RANDOM_FOREST,
            "model_configuration": {},
            "ml_task": Cat.ML.Task.MULTILABEL,
            "mc_method": Cat.ML.MulticlassMethod.ONE_VS_REST,
            "min_train_annotations": 30,
            Component.BALANCER: {
                "balancer_type": Cat.BL.Type.IDENTITY,
                "balancer_config": {}
            }
        }
    },
    Component.ENVIRONMENT: {
        "environment_type": Cat.ENV.Type.MEMORY,

    },
    Component.FEATURE_EXTRACTION:{
        "datatype": Cat.FE.DataType.TEXTINSTANCE,
        "vec_type": Cat.FE.VectorizerType.STACK,
        "vectorizers": [
            {
                "vec_type": Cat.FE.VectorizerType.SKLEARN,
                "sklearn_vec_type": Cat.FE.SklearnVecType.TFIDF_VECTORIZER,
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
                "vec_type": Cat.FE.VectorizerType.DOC2VEC,
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
        self.attach(FeatureExtractionFactory())
        self.attach(ActiveLearningFactory())
        self.attach(EnvironmentFactory())
