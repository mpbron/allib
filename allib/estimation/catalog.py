from enum import Enum

class EstimatorCatalog(Enum):
    HorvitzThompsonLoose = "HorvitzThompsonLoose"
    HorvitzThompson1 = "HorvitzThompson1"
    HorvitzThompson2 = "HorvitzThompson2"
    CHAO = "Chao"