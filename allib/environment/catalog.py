from aenum import Enum


class EnvironmentCatalog:
    class Type(Enum):
        MEMORY = "Memory"
        PANDAS = "Pandas"
