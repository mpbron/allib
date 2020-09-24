from enum import Enum


class EnvironmentCatalog:
    class Type(Enum):
        MEMORY = "Memory"
        PANDAS = "Pandas"
