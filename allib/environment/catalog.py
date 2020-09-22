from enum import Enum

class EnvironmentType(Enum):
    MEMORY = "Memory"
    DJANGO = "Django"
    ELASTIC = "Elastic"
    ELASTIC_CHAT = "ElasticChat"
    ELASTIC_MAIL = "ElasticMail"
