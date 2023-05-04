from enum import Enum


class Experiment(str, Enum):
    baseline = "baseline"
    special_token = "special-token"
    dataset_embeddings = "dataset-embeddings"


class Markers(str, Enum):
    all = "all"
    generic = "generic"
    none = "none"
