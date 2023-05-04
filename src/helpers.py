import pandas as pd
from deepsig import aso


def significance_test(
    model_scores: dict[str, list[float]], seed: int = 1234
) -> pd.DataFrame:
    return aso(my_models_scores, confidence_level=0.95, return_df=True, seed=seed)
