import pandas as pd
from deepsig import aso


def significance_test(
    model_scores: dict[str, list[float]], seed: int = 1234
) -> pd.DataFrame:
    return aso(my_models_scores, confidence_level=0.95, return_df=True, seed=seed)


# (i.e. person, location, organisation, misc, event)


entity_to_generic = {
    "academicjournal": "misc",
    "album": "misc",
    "algorithm": "misc",
    "astronomicalobject": "misc",
    "award": "misc",
    "band": "organisation",
    "book": "misc",
    "chemicalcompound": "misc",
    "chemicalelement": "misc",
    "conference": "event",
    "country": "location",
    "discipline": "misc",
    "election": "event",
    "enzyme": "misc",
    "event": "event",
    "field": "misc",
    "literarygenre": "misc",
    "location": "location",
    "magazine": "misc",
    "metrics": "misc",
    "misc": "misc",
    "musicgenre": "misc",
    "musicalartist": "person",
    "musicalinstrument": "misc",
    "organisation": "organisation",
    "person": "person",
    "poem": "misc",
    "politicalparty": "organisation",
    "politician": "person",
    "product": "misc",
    "programlang": "misc",
    "protein": "misc",
    "researcher": "person",
    "scientist": "person",
    "song": "misc",
    "task": "misc",
    "theory": "misc",
    "university": "organisation",
    "writer": "person",
}
