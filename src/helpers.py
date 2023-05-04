import pandas as pd
from deepsig import aso


def significance_test(
    model_scores: dict[str, list[float]], seed: int = 1234
) -> pd.DataFrame:
    return aso(my_models_scores, confidence_level=0.95, return_df=True, seed=seed)


# (i.e. person, location, organization, miscellaneous, event)


entity_to_generic = {
    "academicjournal": "miscellaneous",
    "album": "miscellaneous",
    "algorithm": "miscellaneous",
    "astronomicalobject": "miscellaneous",
    "award": "miscellaneous",
    "band": "organization",
    "book": "miscellaneous",
    "chemicalcompound": "miscellaneous",
    "chemicalelement": "miscellaneous",
    "conference": "event",
    "country": "location",
    "discipline": "miscellaneous",
    "election": "event",
    "enzyme": "miscellaneous",
    "event": "event",
    "field": "miscellaneous",
    "literarygenre": "miscellaneous",
    "location": "location",
    "magazine": "miscellaneous",
    "metrics": "miscellaneous",
    "misc": "miscellaneous",
    "miscellaneous": "miscellaneous",
    "musicgenre": "miscellaneous",
    "musicalartist": "person",
    "musicalinstrument": "miscellaneous",
    "organization": "organization",
    "organisation": "organization",
    "person": "person",
    "poem": "miscellaneous",
    "politicalparty": "organization",
    "politician": "person",
    "product": "miscellaneous",
    "programlang": "miscellaneous",
    "protein": "miscellaneous",
    "researcher": "person",
    "scientist": "person",
    "song": "miscellaneous",
    "task": "miscellaneous",
    "theory": "miscellaneous",
    "university": "organization",
    "writer": "person",
}
