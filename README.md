# second-year-project

## Setup
Install all the needed package dependencies using the command:
```
pip install -r requirements.txt
```

### Run Experiments
To run the experiments use the `run.sh` script, which has the following optional arguments:

```bash
"-e <path>   Path to the experiments folder. Default: experiments"
"-d <path>   Path to the data folder. Default: crossre_data"
"-t <type>   Type of experiment. Default: baseline (options: special-token, dataset-embeddings, baseline)"
"-m <type>   Type of entity markers. Default: all (options: all, generic, none)"
```

To reproduce the baseline using the command:
```bash
./run.sh -m "generic"
```

To run e.g., the dataset embeddings model use the command:

```bash
./run.sh -t "dataset-embeddings" -e "dataset-emb-experiment" -m "generic"
```
