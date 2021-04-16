# Smells

## Install
run `make` to install.
`
## Configuration
Obtain the data.zip from https://zenodo.org/record/4697491, extract and place the data directory in the root directory.<br>
It includes the data sets necessary for the execution.<br>
The directory should follow the structure.
>~
>> data (dir)
>>> datasets.csv<br>
>>> metrics_datasets.csv<br>
>>> smellsmetrics_datasets.csv

## Run
To run everything

```python crossproject.py```

To run specific Classifier, Project or Approach, follow the help instructions:

```python crossproject.py help```