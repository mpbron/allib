# Active Learning and Technology-Assisted Review library for Python

[![PyPI](https://img.shields.io/pypi/v/python-allib.png)](https://pypi.org/project/python-allib/)
[![Python_version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/instancelib/)
[![License](https://img.shields.io/pypi/l/python-allib.png)](https://www.gnu.org/licenses/lgpl-3.0.en.html)[![DOI](https://zenodo.org/badge/414780120.svg)](https://zenodo.org/doi/10.5281/zenodo.10869682)

------------------------------------------------------------------------

`python-allib` is a library that enables efficient data annotation with
Active Learning on various types of datasets. Through the
library[`instancelib`](https://github.com/mpbron/instancelib) we support
various **machine learning algorithms** and **instance types**. Besides
canonical Active Learning, this library offers Technology-Assisted
Review methods, which aid in making High-Recall Information Retrieval
tasks more efficient.

© Michiel Bron, 2024

## Quick tour of Technology-Assisted Review simulation

### Load dataset

Load the dataset in an `instancelib` environment.

``` python
# Some imports
from pathlib import Path
from allib.benchmarking.datasets import TarDataset, DatasetType

POS = "Relevant"
NEG = "Irrelevant"
# Load a dataset in SYNERGY/ASREVIEW format
dataset_description = TarDataset(
  DatasetType.REVIEW, 
  Path("./allib/tests/testdataset.csv"))

# Get an instancelib Environment object
ds = dataset_description.env

ds
```

    Environment(dataset=InstanceProvider(length=2019), 
       labels=LabelProvider(labelset=frozenset({'Relevant', 'Irrelevant'}), 
       length=0, 
       statistics={'Relevant': 0, 'Irrelevant': 0}), 
       named_providers={}, 
       length=2019, 
       typeinfo=TypeInfo(identifier=int, data=str, vector=NoneType, representation=str)) 

The `ds` object is currently loaded in TAR simulation mode. This means,
that like the at the start of review process, there is no labeled data.
This is visible in the statistics in the `ds` objects. However, as this
is simulation mode, there is a ground truth available. This can be
accessed as follows:

``` python
ds.truth
```

    LabelProvider(labelset=frozenset({'Relevant', 'Irrelevant'}), 
       length=2019, 
       statistics={'Relevant': 101, 'Irrelevant': 1918})

In Active Learning, we are dealing with a partially labeled dataset.
There are two `InstanceProvider` objects inside the `ds` object that
maintain the label status:

``` python
print(f"Unlabeled: {ds.unlabeled}, Labeled: {ds.labeled}")
```

    Unlabeled: InstanceProvider(length=2019), Labeled: InstanceProvider(length=0)

### Basic operations

The `ds` object supports all `instancelib` operations, for example,
dividing the dataset in a train and test set.

``` python
train, test = ds.train_test_split(ds.dataset, train_size=0.70)
print(f"Train: {train}, Test: {test}")
```

    Train: InstanceProvider(length=1413), Test: InstanceProvider(length=606)

### Train a ML model

We can also train Machine Learning methods on the ground truth data in
`ds.truth`.

``` python
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from instancelib.analysis.base import prediction_viewer
import instancelib as il
pipeline = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', LogisticRegression()),
     ])

model = il.SkLearnDataClassifier.build(pipeline, ds)
model.fit_provider(train, ds.truth)
```

With the method `prediction_viewer` we can view the predictions as a
Pandas dataframe.

``` python
# Show the three instances with the highest probability to be Relevant
df = prediction_viewer(model, test, ds.truth).sort_values(
    by="p_Relevant", ascending=False
)
df.head(3)
```

      0%|          | 0/4 [00:00<?, ?it/s]

      0%|          | 0/4 [00:00<?, ?it/s]

<div>

<div>

|      | data                                              | label    | prediction | p_Irrelevant | p_Relevant |
|------|---------------------------------------------------|----------|------------|--------------|------------|
| 175  | A randomized trial of a computer-based interve... | Relevant | Irrelevant | 0.639552     | 0.360448   |
| 1797 | Computerized decision support to reduce potent... | Relevant | Irrelevant | 0.737130     | 0.262870   |
| 956  | Improvement of intraoperative antibiotic proph... | Relevant | Irrelevant | 0.762803     | 0.237197   |

</div>

</div>

Although the predicition probabilities are below 0.50, some of the top
ranked documents have a ground truth label relevant.

### Active Learning

We can integrate the model in an Active Learning method. A simple TAR
method is AutoTAR.

``` python
from allib.activelearning.autotar import AutoTarLearner

al = AutoTarLearner(ds, model, POS, NEG, k_sample=100, batch_size=20)
```

To kick-off the process, we need some labeled data. Let’s give it some
training data.

``` python
pos_instance = al.env.dataset[28]
neg_instance = al.env.dataset[30]
al.env.labels.set_labels(pos_instance, POS)
al.env.labels.set_labels(neg_instance, NEG)
al.set_as_labeled(pos_instance)
al.set_as_labeled(neg_instance)
```

Next, we can retrieve the instance that should be labeled next with the
following command.

``` python
next_instance = next(al)
# next_instance is an Instance object.
# Representation contains a human-readable string version of the instance
print(
    f"{next_instance.representation[:60]}...\n"
    f"Ground Truth Labels: {al.env.truth[next_instance]}"
)
```

      0%|          | 0/11 [00:00<?, ?it/s]

    Oral quinolones in hospitalized patients: an evaluation of a...
    Ground Truth Labels: frozenset({'Relevant'})

### Simulation

Using the ground truth data, we can further simulate the TAR process in
an automated fashion:

``` python
from allib.stopcriterion.heuristic import AprioriRecallTarget
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.simulation import TarSimulator

recall95 = AprioriRecallTarget(POS, 0.95)
recall100 = AprioriRecallTarget(POS, 1.0)
criteria = {
    "Perfect95": recall95,
    "Perfect100": recall100,
}

# Then we can specify our experiment
exp = ExperimentIterator(al, POS, NEG, criteria, {})
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter)
```

``` python
simulator.simulate()
plotter.show()
```

      0%|          | 0/2019 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/9 [00:00<?, ?it/s]

      0%|          | 0/9 [00:00<?, ?it/s]

      0%|          | 0/9 [00:00<?, ?it/s]

      0%|          | 0/9 [00:00<?, ?it/s]

      0%|          | 0/8 [00:00<?, ?it/s]

      0%|          | 0/8 [00:00<?, ?it/s]

      0%|          | 0/7 [00:00<?, ?it/s]

      0%|          | 0/7 [00:00<?, ?it/s]

      0%|          | 0/7 [00:00<?, ?it/s]

      0%|          | 0/6 [00:00<?, ?it/s]

      0%|          | 0/5 [00:00<?, ?it/s]

      0%|          | 0/5 [00:00<?, ?it/s]

      0%|          | 0/4 [00:00<?, ?it/s]

      0%|          | 0/3 [00:00<?, ?it/s]

      0%|          | 0/2 [00:00<?, ?it/s]

      0%|          | 0/1 [00:00<?, ?it/s]

![](README_files/figure-commonmark/cell-12-output-25.png)

## Command Line Interface

Besides importing the library, the code can be used to run some
predefined experiments.

For a CSV in SYNERGY format:

``` console
python -m allib benchmark -m Review -d  ./path/to/dataset -t ./path/to/results/ -e AUTOTAR -r 42
```

For a dataset in TREC-style:

``` console
python -m allib benchmark -m Trec -d  ./path/to/dataset/ -t ./path/to/results/ -e AUTOTAR -r 42
```

Experiment options are:

- `AUTOTAR`
- `AUTOSTOP`
- `CHAO`
- `TARGET`
- `CMH`

The `-r` option is used to supply a seed value that is given to a random
generator.

## Installation

See [installation.md](installation.md) for an extended installation
guide, especially for enabling the `CHAO` method. Short instructions are
below.

| Method | Instructions                                                                                       |
|--------|----------------------------------------------------------------------------------------------------|
| `pip`  | Install from [PyPI](https://pypi.org/project/python-allib/) via `pip install python-allib`.        |
| Local  | Clone this repository and install via `pip install -e .` or locally run `python setup.py install`. |

## Releases

`python-allib` is officially released through
[PyPI](https://pypi.org/project/instancelib/).

See [CHANGELOG](CHANGELOG) for a full overview of the changes for each
version.

## Citation

Use this bibtex to cite this package, or go to
[ZENODO](https://doi.org/10.5281/zenodo.10869682), to cite a specific
version.

``` bibtex
@software{bron_2024_108698682,
  author       = {Bron, Michiel},
  title        = {Python Package python-allib},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.5.1},
  doi          = {10.5281/zenodo.10869682},
  url          = {https://doi.org/10.5281/zenodo.10869682}
}
```

## Maintenance

### Contributors

- [Michiel Bron](https://www.uu.nl/staff/MPBron) (`@mpbron`)
