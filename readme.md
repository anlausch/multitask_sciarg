# Multi-task Learning for Scitorics
This repository contains the code of our single-task and multi-task learning models for the rhetorical analysis of scientific publications.

## Abstract
Exponential growth in the number of scientific publications yields the need for effective automatic analysis of rhetorical aspects of scientific writing. Acknowledging the argumentative nature of scientific text, in this work we investigate the link between the argumentative structure of scientific publications and rhetorical aspects such as discourse categories or citation contexts. To this end, we (1) augment a corpus of scientific publications annotated with four layers of rhetoric annotations with argumentation annotations and (2) investigate neural multi-task learning architectures combining argument extraction with a set of rhetorical classification tasks. By coupling rhetorical classifiers with the extraction of argumentative components in a joint multi-task learning setting, we obtain significant performance gains for different rhetorical analysis tasks.

## Repository Description
- ./baslines/: baseline code (HMM, CRF, SVM)
- ./models/: bi-LSTM models for ST and MT Learning with homoscedastic uncertainty-based task weighting
- ./evaluation/: confusion matrix code
- ./load_conll.py: loader for the data which can be found at http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip
- ./utils.py: util functions

## Data and Annotation Guidelines
- Data: http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip
- Annotation Guidelines: http://data.dws.informatik.uni-mannheim.de/sci-arg/annotation_guidelines.pdf
## Citation
```
@inproceedings{lauscher-etal-2018-investigating,
    title = "Investigating the Role of Argumentation in the Rhetorical Analysis of Scientific Publications with Neural Multi-Task Learning Models",
    author = "Lauscher, Anne  and
      Glava{\v{s}}, Goran  and
      Ponzetto, Simone Paolo  and
      Eckert, Kai",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1370",
    doi = "10.18653/v1/D18-1370",
    pages = "3326--3338"
}
```
