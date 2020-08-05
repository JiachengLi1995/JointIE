# JointIE

This repository contains code used for span-based joint NER and relation extraction.

I refer to the following models and repos:

Generalizing Natural Language Analysis through Span-relation Representations (ACL2020). [paper](https://arxiv.org/abs/1911.03822) [github](https://github.com/neulab/cmu-multinlp)

Entity, Relation, and Event Extraction with Contextualized Span Representations (ACL 2019). [paper](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8) [github](https://github.com/dwadden/dygiepp)

AllenNLP. [github](https://github.com/allenai/allennlp)

## Recommended Environment

torch==1.5.1

transformers==2.11.0

## Datasets

5 datasets are preprocessed and included in this repository.

| Dataset             | Task code     | Dir                      | Source   |
|---------------------|---------------|--------------------------|----------|
| Wet Lab Protocols   | wlp           | data/wlp                 | [link](https://github.com/chaitanya2334/WLP-Dataset)|
| SciERC              | scierc        | data/scierc              | [link](http://nlp.cs.washington.edu/sciIE/)|
| NYT24               | nyt24         | data/nyt24               | [link](https://drive.google.com/drive/folders/1RPD9kuHUHp4O3gQLLD1CgDPigAlRiR7L)|
| NYT29               | nyt29         | data/nyt29               | [link](https://drive.google.com/drive/folders/1RPD9kuHUHp4O3gQLLD1CgDPigAlRiR7L)|
| WebNLG              | webnlg        | data/webnlg              | [link](https://drive.google.com/file/d/1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj/view)|


## Train and Evaluate Models

Train and evaluate model with default configure.
```bash
python train_demo.py --dataset scierc
```

## Results with Default Configure on Test Set

| Dataset             | NER (F1)     | Relation (F1) |
|---------------------|--------------|---------------|
| Wet Lab Protocols   | 81.5         | 66.26         |
| SciERC              | 67.27        | 41.77         |
| NYT24               | 94.4         | 77.48         |
| NYT29               | 89.19        | 62.53         |
| WebNLG              | 94.45        | 77.5          |

## Contact

E-mail: j9li@eng.ucsd.edu