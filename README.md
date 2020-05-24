# dutch-simpletransformers

Use simpletransformers for Dutch language spell checking.

Ref.
* https://simpletransformers.ai/
* https://github.com/ThilinaRajapakse/simpletransformers#minimal-start-for-binary-classification

## Basic experiments

Try the binary classification with `BertForSequenceClassification`  from simpletransformers.

First with the exact English language example.

Next with a Dutch language version, using the `bert-base-dutch-cased` Huggingface pretrained model
from [BERTje](https://github.com/wietsedv/bertje).

=> Initial results as comments in [binary_classification_BERTje.py](./src/binary_classification_BERTje.py)
