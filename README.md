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

## Add more training and validation data

This is *only* for the detection of "ik word, word ik, jij wordt, word jij, (hij/zij) wordt, wordt (hij/zij)".

With trainging data of approx.

* 64 synthetic cases
* 140 cases from nl.wikipedia (manually verified and corrected)

and validation data of approx.

* 16 synthetic cases
* 30 cases from nl.wikipedia (manually verified and corrected)

We seem to reach an accuracy of 97.8% (only 1 false positive) on the 46 validation cases:

`{'mcc': 0.957427107756338, 'tp': 23, 'tn': 22, 'fp': 1, 'fn': 0, 'eval_loss': 0.02614415737237626}`

The convergence of the loss was clearly visible from the 5th epoch. Not sure if the accuracy would
be better on this small training set for more then 10 epoch's or maybe we would be overfitting ?

I presume a large part of the accuracy is due to the use of the existing [BERTje](https://github.com/wietsedv/bertje) model.
We are only training 1 final classification layer here.
