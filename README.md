# dutch-simpletransformers

Use simpletransformers for Dutch language spell checking (starting with "dt"-mistakes).

Ref.
* https://simpletransformers.ai/
* https://github.com/ThilinaRajapakse/simpletransformers#minimal-start-for-binary-classification

## Basic experiments

Try the binary classification with `BertForSequenceClassification` from simpletransformers.

First with the exact English language example.

Next with a Dutch language version, using the `bert-base-dutch-cased` Huggingface pretrained model
from [BERTje](https://github.com/wietsedv/bertje).

=> Initial results as comments in [dt_classification_BERTje.py](./src/dt_classification_BERTje.py)

## Add more training and validation data

### Worden

This is *only* for the detection of "ik word, word ik, jij wordt, word jij, (hij/zij) wordt, wordt (hij/zij)".

With trainging data of approx.

* 100 synthetic cases
* 140 cases from nl.wikipedia (manually verified and corrected)

and validation data of approx.

* 22 synthetic cases
* 30 cases from nl.wikipedia (manually verified and corrected)

We seem to reach an accuracy ofi 100.0% (only 1 false positive) on the 46 validation cases:
{'mcc': 1.0, 'tp': 26, 'tn': 26, 'fp': 0, 'fn': 0, 'eval_loss': 7.357895550999924e-05}

The convergence of the loss was clearly visible from the 5th epoch. Not sure if the accuracy would
be better on this small training set for more then 10 epoch's or maybe we would be overfitting ?

I presume a large part of the accuracy is due to the use of the existing [BERTje](https://github.com/wietsedv/bertje) model.
We are only training 1 final classification layer here.

### Zenden

Testing for "zenden" (zend vs. zendt) without dedicated training examples for zenden
resulted in seemingly random results.

Adding a large number of synthetic and examples from nl.wikipedia got a better accuracy,
approx. 90% but still unable to reach "exact" 100% accuracy (even if all cases are 100%
clear for a human reader). Added some synthetic cases specifically targeted at the failing
cases for zenden (mainly imperative usage in bible texts). With these synthetic examples,
the accuracy for "zenden" got better (but not 100%), but some of the validations for
"worden" that where very stable, started to become unstable again ...

The best accuracy up to now, with "worden" (word vs. wordt) and "zenden" (zend vs. zendt)
combined is:

`# {'mcc': 0.9271050693011065, 'tp': 39, 'tn': 40, 'fp': 1, 'fn': 2, 'eval_loss': 0.18659917498536577}`

The training was reaching a loss of like 0.001 after 10 or 15 epochs. But this was not
resulting in a similar drop in the eval_loss. The reason for this is not clear.

## Trying RoBERTa (starting from RobBERT)

Since I failed to reach 100% validation accuracy with the BERT based model for `worden` en
`zenden`, I also tried to use the exact same training and validation set on a RoBERTa based
model, starting from the Dutch language `RobBERT` model.

Very interesting findings:
* the first try converged (locally saved as `outputs/RoBERTa-001`)
* the version of the model yields validation accuracy of 100% see [results](https://gitlab.com/spelfouten/dutch-simpletransformers/-/blob/8e7b92b782dafc63f730ac2b756602404c7c5e47/src/dt_classification_RobBERT.py#L18-105)
* and very interestingly, this model gets a 95% accuracy on a short test set evaluating
  _non-trained_ verbs (`vinden` and `lopen`); See bottom cell of this [notebook](https://gitlab.com/spelfouten/dutch-simpletransformers/-/blob/8e7b92b782dafc63f730ac2b756602404c7c5e47/experiments.ipynb)
* the second try to train the model did _not_ converge in 15 epochs
  (I never saw this with BERT, the training always converged)
* the third try is converging, validation results are TBD
