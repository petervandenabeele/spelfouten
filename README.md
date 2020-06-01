# Spelfouten: Dutch spell checking with BERT based models ("dt"-fouten)

## Abstract

As part of my #CoronaSabbatical in May 2020, I experimented with BERT-based
Dutch language models (`RobBERT`, based on `RoBERTa` and `BERTje`, based on `BERT`),
to learn how good I could train the models to predict "dt-fouten" and other
spelling mistakes in Dutch language.

My best result was **{'mcc': 0.899370194601987, 'tp': 83, 'tn': 9998, 'fp': 2, 'fn': 17, 'eval_loss': 0.01631144658936272}**
for a validation set of 10,000 negatives (correct sentences from nl.wikipedia)
and 100 positives (sentences with "dt"-fouten on the 3 trained verbs only),
starting from the `RobBERT` model. In my approach, I got consistently beter
accuracy (both on false negatives and fals positives) with RobBERT, compared
to BERTje. I have no clear explanation why that is (I am very new at this!).

As soon as I added "dt"-fouten on untrained verbs, I got near-random results
(only 60% detected). Adding other spelling mistakes where completely _not_
detected (only 2% detected). So, I was unable to use training on a few common
example verbs ('worden', 'vinden', 'zenden') to generalize to "dt"-fouten
in other similar verbs and even more unable to generalize to generic
spelling mistakes. Of course, this can be due to my limited training sets,
and my limited experience in this matter. Proposed future work may solve this.

## Introduction

As part of my #CoronaSabbatical in May 2020, I tried to use BERT-based models to
find "dt"-mistakes and other spelling mistakes in Dutch text. In the Dutch language
there is a specifiek kind of mistake: "dt-fouten", that is hard to find and avoid.
A text with a "dt-fout" looses a lot of its credibility, just for that single reason
(and in school it would also lose most, if not all, of its score).

I used pretrained [RobBERT](https://ipieter.github.io/blog/robbert/) and
[BERTje](https://github.com/wietsedv/bertje) Dutch language models and trained a
SequenceClassification layer on top. The SequenceClassification layer had a top layer
with 2 neurons, representing the 2 cases: negative (label "0", correct spelling) and
positive (label "1", spelling mistake), per sentence.

I first experimented with direct usage of transformers, but then discovered
[simpletransformers](https://simpletransformers.ai/) that made it easier to set-up initial
experiments. Also the default hyperparameters where set-up easily with simpletransformers.

A lot of my "learning" history can be found in the history of this repo and this
README.

## Caveat

This is early work in the field of #NLP and Deep Learning. This work will probably
contain mistakes, omissions, incorrect data sets and incorrect interpretations. I
would be very glad with any feedback that can improve the quality of this work, or
at least fix errors in representation of the results and conclusions ... Feel free
to file an Issue or contact me in private for questions or fixes.

## Experimental

I used pre-trained models for RobBERT and BERTje, by using these models in simpletransformers:

* "roberta", "pdelobelle/robBERT-base", training over 25 epochs
* "bert", "bert-base-dutch-cased" , training over 25 epochs

The training data can be seen in this [code](./src/dt_classification_RoBERTa_BERT.py).
It contains a few hundred examples with labels `0` for correct conjugations of
`worden`, `zenden`, `vinden` and label `1` for incorrect conjugations of these verbs.

The validation set used immediately after the training consists of mainly synthetic examples
of 102 elements.


Later, a larger validation set was made with:

* 10,000 [correct examples](./data/correct-sentences-nlwiki-10K.txt) from nl.wikipedia
* 100 (unseen) [incorrect examples](./data/incorrect_dt_on_trained_verbs.txt) with 'dt' mistakes in the trained verbs (worden, zenden, vinden)
* 50 (unseen) [incorrect examples](./data/incorrect_dt_on_untrained_verbs.txt) with 'dt' mistakes in other verbs
* 50 (unseen) [incorrect examples](./data/incorrect_other_mistakes.txt) for other random spelling errors

The 10,000 "correct" examples where not all validated manually. There was a general manual scan and some
aspects where validated and fixed systematically, but not everything. Only when false negatives showed up,
these where validated in more detail and sometimes turned out to be "true positives". In that case, the
sentence was corrected manually by me, or sometimes removed (e.g. for a pure German or English language
sentence that is out of scope).

## Results

## Training and immediate validation

The validation set used during the training sessions, with strongly correlated examples in training and validation
(see this training [code](./src/dt_classification_RoBERTa_BERT.py)), yields these results. Details of the false
negatives are listed in the code. A few hundred training examples, 102 symmetrically balanced evaluation examples.

* based on `RobBERT` : **{'mcc': 0.9805806756909202, 'tp': 50, 'tn': 51, 'fp': 0, 'fn': 1, 'eval_loss': 0.08347923998371698}**
* based on `BERTje`  : **{'mcc': 0.9615239476408232, 'tp': 49, 'tn': 51, 'fp': 0, 'fn': 2, 'eval_loss': 0.20460643590251074}**

### Large validation set

A larger validation set of 10,000 sentences that is created from a larger data set of 14M nl.wikipedia sentences, by picking one
sentence every 1,000 entries, yields these results (ONLY negative labels, presumed correct spelling, not fully validated). This
is the [validation code](./src/dt_classification_RoBERTa_BERT_final_validation.py).

* based on `RobBERT` : **{'mcc': 0.0, 'tp': 0, 'tn': 9998, 'fp': 2, 'fn': 0, 'eval_loss': 0.0010510498600706342}**
```
INFO:simpletransformers.classification.classification_model:{'mcc': 0.0, 'tp': 0, 'tn': 9998, 'fp': 2, 'fn': 0, 'eval_loss': 0.0010510498600706342}
[[ 5.208954  -5.6316767]
 [ 4.8714986 -5.372314 ]
 [ 5.1642437 -5.710761 ]
 ...
 [ 4.9983377 -5.526781 ]
 [ 5.0196595 -5.548851 ]
 [ 4.6084604 -5.089196 ]]
Toen De Ruyter uitvoer, was Rupert eerst zeer verheugd.
0
Het is een bundel met documentaire-achtige verhalen over muzikanten en belangrijke Groningse muziekplekken zoals Vera en Het Viadukt.
0
```

* based on `BERTje`  : **{'mcc': 0.0, 'tp': 0, 'tn': 9991, 'fp': 9, 'fn': 0, 'eval_loss': 0.007517435135614505}**

```
INFO:simpletransformers.classification.classification_model:
[[ 5.524736  -5.7299075]
 [ 5.3889146 -5.4308124]
 [ 5.4801917 -5.717814 ]
 ...
 [ 4.941821  -5.09058  ]
 [ 5.34403   -5.736093 ]
 [ 5.417193  -5.765853 ]]
De Bergsche Maas doorsneed het Land van Heusden, waarvan de zuidelijke “bovendorpen” en ook Heusden zelf zich vervolgens meer op de rest van Brabant zijn gaan oriënteren.
0
Ook de absolute jij-vorm komt voor, bijvoorbeeld als verkapt ik-perspectief.
0
Gewoon even uit nieuwgierigheid, als ik een onderwerp toevoeg aan De Kroeg en dan op opslaan klik, wordt dan alle tekst die al op de kroeg stond opnieuw opgeslagen?
0
Als er overlegd wordt over de naam van een pagina of over het samenvoegen met een andere pagina, noem dan altijd de huidige paginanaam.
0
Later maake zijn spontane aanpak plaats voor een meer overwogen stijl.
0
Maar Pieter Kuiper vindt dat “zijn” historicus de enige juiste visie heeft, en bestempelt de historicus die ik aanhaalde als “volslagen onbetrouwbaar”.
0
Zet op Wiki een bijdrage neer, koppel er een licentie aan, en gebruik hem daarna rechtenvrij!
0
Ik heb er geen probleem met Checkusers vertrouwen te geven, waar ik wel een probleem mee heb dat er geen bewijs wordt geleverd.
0
Dit was verder een eenmalig bericht van mij, ik hoop dat er wat mee wordt gedaan en dat de betrokken personen zich aangesproken voelen.
0
```

As was already clear during earlier training phases of the project, my implementation on top of RobBERT seems to lead to
less false positives then BERTje. The reason for this could be in choices of parameters made in underlaying models. The
value of mcc (the Matthews correlation coefficient) is obviously zero, since this is an imbalanced validation set with
only negatives.

### "dt"-fouten for **trained** verbs

Adding to this 10,000 negative labels, a set of 100 sentences with "dt" mistake in the _trained verbs_ (`worden`, `zenden`, `vinden`),
yields these results:

* based on `RobBERT` : **{'mcc': 0.899370194601987, 'tp': 83, 'tn': 9998, 'fp': 2, 'fn': 17, 'eval_loss': 0.01631144658936272}**

```
# I added the '*' marks around the incorrect verb after running the program for the false negatives.

[[ 5.208954  -5.6316767]
 [ 4.8714986 -5.372314 ]
 [ 5.1642437 -5.710761 ]
 ...
 [-4.5101824  5.042265 ]
 [-4.9725046  5.5225797]
 [ 5.0315113 -5.4491963]]
Toen De Ruyter uitvoer, was Rupert eerst zeer verheugd.
0
Het is een bundel met documentaire-achtige verhalen over muzikanten en belangrijke Groningse muziekplekken zoals Vera en Het Viadukt.
0
Hoewel tegenwoordig de joule in Nederland uitgesproken *word* als rijmend op zwoel of zwoele, werd vroeger meestal de uitspraak rijmend op het Engelse owl gehanteerd.
1
Soms wordt gegrapt dat de barbier een vrouw is, waardoor de paradox vermeden *word*.
1
Gedurende één celcyclus *vind* volledige opbouw en afbraak van de celkern plaats.
1
Toekenning van zo'n status *vind* plaats op basis van een inventarisatie van beveiligingsmaatregelen door controleurs van het ministerie van Justitie.
1
In de omgeving van 's-Gravenvoeren *vind* men verlaten grind-, krijt-, kiezeloöliet- en silexgroeven.
1
De eerste bronnen *vind* men in de gezangen die in de synagogen en tempels werden gebezigd om de Heilige Schrift voor te dragen.
1
Heb je wettelijk recht op 20 dagen betaald verlof, dan *word* bijvoorbeeld in de maand juli 5 dagen gewoon loon uitbetaald, en 20 dagen 'enkel vakantiegeld'.
1
Als je geraakt *word* door beide speren tegelijkertijd, word je gevoelloos.
1
Dit is vaak niet verstandig, maar het *word* je soms te veel.
1
En daarmee *word* het even stil: ik althans wil niet over één nacht ijs, en ik ga eerst eens een eigen subpagina maken om mijn gedachten vorm te geven.
1
Omdat het artikel eventueel volgende week woensdag, 9 november, verwijderd *word*, wilde ik u vragen of u mij misschien kan helpen?
1
Een aantal werken *vind* men hieronder.
1
De kenmerken die typisch zijn voor Amerikaanse soaps *vindt* je in "Verbotene Liebe" niet terug.
1
In Blackfield *vindt* je ook het Blackfield Stadium, de motorrijschool en de Blackfield Chapel.
1
Als je *vind* dat hersenloze gebruikers niet mogen knippen ga alsjeblieft aan de slag met de principes van Wikipedia.
1
Hieronder *vindt* je een lijst van genomineerden en uiteindelijke winnaars van een Vlaamse Musicalprijs.
1
Het is een hele boterham, *zendt* het toch maar direct door.
1
```

* based on `BERTje` : **{'mcc': 0.7793952404806573, 'tp': 69, 'tn': 9991, 'fp': 9, 'fn': 31, 'eval_loss': 0.03574260821650488}**

```
# I added the '*' marks around the incorrect verb after running the program for the false negatives.


[[ 5.524736  -5.7299075]
 [ 5.3889146 -5.4308124]
 [ 5.4801917 -5.717814 ]
 ...
 [ 4.581265  -5.173634 ]
 [-4.737512   4.4887786]
 [ 4.921472  -5.473282 ]]
De Bergsche Maas doorsneed het Land van Heusden, waarvan de zuidelijke “bovendorpen” en ook Heusden zelf zich vervolgens meer op de rest van Brabant zijn gaan oriënteren.
0
Ook de absolute jij-vorm komt voor, bijvoorbeeld als verkapt ik-perspectief.
0
Gewoon even uit nieuwgierigheid, als ik een onderwerp toevoeg aan De Kroeg en dan op opslaan klik, wordt dan alle tekst die al op de kroeg stond opnieuw opgeslagen?
0
Als er overlegd wordt over de naam van een pagina of over het samenvoegen met een andere pagina, noem dan altijd de huidige paginanaam.
0
Later maake zijn spontane aanpak plaats voor een meer overwogen stijl.
0
Maar Pieter Kuiper vindt dat “zijn” historicus de enige juiste visie heeft, en bestempelt de historicus die ik aanhaalde als “volslagen onbetrouwbaar”.
0
Zet op Wiki een bijdrage neer, koppel er een licentie aan, en gebruik hem daarna rechtenvrij!
0
Ik heb er geen probleem met Checkusers vertrouwen te geven, waar ik wel een probleem mee heb dat er geen bewijs wordt geleverd.
0
Dit was verder een eenmalig bericht van mij, ik hoop dat er wat mee wordt gedaan en dat de betrokken personen zich aangesproken voelen.
0
Het *verbind* steden als Madrid, Barcelona, Málaga, Sevilla en Valladolid.
1
Pas in 1968 *word* Mauritius onafhankelijk.
1
Drugsgebruik *word* in verband gebracht met een stijging van het risico op het vervroegd ontwikkelen van een psychotische stoornis maar is niet de oorzaak.
1
Soms wordt gegrapt dat de barbier een vrouw is, waardoor de paradox vermeden *word*.
1
Het *word* zo genoemd omdat Marokko destijds het uiterste westelijk gelegen gebied was.
1
In Australië *word* een wind- of stofhoos een Willy Willy genoemd, naar de traditionele Aboriginalnaam ervoor.
1
Niettemin is, behoudens bijzondere omstandigheden, het gebruik van ammoniumnitraat de laatste decennia altijd goedkoper geweest, zodat de methode eigenlijk tegenwoordig geen toepassing meer *vind*.
1
Toekenning van zo'n status *vind* plaats op basis van een inventarisatie van beveiligingsmaatregelen door controleurs van het ministerie van Justitie.
1
De paring *vind* plaats in het water.
1
In de omgeving van 's-Gravenvoeren *vind* men verlaten grind-, krijt-, kiezeloöliet- en silexgroeven.
1
Soms voedt hij zich met andere ongewervelden die hij op de bosbodem *vind*.
1
Hij *vind* zijn oorsprong op een hoogte van 1287 meter en heeft een lengte van 103 km.
1
De eerste bronnen *vind* men in de gezangen die in de synagogen en tempels werden gebezigd om de Heilige Schrift voor te dragen.
1
Het *vind* plaats in de periode na de Ramadan, na het Suikerfeest en vóór het Offerfeest.
1
Met de subductie van de Pacifische plaat onder de Zuid-Amerikaanse, *vind* en vond in heel westelijk Zuid-Amerika vulkanisme plaats.
1
Heb je wettelijk recht op 20 dagen betaald verlof, dan *word* bijvoorbeeld in de maand juli 5 dagen gewoon loon uitbetaald, en 20 dagen 'enkel vakantiegeld'.
1
Als je geraakt *word* door beide speren tegelijkertijd, word je gevoelloos.
1
Dit is vaak niet verstandig, maar het *word* je soms te veel.
1
En daarmee *word* het even stil: ik althans wil niet over één nacht ijs, en ik ga eerst eens een eigen subpagina maken om mijn gedachten vorm te geven.
1
Omdat het artikel eventueel volgende week woensdag, 9 november, verwijderd *word*, wilde ik u vragen of u mij misschien kan helpen?
1
Een lijnbus is een bus die *word* ingezet voor openbaar vervoer.
1
Als de rechter *vind* dat iemand wel schuldig is en daarvoor ook moet worden gestraft, dan kan hij verschillende soorten straffen opleggen.
1
In 1894 *vind* een scheuring in de SDB plaats en wordt de Sociaal-Democratische Arbeiderspartij opgericht.
1
Daarna houdt de officier van justitie zijn requisitoir: een betoog waarin hij de rechter vertelt wat hij van de zaak *vind* en een straf eist.
1
De kenmerken die typisch zijn voor Amerikaanse soaps *vindt* je in "Verbotene Liebe" niet terug.
1
De paring *vind* plaats in de woonbuis van het vrouwtje en daar maakt ze ook het eipakket.
1
Als je *vind* dat hersenloze gebruikers niet mogen knippen ga alsjeblieft aan de slag met de principes van Wikipedia.
1
De openingsscène *vind* plaats in de wijk Central Ward in Newark, New Jersey.
1
Hoe *vind* u onze service?
1
Soms moet je toch wel even zoeken voor je de oplossing *vind*.
1
Het is een hele boterham, *zendt* het toch maar direct door.
1
```

### Testing on untrained verbs and other spelling mistakes

To the training set, I first added 50 more examples of untrained verbs with spelling mistakes.
Total is then:

* 10,000 negatives
* 100 positives on trained verbs
* 50 positives on untrained verbs

The prediction of spelling mistakes on untrained verbs is weak (30 tp, 20 fn, that is close to random ...).
So this approach does not easily seem to generalize from training on a few common verbs with 'dt' mistakes
to other verbs with 'dt' mistakes. Maybe this does get better when more different verbs and variable sentences
are trained. By design, Wikipedia has a quite "uniform" descriptive nature about facts and historic situations.

* based on `RobBERT`: {'mcc': 0.8586375766558695, 'tp': 113, 'tn': 9998, 'fp': 2, 'fn': 37, 'eval_loss': 0.05333949709226241}
* based on `BERTje`:  {'mcc': 0.7794099474524135, 'tp': 100, 'tn': 9991, 'fp': 9, 'fn': 50, 'eval_loss': 0.06995266284639894}


As a last step, I also added 50 more exmaples of other random spelling mistakes.

* based on `RobBERT`: {'mcc': 0.7450613407884148, 'tp': 114, 'tn': 9998, 'fp': 2, 'fn': 86, 'eval_loss': 0.08463327962564625}
* based on `BERTje`:  {'mcc': 0.6766900170856324, 'tp': 101, 'tn': 9991, 'fp': 9, 'fn': 99, 'eval_loss': 0.10399121355988196}

For the random spelling mistakes, there is _no_ prediction at all. Of the 50 examples, only 1 was
discovered as a true positive, the other 49 are false negative (the `fn` value increases with 49, for bith models) :-/


## Performance

For the training, we used the CPU (7 cores on my laptop), and this took **130 minutes** in the end. One of the
reasons I used the CPU (and not the GPU) for training, is that I would run into OOM (Our Of Memory) errors on
the GPU quickly. There was 2.4 GB of the 4 GB assigned for the python program, but already early in the process,
some OOM would show up. This did not occur with the CPU training, and did also not occur with the GPU inference
(both with eval_batch_size of default 8 and the manually set 100).

For the inference, I saw a factor 4 faster inference on the GPU (nvidia leightweight Mobile GPU of a few years old).
By using a batch size of approx. 100, an inference costed about **30 ms per inference** (3 seconds per batch of 100 inferences).

The performance seems very similar (within a few percent) for BERT and RoBERTa based models.


## Discussion

For the "narrow" task of seeing "dt"-fouten for verbs for which I explicitly trained, good accuracy can be obtained
(but a 100% accuracy does not seem realistic at this level of limited training volume and possibly also "limited"
model complexity ?).

For this task, the RoBERTa based model (RobBERT) seems more accurate, compared to the BERT based model (BERTje)
both in true positives as avoiding false positives. Maybe (???) this is caused by the more modern approach in RoBERTa
and the fact that I do not use multi-sentence correlation in my experiments. Also, the vocabulary is very different
for BERTje (a set of 30,000 parts of words with WordPiece) vs. RobBERTa (a set of 50,000 English words, and probably
only the single letter entries are really used from the vocab). It is unclear to me which impact this different
vocab has on the results.

It is clear that at this level of experimentation, training the Dutch language model on a few verbs for "dt"-fouten,
does NOT scale well to "dt-fouten" in other verbs and does NOT scale at all to other random spelling mistakes.

## Future work

A large number of extensions could be imagined to improve the model or verify if this approach is feasible for
production applications (good quality validation of "dt"-fouten and other spelling mistakes in Dutch texts).

Possible ideas:
* Train and validate with large quantity of "dt" sensitive verbs  \
  (would it generalize then ??)
* also train "men", "u", "ge", "gij", ...
* Try other typical spelling mistakes

* Do we train only the top Sequence layer, or can we also train
  one of the core "dense" layers ?
* Could we use a better tokenizer (maybe focussed on the "stam" of
  the verbs, like "zend", "vind", "word", "bid", "baad", ...)
* Could we use a training set that is "cleaned" to have mostly
  exactly correct spelling. Using the model to evaluate its own
  source data might reveal some suspicious cases, that can then
  be validated or corrected manually (I am already doing that small
  scale manually on a corrected version of the nl.wikipedia export).
* Could we use a training set that is a mix of e.g.

  * 50% Dutch
  * 30% French
  * 10% English
  * 5% German
  * 5% other languages

  to accommodate the fact that in Belgium/Netherlands, there is a mix of languages.
* Could we use larger models (would that improve the accuracy ?)
* Could we use better tokenizers ?
* Could we use more diverse corpora (books, letters, news,  emails, ...)?
* Could we get funding from one of the cloud players for GPU/TPU machine time?

If you want to comment, feel free to add an issue, contact me on twitter at @peter_v
or send me an email at peter AT vandenabeele DOT com. Also, I do have the models
available (500 MB per model), if you would like to play with them.

## Conclusions

The primary goal of this exercise during my #CoronaSabbatical of May 2020, clearly succeeded:
* it was a nice way to get some hands-on experience with #NLP and #BERT #AI
* I got an initial indication how far you can generalize from examples with a strong #NLP model like #BERT.

For narrow tasks, the accuracy is good on pre-trained verbs (`worden`, `vinden`, `zende`), but the
generalization to "dt"-fouten for other verbs failed in my experiments.
