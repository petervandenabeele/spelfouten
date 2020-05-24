from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Ik word is zonder t", 0],
    ["Ik word warm.", 0],
    ["Ik word enthousiast.", 0],
    ["Ik word eigenaar van een nieuwe kat.", 0],
    ["Hierdoor word ik ook helemaal verrast.", 0],
    ["Hoe word ik eigenlijk geholpen?", 0],
    ["Waarom word ik van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder word ik hierdoor afgezet?", 0],

    ["Ik wordt is zonder t", 1],
    ["Ik wordt warm.", 1],
    ["Ik wordt enthousiast.", 1],
    ["Ik wordt eigenaar van een nieuwe kat.", 1],
    ["Hierdoor wordt ik ook helemaal verrast.", 1],
    ["Hoe wordt ik eigenlijk geholpen?", 1],
    ["Waarom wordt ik van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder wordt ik hierdoor afgezet?", 1],

    ["Jij wordt is met t", 0],
    ["Jij wordt warm.", 0],
    ["Jij wordt enthousiast.", 0],
    ["Jij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor word jij ook helemaal verrast.", 0],
    ["Hoe word jij eigenlijk geholpen?", 0],
    ["Waarom word jij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder word jij hierdoor afgezet?", 0],

    ["Jij word is met t", 1],
    ["Jij word warm.", 1],
    ["Jij word enthousiast.", 1],
    ["Jij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor wordt jij ook helemaal verrast.", 1],
    ["Hoe wordt jij eigenlijk geholpen?", 1],
    ["Waarom wordt jij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder wordt jij hierdoor afgezet?", 1],

    ["Hij wordt is met t", 0],
    ["Hij wordt warm.", 0],
    ["Hij wordt enthousiast.", 0],
    ["Hij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor wordt hij ook helemaal verrast.", 0],
    ["Hoe wordt hij eigenlijk geholpen?", 0],
    ["Waarom wordt hij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder wordt hij hierdoor afgezet?", 0],

    ["Hij word is met t", 1],
    ["Hij word warm.", 1],
    ["Hij word enthousiast.", 1],
    ["Hij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor word hij ook helemaal verrast.", 1],
    ["Hoe word hij eigenlijk geholpen?", 1],
    ["Waarom word hij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder word hij hierdoor afgezet?", 1],

    ["Zij wordt is met t", 0],
    ["Zij wordt warm.", 0],
    ["Zij wordt enthousiast.", 0],
    ["Zij wordt eigenaar van een nieuwe kat.", 0],
    ["Hierdoor wordt zij ook helemaal verrast.", 0],
    ["Hoe wordt zij eigenlijk geholpen?", 0],
    ["Waarom wordt zij van het kastje naar de muur gestuurd?", 0],
    ["Hoeveel verder wordt zij hierdoor afgezet?", 0],

    ["Zij word is met t", 1],
    ["Zij word warm.", 1],
    ["Zij word enthousiast.", 1],
    ["Zij word eigenaar van een nieuwe kat.", 1],
    ["Hierdoor word zij ook helemaal verrast.", 1],
    ["Hoe word zij eigenlijk geholpen?", 1],
    ["Waarom word zij van het kastje naar de muur gestuurd?", 1],
    ["Hoeveel verder word zij hierdoor afgezet?", 1],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data

## RESULTS on this very simple version: 14 of the 16 validations are very strongly correct :-)
# [[ 0.934536   -0.79441744]
#  [-0.38130385  0.34947482]
#  [ 0.56628114 -0.31090865]
#  [ 0.24857064 -0.13078722]  # incorrect
#  [ 1.1392815  -0.768657  ]
#  [-1.1988978   1.0266285 ]
#  [ 0.38991755 -0.34029955]
#  [ 0.34916613 -0.29456055]  # incorrect
#  [ 1.3438416  -0.99393845]
#  [-1.3128942   1.3501518 ]
#  [ 1.1691108  -0.7768973 ]
#  [-0.2947425   0.29919532]
#  [ 1.4490408  -1.1109018 ]
#  [-1.4008088   1.3720468 ]
#  [ 1.3757272  -1.0468719 ]
#  [-0.54108465  0.3590323 ]]
# Wordt ik nu al opgeroepen?
# 1
# Wordt jij nu al opgeroepen?
# 1

eval_data = [
    ["Ik word volgend jaar ook getest.", 0],          #  [ 0.934536   -0.79441744]
    ["Ik wordt helemaal naar hier gehaald.", 1],      #  [-0.38130385  0.34947482]
    ["Word ik volgend jaar ook uitgenodigd?", 0],     #  [ 0.56628114 -0.31090865]
    ["Wordt ik nu al opgeroepen?", 1], # *INCORRECT*  #  [ 0.24857064 -0.13078722]

    ["Jij wordt volgend jaar ook getest.", 0],        #  [ 1.1392815  -0.768657  ]
    ["Jij word helemaal naar hier gehaald.", 1],      #  [-1.1988978   1.0266285 ]
    ["Word jij volgend jaar ook uitgenodigd?", 0],    #  [ 0.38991755 -0.34029955]
    ["Wordt jij nu al opgeroepen?", 1], # *INCORRECT* #  [ 0.34916613 -0.29456055]

    ["Hij wordt volgend jaar ook getest.", 0],        #  [ 1.3438416  -0.99393845]
    ["Hij word helemaal naar hier gehaald.", 1],      #  [-1.3128942   1.3501518 ]
    ["Wordt hij volgend jaar ook uitgenodigd?", 0],   #  [ 1.1691108  -0.7768973 ]
    ["Word hij nu al opgeroepen?", 1],                #  [-0.2947425   0.29919532]

    ["Zij wordt volgend jaar ook getest.", 0],        #  [ 1.4490408  -1.1109018 ]
    ["Zij word helemaal naar hier gehaald.", 1],      #  [-1.4008088   1.3720468 ]
    ["Wordt zij volgend jaar ook uitgenodigd?", 0],   #  [ 1.3757272  -1.0468719 ]
    ["Word zij nu al opgeroepen?", 1],                #  [-0.54108465  0.3590323 ]
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
# with 10 epochs, took a few minutes to train on laptop CPU
model_args = {
    "num_train_epochs": 10,
    "overwrite_output_dir": 1,
}

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-dutch-cased", args=model_args, use_cuda=False,
)
print(type(model))
# <class 'simpletransformers.classification.classification_model.ClassificationModel'>

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
# INFO {'mcc': 0.7745966692414834, 'tp': 6, 'tn': 8, 'fp': 0, 'fn': 2, 'eval_loss': 0.29993630945682526}
print(model_outputs) # see above
for wrong_prediction in wrong_predictions:
    print(wrong_prediction.text_a)
    print(wrong_prediction.label)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Ik wordt nieuwsgierig."])
print(predictions)
print(raw_outputs)
# [1]
# [[-0.5602998  0.8519003]]  => strongly correct

predictions, raw_outputs = model.predict(["Wordt jij ook enthousiast?"])
print(predictions)
print(raw_outputs)
# [0]
# [[ 1.0839487  -0.87757194]] => strongly *INCORRECT*

predictions, raw_outputs = model.predict(["Wordt zij hierdoor bekend?"])
print(predictions)
print(raw_outputs)
# [0]
# [[ 1.0873619 -0.8918497]]  => strongly correct

predictions, raw_outputs = model.predict(["Ik word enthousiast."])
print(predictions)
print(raw_outputs)
# [0]
# [[ 1.5705453 -1.4946615]] => strongly correct
