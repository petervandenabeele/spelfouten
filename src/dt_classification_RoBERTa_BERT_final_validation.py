# Run this with python ./src/dt_classification_RoBERTa_BERT_final_validation.py

from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# EXPERIMENT 2020-06-01 12:30
#
# Using RoBERTa-006 and BERT-002 trained models on large validation set:
#
# 10,000 new, unseen, hopefully, "correct" Dutch sentences from nl.wikipedia
#
# 100%|███████| 10000/10000 [00:01<00:00, 6828.97it/s]
# 100%|███████|   1250/1250 [04:35<00:00,  4.54it/s]
# {'mcc': 0.0, 'tp': 0, 'tn': 9998, 'fp': 2, 'fn': 0, 'eval_loss': 0.0010510552614927291}
# [[ 5.208954  -5.6316767]
#  [ 4.8714986 -5.3723135]
#  [ 5.1642437 -5.710761 ]
#  ...
#  [ 4.9983373 -5.526781 ]
#  [ 5.0196595 -5.548851 ]
#  [ 4.6084604 -5.089196 ]]
#
# # Toen De Ruyter uitvoer, was Rupert eerst zeer verheugd.
# 0
# Het is een bundel met documentaire-achtige verhalen over muzikanten en belangrijke Groningse muziekplekken zoals Vera en Het Viadukt.
# 0

# with 100 sentences with on purpose incorrect "dt" mistakes in the trained verbs (worden, zenden, vinden)
#
# with 100 sentences with on purpose incorrect "dt" mistakes in non-trained verbs
#
# with 100 sentences with on purpose spelling mistakes in other words


# Preparing eval data
# Reference 10K sentences that are (hopefully) correct

correct_10K_sentences_file_name = "./data/correct-sentences-nlwiki-10K.txt"
sentences_file_name = correct_10K_sentences_file_name

eval_data = []

with open(sentences_file_name, 'r') as sentences_file:
    for line in sentences_file:
        clean_line = line.rstrip()
        eval_data.append([line, 0])

eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]


# Optional model configuration
model_args = {
    "overwrite_output_dir": 0, # we are moving the dir to a named dir now
    "output_dir": "outputs/RoBERTa"
}

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "outputs/RoBERTa-006", args=model_args, use_cuda=True,
)
print(type(model))
# <class 'simpletransformers.classification.classification_model.ClassificationModel'>

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(model_outputs) # see above

# Print the incorrect predictions
for wrong_prediction in wrong_predictions:
    print(wrong_prediction.text_a)
    print(wrong_prediction.label)
