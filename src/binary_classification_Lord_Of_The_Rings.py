from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Theoden was the king of Rohan", 1],
    ["Frodo was the heir of Isildur", 0],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
    ["Theoden was the king of Rohan", 1],
    ["Merry was the king of Rohan", 0],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = {
    "num_train_epochs": 10,
    "overwrite_output_dir": 1,
}

print('before model creation')

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args, use_cuda=False,
)

print('after model creation')

# Train the model
model.train_model(train_df)

print('after model training')

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print(model_outputs)
for wrong_prediction in wrong_predictions:
    print(wrong_prediction)

print('after model evaluation')

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])
print(predictions)
print(raw_outputs)

predictions, raw_outputs = model.predict(["Aragorn was the heir of Isildur"])
print(predictions)
print(raw_outputs)

predictions, raw_outputs = model.predict(["Frodo was the heir of Isildur"])
print(predictions)
print(raw_outputs)
