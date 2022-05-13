DATSET_PATH = "data/stsbenchmark"
FILE_PATH = {
    "TRAIN":"sts_train_cleaned.csv",
    "TEST":"sts_dev_cleaned.csv",
    "DEV":"sts_test_cleaned.csv"
}


OUTPUT_PATH = "data/output"

max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 100

# # Labels in our dataset.
# labels = ["contradiction", "entailment", "neutral"]