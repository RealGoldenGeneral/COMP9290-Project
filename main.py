import data_processing
import sys

MICROBATCH_SIZE = 4
ACCUM_STEPS = 4
MAX_LENGTH = 256
SEED = 9290
def main(filepath):
    train_df, val_df, test_df = data_processing.process_data(filepath, SEED)
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    data = {}
    tokenizer = data_processing.load_tokenizer()
    for name, split in splits.items():
        split = data_processing.tokenize_df(split, tokenizer, MAX_LENGTH, name)
        data[name] = data_processing.create_batches (split, MICROBATCH_SIZE, ACCUM_STEPS, shuffle=(name == "train"), seed=SEED)
    return data

if __name__ == "__main__":
    data = main(sys.argv[1])
    first_batch = data["train"][0]
    mb = first_batch[0]
    print("Input ID shape:", mb["input_ids"].shape)
    print("Attention mask shape:", mb["attention_mask"].shape)
    print("Labels shape:", mb["labels"].shape)
    print("Number of microbatches per accumulation group:", len(first_batch))