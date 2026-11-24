import data_processing
import sys

def main(filepath):
    train_df, val_df, test_df = data_processing.process_data(filepath)
    splits = [train_df, val_df, test_df]
    chunks = []
    for split in splits:
        split = data_processing.tokenize_df(split)
        split = data_processing.pad_tokens(split)
        chunk = data_processing.create_chunks(split, 10)
        chunks.append(chunk)
    batches = []
    for chunk in chunks:
        batch = data_processing.create_batches(chunk, 32)
        batches.append(batch)
    return batches

if __name__ == "__main__":
    splits = main(sys.argv[1])
    print(splits[0][0])