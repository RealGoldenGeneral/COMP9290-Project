import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch

COMMIT = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"

# Process file and create train/test/validation split
def process_data(filepath, seed):

  # Read the tab-separated file
  df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "text"])
  print("Rows before deduplication: ", len(df))

  # Drop duplicates
  df = df.drop_duplicates(subset="text")
  print("Rows after deduplication: ", len(df))

  value_counts = df["label"].value_counts()
  print("Spam:", value_counts["spam"])
  print("Ham:", value_counts["ham"])

  # Split into train, validation, and test dataframes
  train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
  val_df, test_df = train_test_split(temp_df, test_size=2/3, stratify=temp_df["label"], random_state=seed)

  train_df = train_df.reset_index(drop=True)
  val_df = val_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  print(f"Training rows: {len(train_df)}, {int(len(train_df)/len(df) * 100)}% split.")
  print(f"Validation rows: {len(val_df)}, {int(len(val_df)/len(df) * 100)}% split.")
  print(f"Testing rows: {len(test_df)}, {int(len(test_df)/len(df) * 100)}% split.")

  return train_df, val_df, test_df

# Load tokenizer
def load_tokenizer():
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base", revision=COMMIT, padding_side="right")
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  print("Qwen 3.5 Tokenizer initalized.")
  return tokenizer

# Tokenize dataframes to Qwen 3.5 tokens
def tokenize_df(df, tokenizer, max_length, name):
  # Snippet AI generated
  enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length, padding="max_length")
  df = df.copy()
  df["input_ids"] = enc["input_ids"]
  df["attention_mask"] = enc["attention_mask"]
  df["labels"] = df["label"].map({"ham": 0, "spam": 1})
  print(f"N/A tokens in {name}:", df.isna().all(axis=1).sum())
  return df

# Create batches 
def create_batches(df, batch_size, accum_steps, shuffle, seed):
  if shuffle:
    df = df.sample(frac=1, random_state=seed)
  
  microbatches = []
  for i in range(0, len(df), batch_size):
    mb = df.iloc[i:i + batch_size]
    microbatches.append({
      "input_ids": torch.tensor(mb["input_ids"].tolist(), dtype=torch.long),
      "attention_mask": torch.tensor(mb["attention_mask"].tolist(), dtype=torch.long),
      "labels": torch.tensor(mb["labels"].tolist(), dtype=torch.long)
    })

  batches = []
  for j in range(0, len(microbatches), accum_steps):
    batches.append(microbatches[j:j + accum_steps])
  return batches