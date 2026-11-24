import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

# Process file and create train/test/validation split
def process_data(filepath):

  # Read the tab-separated file
  df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "text"])

  # Balance the classes
  spam_df = df[df["label"] == "spam"]
  ham_df = df[df["label"] == "ham"].sample(n=len(spam_df))
  df = pd.concat([spam_df, ham_df])

  # Split into train, validation, and test dataframes
  train_df = df.sample(frac=0.85, random_state=42)
  test_df = df.drop(train_df.index)
  val_df = test_df.sample(frac=0.25, random_state=42)
  test_df = test_df.drop(val_df.index)

  return train_df, val_df, test_df

# Tokenize dataframes to Qwen 3.5 tokens
def tokenize_df(df):
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
  df["tokens"] = df["text"].apply(lambda x: tokenizer.tokenize(x))
  df["token_count"] = df["tokens"].apply(lambda x: len(x))
  return df

# Pad tokens to longest length
def pad_tokens(df):
  max_length = df["token_count"].max()
  df["tokens"] = df["tokens"].apply(lambda x: x + ["<pad>"] * (max_length - len(x)))
  df["token_count"] = max_length
  return df

# Create chunk dataframes
def create_chunks(df, chunk_size):
  chunks = []
  for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunks.append(chunk)
  return chunks

# Create batches (concatenates chunks first to allow larger batch sizes across chunk boundaries)
def create_batches(chunks, batch_size):
  import pandas as pd
  # Combine chunks back into a single DataFrame to create batches properly across boundary limits
  combined_df = pd.concat(chunks).reset_index(drop=True)
  batches = []
  for i in range(0, len(combined_df), batch_size):
    batch = combined_df.iloc[i:i+batch_size]
    batches.append(batch)
  return batches