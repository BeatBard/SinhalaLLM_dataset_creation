import pandas as pd
from deep_translator import GoogleTranslator
from datasets import load_dataset
import subprocess
import os
import time

# Load the ajibawa-2023/WikiHow dataset from Hugging Face
try:
    dataset = load_dataset("ajibawa-2023/WikiHow", split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)  # Exit the script if dataset loading fails

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(dataset)

# Function to translate text into Sinhala using DeepTranslator
def translate_to_sinhala(text):
    try:
        translated_text = GoogleTranslator(source="en", target="si").translate(text)
        return translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text  # Return original text in case of failure

# Function to split text into chunks ending in a full stop (max chunk size: 4975)
def split_text_into_chunks(text, chunk_size=2000):
    chunks = []
    try:
        while len(text) > chunk_size:
            split_idx = text[:chunk_size].rfind(".") + 1
            if split_idx == 0:  # No full stop found, force split
                split_idx = chunk_size
            chunks.append(text[:split_idx].strip())
            text = text[split_idx:].strip()
        if text:  # Append remaining text
            chunks.append(text.strip())
    except Exception as e:
        print(f"Error splitting text: {e}")
    return chunks

# Create a new DataFrame to store translated content
translated_data = []

# Iterate through the rows of the dataset
for index, row in df.iterrows():
    try:
        # Translate the 'prompt' column
        prompt_si = translate_to_sinhala(row['prompt'])

        # Handle long text in the 'text' column
        if len(row['text']) > 4975:
            text_chunks = split_text_into_chunks(row['text'], chunk_size=2000)
            text_si_chunks = [translate_to_sinhala(chunk) for chunk in text_chunks]
        else:
            text_si_chunks = [translate_to_sinhala(row['text'])]

        # Append translated rows to the new data list
        for chunk in text_si_chunks:
            translated_data.append({"prompt": prompt_si, "text": chunk})

        # Save progress after every iteration
        pd.DataFrame(translated_data).to_csv("translated_wikihow_dataset.csv", index=False)
        print(f"Row {index + 1} translated and saved.")

        # Commit and push changes to GitHub
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Progress: Translated row {index + 1}"], check=True)
            subprocess.run(["git", "push"], check=True)
            print(f"Row {index + 1} changes pushed to GitHub.")
        except subprocess.CalledProcessError as git_error:
            print(f"Git error: {git_error}")
            print("Retrying Git operations in the next iteration...")

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(5)  # Add a delay to prevent rapid failures

# Convert the translated data into a new DataFrame (Final Save)
translated_df = pd.DataFrame(translated_data)

# Save the translated dataset as a CSV file
try:
    translated_df.to_csv("translated_wikihow_dataset.csv", index=False)
    print("Final translation and processing complete. File saved as 'translated_wikihow_dataset.csv'.")
except Exception as e:
    print(f"Error saving final dataset: {e}")
