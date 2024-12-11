import pandas as pd
from deep_translator import GoogleTranslator
import subprocess
import logging
import time
from huggingface_hub import hf_hub_download, login

# Configure logging for better tracking and debugging
logging.basicConfig(
    filename="translation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Authenticate with Hugging Face
try:
    logging.info("Authenticating with Hugging Face...")
    login("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")  # Replace with your actual Hugging Face token
    logging.info("Successfully authenticated with Hugging Face.")
except Exception as e:
    logging.error(f"Error during Hugging Face authentication: {e}. Exiting...")
    exit(1)

# Download dataset file from Hugging Face
try:
    logging.info("Downloading dataset from Hugging Face...")
    # Download the dataset JSONL file
    file_path = hf_hub_download(
        repo_id="Tensoic/airoboros-3.2_kn",
        filename="data.jsonl",  # Replace with the correct file name if different
        repo_type="dataset"
    )
    logging.info(f"Dataset downloaded successfully to {file_path}.")

    # Load the JSONL file into a Pandas DataFrame
    df = pd.read_json(file_path, lines=True)
    logging.info(f"Dataset loaded successfully with {len(df)} rows.")
except Exception as e:
    logging.error(f"Error downloading or loading dataset: {e}. Exiting...")
    exit(1)

# Function to split text into chunks of maximum length, ending at logical points (e.g., full stop)
def split_text_into_chunks(text, max_length=4975):
    chunks = []
    while len(text) > max_length:
        split_idx = text[:max_length].rfind(".") + 1  # Find the last full stop within the limit
        if split_idx == 0:  # No full stop found, force split
            split_idx = max_length
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:  # Add remaining text
        chunks.append(text.strip())
    return chunks

# Function to translate text using GoogleTranslator
def translate_to_sinhala(text):
    try:
        return GoogleTranslator(source="kannada", target="sinhala").translate(text)
    except Exception as e:
        logging.error(f"Error translating text '{text}': {e}")
        return text  # Return original text on failure

# Initialize list to store translated data
translated_data = []

# Start translation from the 161st row
start_row = 160  # Zero-based index, so 161st row corresponds to index 160

# Translate row by row starting from the specified index
for index, row in df.iloc[start_row:].iterrows():  # Use iloc to skip the first 160 rows
    try:
        translated_chunks = []

        # Translate input field, handling large text
        for chunk in split_text_into_chunks(row["input"], max_length=4975):
            translated_chunks.append(translate_to_sinhala(chunk))
        translated_input = " ".join(translated_chunks)  # Combine all translated chunks

        # Translate other fields
        translated_instruction = translate_to_sinhala(row["instruction"])
        translated_output = translate_to_sinhala(row["output"])

        # Add translated row to data
        translated_row = {
            "input": translated_input,
            "instruction": translated_instruction,
            "output": translated_output,
        }
        translated_data.append(translated_row)
        logging.info(f"Successfully translated row {index + 1}/{len(df)}.")

        # Save progress after every iteration
        progress_file = "translated_dataset.csv"
        pd.DataFrame(translated_data).to_csv(progress_file, index=False)
        logging.info(f"Progress saved to {progress_file}.")

        # Commit and push updates to GitHub
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Progress: Translated row {index + 1}"], check=True)
            subprocess.run(["git", "push"], check=True)
            logging.info(f"Changes pushed to GitHub after translating row {index + 1}.")
        except subprocess.CalledProcessError as git_error:
            logging.error(f"Git error during push: {git_error}")
            logging.warning("Retrying Git operations in the next iteration...")

    except KeyError as key_error:
        logging.error(f"Missing expected key in dataset row {index}: {key_error}. Skipping row.")
        continue
    except Exception as e:
        logging.error(f"Unexpected error processing row {index}: {e}. Skipping row.")
        time.sleep(2)  # Add delay to avoid rapid failures

# Final save
final_file = "translated_dataset_final.csv"
try:
    final_df = pd.DataFrame(translated_data)
    final_df.to_csv(final_file, index=False)
    logging.info(f"Translation complete. Final dataset saved to {final_file}.")
except Exception as e:
    logging.error(f"Error saving final dataset: {e}")
