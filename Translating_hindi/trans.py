import os
import gc
import pandas as pd
from deep_translator import GoogleTranslator
import subprocess
import logging
import time
from datasets import load_dataset
from huggingface_hub import login

# Configure logging
logging.basicConfig(
    filename="translation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Authenticate with Hugging Face
try:
    logging.info("Authenticating with Hugging Face...")
    login("<your-hugging-face-token>")  # Replace with your Hugging Face token
    logging.info("Authenticated with Hugging Face successfully.")
except Exception as e:
    logging.error(f"Authentication failed: {e}. Exiting...")
    exit(1)

# Load the dataset
try:
    logging.info("Loading 'anudesh' config with 'hi' split from Hugging Face...")
    dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", "anudesh", split="hi", streaming=True)
    logging.info("Dataset loaded in streaming mode.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}. Exiting...")
    exit(1)

# Function to split text into chunks of max length 4975
def split_text_into_chunks(text, max_length=4975):
    chunks = []
    while len(text) > max_length:
        split_idx = text[:max_length].rfind(".") + 1
        if split_idx == 0:
            split_idx = max_length
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:
        chunks.append(text.strip())
    return chunks

# Function to translate text
def translate_to_sinhala(text):
    try:
        time.sleep(1)  # Introduce delay to reduce memory usage
        return GoogleTranslator(source="hi", target="si").translate(text)
    except Exception as e:
        logging.error(f"Error translating text '{text}': {e}")
        return text  # Return original text on failure

# Initialize file for saving progress
progress_file = "translated_dataset.csv"
if not os.path.exists(progress_file):
    pd.DataFrame(columns=["prompt", "output"]).to_csv(progress_file, index=False)

# Process dataset row by row
start_index = sum(1 for _ in open(progress_file)) - 1  # Determine the starting row from the file
logging.info(f"Resuming from row {start_index}.")

for index, row in enumerate(dataset):
    if index < start_index:
        continue  # Skip rows already processed

    try:
        # Extract and validate messages
        messages = row.get("messages", [])
        if not messages or not isinstance(messages, list):
            logging.warning(f"Row {index} has invalid 'messages'. Skipping...")
            continue

        # Extract prompt and output
        prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
        output = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if not prompt or not output:
            logging.warning(f"Row {index} is missing 'prompt' or 'output'. Skipping...")
            continue

        # Translate prompt and output
        translated_prompt = (
            " ".join(translate_to_sinhala(chunk) for chunk in split_text_into_chunks(prompt))
            if len(prompt) > 4975 else translate_to_sinhala(prompt)
        )
        translated_output = (
            " ".join(translate_to_sinhala(chunk) for chunk in split_text_into_chunks(output))
            if len(output) > 4975 else translate_to_sinhala(output)
        )

        # Write translation to file
        with open(progress_file, "a") as f:
            f.write(f'"{translated_prompt}","{translated_output}"\n')

        logging.info(f"Translated row {index + 1}.")

        # Commit and push to GitHub
        try:
            subprocess.run(["git", "add", progress_file], check=True)
            subprocess.run(["git", "commit", "-m", f"Progress: Translated row {index + 1}"], check=True)
            subprocess.run(["git", "push"], check=True)
        except subprocess.CalledProcessError as git_error:
            logging.error(f"Git error: {git_error}")

    except Exception as e:
        logging.error(f"Error processing row {index}: {e}. Skipping...")

    # Clear memory
    gc.collect()

logging.info("Translation process completed.")
