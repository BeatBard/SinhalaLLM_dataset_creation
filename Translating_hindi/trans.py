import pandas as pd
from deep_translator import GoogleTranslator
import subprocess
import logging
import time
from datasets import load_dataset
from huggingface_hub import login

# Configure logging for better tracking and debugging
logging.basicConfig(
    filename="translation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Authenticate with Hugging Face
try:
    logging.info("Authenticating with Hugging Face...")
    login("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")  # Replace with your Hugging Face token
    logging.info("Authenticated with Hugging Face successfully.")
except Exception as e:
    logging.error(f"Authentication failed: {e}. Exiting...")
    exit(1)

# Load the dataset
try:
    logging.info("Loading 'anudesh' config with 'hi' split from Hugging Face...")
    dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", "anudesh", split="hi")
    logging.info(f"Successfully loaded 'anudesh' config with 'hi' split containing {len(dataset)} rows.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}. Exiting...")
    exit(1)

# Function to split text into chunks of max length 4975
def split_text_into_chunks(text, max_length=4975):
    chunks = []
    while len(text) > max_length:
        # Find the last full stop within the max length to split logically
        split_idx = text[:max_length].rfind(".") + 1
        if split_idx == 0:  # No full stop found, force split at max length
            split_idx = max_length
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:
        chunks.append(text.strip())
    return chunks

# Function to translate text using GoogleTranslator
def translate_to_sinhala(text):
    try:
        return GoogleTranslator(source="hi", target="si").translate(text)
    except Exception as e:
        logging.error(f"Error translating text '{text}': {e}")
        return text  # Return the original text on failure

# Initialize list to store translated data
translated_data = []

# Translate the dataset row by row
for index, row in enumerate(dataset):
    try:
        messages = row.get("messages", [])
        if not messages or not isinstance(messages, list):
            logging.warning(f"Row {index} is missing 'messages' or has an invalid format. Skipping...")
            continue

        # Extract user prompt and assistant output
        prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
        output = next((m["content"] for m in messages if m["role"] == "assistant"), None)

        if not prompt or not output:
            logging.warning(f"Row {index} is missing a valid 'prompt' or 'output'. Skipping...")
            continue

        # Handle chunking for prompt and output if longer than 4975 characters
        translated_prompt = (
            " ".join(translate_to_sinhala(chunk) for chunk in split_text_into_chunks(prompt))
            if len(prompt) > 4975 else translate_to_sinhala(prompt)
        )
        translated_output = (
            " ".join(translate_to_sinhala(chunk) for chunk in split_text_into_chunks(output))
            if len(output) > 4975 else translate_to_sinhala(output)
        )

        # Add translated data to list
        translated_data.append({"prompt": translated_prompt, "output": translated_output})
        logging.info(f"Successfully translated row {index + 1}/{len(dataset)}.")

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

    except Exception as e:
        logging.error(f"Error processing row {index}: {e}. Skipping row...")
        time.sleep(2)  # Add a delay to avoid rapid failures

# Final save
final_file = "translated_dataset_final.csv"
try:
    final_df = pd.DataFrame(translated_data)
    final_df.to_csv(final_file, index=False)
    logging.info(f"Translation completed. Final dataset saved to {final_file}.")
except Exception as e:
    logging.error(f"Error saving final dataset: {e}")
