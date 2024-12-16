import os
import gc
import pandas as pd
import logging
import time
from deep_translator import GoogleTranslator
import subprocess

# Configure logging
logging.basicConfig(
    filename="translation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to calculate byte length of text
def calculate_byte_length(text):
    """Calculate the byte length of a given text."""
    return len(text.encode('utf-8'))

# Function to split text into chunks of max byte length 4975
def split_text_into_chunks_by_bytes(text, max_byte_length=4975):
    """Split text into smaller chunks based on byte size."""
    chunks = []
    current_chunk = ""

    for word in text.split():
        # Check if adding this word exceeds the max byte length
        if calculate_byte_length(current_chunk + " " + word) > max_byte_length:
            chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            current_chunk += " " + word

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to translate text using GoogleTranslator
def translate_to_sinhala(text, retries=3, delay=1):
    """Translate text from Hindi to Sinhala with retries."""
    for attempt in range(retries):
        try:
            time.sleep(delay)  # Add delay for API rate-limiting
            translated_text = GoogleTranslator(source="hi", target="si").translate(text)
            return translated_text
        except Exception as e:
            logging.warning(f"Error during translation: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(2 ** attempt)  # Exponential backoff

    logging.error("Translation failed after multiple attempts.")
    return text  # Return original text as fallback

# Load the input CSV
input_csv = "translated_dataset.csv"  # Replace with your file name
output_csv = "final_translated_dataset.csv"

try:
    # Load the CSV file while skipping bad lines
    input_df = pd.read_csv(input_csv, on_bad_lines="skip")
    logging.info(f"Loaded input CSV with {len(input_df)} rows.")
except Exception as e:
    logging.error(f"Error loading input CSV: {e}")
    exit(1)

# Detect and re-translate untranslated rows
translated_rows = []

for index, row in input_df.iterrows():
    try:
        # Safely access row data
        prompt = row.get("prompt", "")
        output = row.get("output", "")

        # Check if the output is untranslated (e.g., still in Hindi)
        if output.strip() == "" or any("\u0900" <= c <= "\u097F" for c in output):  # Unicode range for Hindi
            logging.info(f"Row {index} requires translation.")

            # Handle chunking for prompt and output if longer than 4975 bytes
            translated_prompt = (
                " ".join(translate_to_sinhala(chunk) for chunk in split_text_into_chunks_by_bytes(prompt))
                if calculate_byte_length(prompt) > 4975 else translate_to_sinhala(prompt)
            )
            translated_output = (
                " ".join(translate_to_sinhala(chunk) for chunk in split_text_into_chunks_by_bytes(output))
                if calculate_byte_length(output) > 4975 else translate_to_sinhala(output)
            )

            # Add the translated row
            translated_rows.append({"prompt": translated_prompt, "output": translated_output})
        else:
            # Keep already translated rows
            translated_rows.append({"prompt": prompt, "output": output})

        # Save progress incrementally
        pd.DataFrame(translated_rows).to_csv(output_csv, index=False)
        logging.info(f"Progress saved after processing row {index}.")

        # Commit and push updates to GitHub
        try:
            subprocess.run(["git", "add", output_csv], check=True)
            subprocess.run(["git", "commit", "-m", f"Updated row {index}"], check=True)
            subprocess.run(["git", "push"], check=True)
            logging.info(f"Changes pushed to GitHub after processing row {index}.")
        except subprocess.CalledProcessError as git_error:
            logging.error(f"Git error during push: {git_error}")

    except Exception as e:
        logging.error(f"Error processing row {index}: {e}. Skipping row...")

    # Clear memory and garbage collect
    gc.collect()

# Final save
try:
    pd.DataFrame(translated_rows).to_csv(output_csv, index=False)
    logging.info(f"Final dataset saved to {output_csv}.")
except Exception as e:
    logging.error(f"Error saving final dataset: {e}")
