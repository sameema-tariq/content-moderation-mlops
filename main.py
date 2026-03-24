"""Entry point — runs the full pipeline: extract → preprocess → train → save."""

from pipeline.extract import extract_to_csv
from pipeline.preprocessing import load_preprocessed_data
from pipeline.utils import summarize_sms_info, save_preprocessed_csv
from pipeline.train import train
from config.settings import INPUT_PATH, validate_paths


def main():
    """Run the full ML pipeline: validate → extract → preprocess → train → save."""
    validate_paths()
    df_raw = extract_to_csv(INPUT_PATH)
    df_clean = load_preprocessed_data(df_raw)
    summarize_sms_info(df_clean)
    save_preprocessed_csv(df_clean)
    train(df_clean)


if __name__ == "__main__":
    main()

