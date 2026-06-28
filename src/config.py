from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "support_tickets.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tickets_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.30
LABEL_MAP = {"non_urgent": 0, "urgent": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
