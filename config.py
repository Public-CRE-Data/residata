"""
Global configuration for the REIT rental scraper.
"""
import os

# --- REIT targets ---
REITS = ["MAA", "CPT", "EQR", "AVB", "INVH", "AMH", "UDR", "ESS"]

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
WEEKLY_DIR = os.path.join(DATA_DIR, "weekly")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CUMULATIVE_EXCEL = os.path.join(OUTPUT_DIR, "reit_rental_tracker.xlsx")

# --- Request settings ---
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
REQUEST_TIMEOUT = 20          # seconds per page
DELAY_BETWEEN_PAGES = 1.5     # seconds between community requests (be polite)
MAX_RETRIES = 3

# --- 22-field data model column order ---
SCHEMA_COLUMNS = [
    "reit",
    "scrape_date",
    "scrape_week",
    "community_name",
    "state",
    "city",
    "zip_code",
    "address",
    "latitude",
    "longitude",
    "community_url",
    "rentcafe_property_id",
    "unit_number",
    "floorplan_name",
    "bedrooms",
    "bathrooms",
    "sqft",
    "floor_level",
    "asking_rent",
    "available_from",
    "has_concession",
    "concession_description",
]
