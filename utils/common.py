"""
Shared utilities for all REIT scrapers.
"""
import time
import logging
import re
import requests
from datetime import date
from typing import Optional

from config import REQUEST_HEADERS, REQUEST_TIMEOUT, DELAY_BETWEEN_PAGES, MAX_RETRIES

logger = logging.getLogger(__name__)


def get_page(url: str, session: requests.Session) -> Optional[str]:
    """Fetch a URL with retries and polite delay. Returns HTML string or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            time.sleep(DELAY_BETWEEN_PAGES)
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(DELAY_BETWEEN_PAGES * attempt * 2)
    logger.error(f"All retries exhausted for {url}")
    return None


def today_str() -> str:
    return date.today().isoformat()


def iso_week_str() -> str:
    """Return 'YYYY-Www' e.g. '2026-W14'."""
    d = date.today()
    return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"


def parse_int(text: str) -> Optional[int]:
    """Extract first integer from a string, e.g. '$1,138' -> 1138."""
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def parse_float(text: str) -> Optional[float]:
    """Extract first float from a string."""
    if not text:
        return None
    m = re.search(r"[-\d.]+", text.strip())
    return float(m.group()) if m else None
