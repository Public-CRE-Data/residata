# build_excel.py — REIT Rental Analysis Workbook Builder
#
# SETUP (one-time, on any machine):
#   pip install requests pandas openpyxl
#
# USAGE:
#   1. Set GITHUB_TOKEN below to your fine-grained PAT (read scope: Public-CRE-Data/residata)
#   2. Run: python build_excel.py
#   3. Find output: REIT_Rental_Analysis_{date}.xlsx in your OneDrive folder
#      → accessible from GS remote machine via OneDrive (no install needed)
#
# WEEKLY WORKFLOW:
#   - Scraper runs on main machine → pushes new dated CSVs to GitHub
#   - Run this script on main machine → Excel saves to OneDrive, auto-syncs to remote
#   - Same-property WoW analysis activates automatically from Week 2 onwards

import os
import io
import json
import re
import sys
import requests
import pandas as pd
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.chart.series import SeriesLabel

# Ensure stdout handles Unicode on Windows (cp1252 terminals)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_OWNER  = "Public-CRE-Data"
GITHUB_REPO   = "residata"
# Set GITHUB_TOKEN to your fine-grained PAT (read-only access to this repo)
GITHUB_TOKEN  = ""
DATA_PATH     = "data/raw"          # folder in repo where CSVs live
REGISTRY_PATH = "data/registry/unit_registry.csv"
OUTPUT_DIR    = r"C:\Users\user\OneDrive\Documents\AI@GSB"  # syncs to OneDrive automatically
CACHE_DIR     = "./residata_cache"  # local cache to avoid re-downloading unchanged files

# ─────────────────────────────────────────────────────────────────────────────
# MACRO MARKET MAP
# ─────────────────────────────────────────────────────────────────────────────
MACRO_MAP = {
    # Seattle
    "Seattle": "Seattle", "Bellevue": "Seattle", "Redmond": "Seattle", "Kirkland": "Seattle",
    "Bothell": "Seattle", "Everett": "Seattle", "Renton": "Seattle", "Kent": "Seattle",
    "Federal Way": "Seattle", "Shoreline": "Seattle", "Lynnwood": "Seattle",
    "Marysville": "Seattle", "Issaquah": "Seattle", "Sammamish": "Seattle",
    "Burien": "Seattle", "Covington WA": "Seattle", "Des Moines WA": "Seattle",
    "Edmonds WA": "Seattle", "Monroe WA": "Seattle", "Mountlake Terrace": "Seattle",
    "Mukilteo": "Seattle", "Puyallup": "Seattle", "SeaTac": "Seattle",
    "Snohomish": "Seattle", "Tacoma": "Seattle",

    # Atlanta
    "Atlanta": "Atlanta", "Alpharetta": "Atlanta", "Marietta": "Atlanta",
    "Sandy Springs": "Atlanta", "Smyrna GA": "Atlanta", "Smyrna": "Atlanta",
    "Kennesaw": "Atlanta", "Roswell GA": "Atlanta", "Roswell": "Atlanta",
    "Decatur": "Atlanta", "Duluth GA": "Atlanta", "Duluth": "Atlanta",
    "Norcross": "Atlanta", "Peachtree City": "Atlanta", "Lawrenceville": "Atlanta",
    "Acworth GA": "Atlanta", "Acworth": "Atlanta", "Auburn GA": "Atlanta",
    "Buford": "Atlanta", "Canton GA": "Atlanta", "Carrollton GA": "Atlanta",
    "Chamblee": "Atlanta", "College Park GA": "Atlanta", "Conyers": "Atlanta",
    "Covington GA": "Atlanta", "Cumming": "Atlanta", "Dacula": "Atlanta",
    "Dallas GA": "Atlanta", "Douglasville": "Atlanta", "Dunwoody": "Atlanta",
    "East Point": "Atlanta", "Fayetteville GA": "Atlanta", "Forest Park": "Atlanta",
    "Griffin GA": "Atlanta", "Hiram": "Atlanta", "Johns Creek": "Atlanta",
    "Jonesboro GA": "Atlanta", "Lilburn": "Atlanta", "Lithia Springs": "Atlanta",
    "Loganville": "Atlanta", "Mableton": "Atlanta", "McDonough": "Atlanta",
    "Newnan": "Atlanta", "Powder Springs": "Atlanta", "Riverdale GA": "Atlanta",
    "Rome GA": "Atlanta", "Snellville": "Atlanta", "Stockbridge GA": "Atlanta",
    "Stone Mountain": "Atlanta", "Sugar Hill GA": "Atlanta", "Suwanee": "Atlanta",
    "Tucker GA": "Atlanta", "Union City GA": "Atlanta", "Winder GA": "Atlanta",
    "Woodstock GA": "Atlanta",

    # Miami/Fort Lauderdale
    "Miami": "Miami/Fort Lauderdale", "Fort Lauderdale": "Miami/Fort Lauderdale",
    "Boca Raton": "Miami/Fort Lauderdale", "Dania Beach": "Miami/Fort Lauderdale",
    "Davie": "Miami/Fort Lauderdale", "Hollywood FL": "Miami/Fort Lauderdale",
    "Hollywood": "Miami/Fort Lauderdale", "Plantation FL": "Miami/Fort Lauderdale",
    "Plantation": "Miami/Fort Lauderdale", "Pembroke Pines": "Miami/Fort Lauderdale",
    "Sunrise FL": "Miami/Fort Lauderdale", "Sunrise": "Miami/Fort Lauderdale",
    "Coral Springs": "Miami/Fort Lauderdale", "Aventura": "Miami/Fort Lauderdale",
    "Hallandale Beach": "Miami/Fort Lauderdale", "Hialeah": "Miami/Fort Lauderdale",
    "Homestead": "Miami/Fort Lauderdale", "Kendall FL": "Miami/Fort Lauderdale",
    "Margate FL": "Miami/Fort Lauderdale", "Margate": "Miami/Fort Lauderdale",
    "Miami Beach": "Miami/Fort Lauderdale", "Miami Gardens": "Miami/Fort Lauderdale",
    "Miramar FL": "Miami/Fort Lauderdale", "Miramar": "Miami/Fort Lauderdale",
    "North Lauderdale": "Miami/Fort Lauderdale", "North Miami": "Miami/Fort Lauderdale",
    "Oakland Park FL": "Miami/Fort Lauderdale", "Oakland Park": "Miami/Fort Lauderdale",
    "Boynton Beach": "Miami/Fort Lauderdale", "Delray Beach": "Miami/Fort Lauderdale",
    "Pompano Beach": "Miami/Fort Lauderdale", "Tamarac": "Miami/Fort Lauderdale",
    "West Palm Beach": "Miami/Fort Lauderdale", "Weston FL": "Miami/Fort Lauderdale",
    "Weston": "Miami/Fort Lauderdale",

    # Orlando
    "Orlando": "Orlando", "Winter Park FL": "Orlando", "Winter Park": "Orlando",
    "Kissimmee": "Orlando", "Ocoee": "Orlando", "Altamonte Springs": "Orlando",
    "Lake Mary FL": "Orlando", "Lake Mary": "Orlando", "Sanford FL": "Orlando",
    "Sanford": "Orlando", "Casselberry": "Orlando", "Apopka": "Orlando",
    "Clermont FL": "Orlando", "Daytona Beach": "Orlando", "Deltona": "Orlando",
    "Longwood FL": "Orlando", "Longwood": "Orlando", "Melbourne FL": "Orlando",
    "Orange City FL": "Orlando", "Oviedo": "Orlando", "Palm Bay": "Orlando",
    "Poinciana": "Orlando", "Port Orange": "Orlando", "St Cloud FL": "Orlando",
    "Winter Garden": "Orlando", "Winter Haven": "Orlando", "Winter Springs": "Orlando",
    "Osceola County": "Orlando",

    # Tampa
    "Tampa": "Tampa", "St. Petersburg": "Tampa", "Clearwater": "Tampa",
    "Wesley Chapel": "Tampa", "Sarasota": "Tampa", "Brandon FL": "Tampa",
    "Brandon": "Tampa", "Riverview FL": "Tampa", "Riverview": "Tampa",
    "Land O Lakes": "Tampa", "Bradenton": "Tampa", "Brooksville FL": "Tampa",
    "Dunedin FL": "Tampa", "Hernando County FL": "Tampa", "Largo FL": "Tampa",
    "Largo": "Tampa", "Lutz FL": "Tampa", "Lutz": "Tampa",
    "New Port Richey": "Tampa", "Nokomis": "Tampa", "Palmetto FL": "Tampa",
    "Plant City": "Tampa", "Ruskin FL": "Tampa", "Seffner": "Tampa",
    "Spring Hill FL": "Tampa", "Spring Hill": "Tampa", "St. Pete Beach": "Tampa",
    "Sun City Center": "Tampa", "Temple Terrace": "Tampa", "Venice FL": "Tampa",
    "Wimauma": "Tampa", "Zephyrhills": "Tampa", "Lakeland": "Tampa",

    # Denver
    "Denver": "Denver", "Aurora CO": "Denver", "Aurora": "Denver",
    "Westminster CO": "Denver", "Westminster": "Denver", "Englewood CO": "Denver",
    "Englewood": "Denver", "Lakewood CO": "Denver", "Lakewood": "Denver",
    "Thornton": "Denver", "Arvada": "Denver", "Broomfield": "Denver",
    "Brighton CO": "Denver", "Littleton CO": "Denver", "Littleton": "Denver",
    "Centennial CO": "Denver", "Centennial": "Denver", "Parker CO": "Denver",
    "Parker": "Denver", "Northglenn": "Denver", "Highlands Ranch": "Denver",
    "Castle Rock CO": "Denver", "Commerce City CO": "Denver", "Erie CO": "Denver",
    "Firestone": "Denver", "Fort Collins": "Denver", "Boulder CO": "Denver",
    "Longmont": "Denver", "Louisville CO": "Denver", "Wheat Ridge": "Denver",

    # Raleigh/Durham
    "Raleigh": "Raleigh/Durham", "Durham": "Raleigh/Durham", "Cary": "Raleigh/Durham",
    "Chapel Hill": "Raleigh/Durham", "Morrisville": "Raleigh/Durham",
    "Apex": "Raleigh/Durham", "Wake Forest": "Raleigh/Durham",
    "Garner": "Raleigh/Durham", "Clayton NC": "Raleigh/Durham",
    "Carrboro": "Raleigh/Durham", "Fuquay-Varina": "Raleigh/Durham",
    "Holly Springs NC": "Raleigh/Durham", "Knightdale": "Raleigh/Durham",
    "Zebulon": "Raleigh/Durham",

    # Charlotte
    "Charlotte": "Charlotte", "Concord NC": "Charlotte", "Concord": "Charlotte",
    "Matthews": "Charlotte", "Huntersville": "Charlotte", "Mooresville": "Charlotte",
    "Gastonia": "Charlotte", "Mint Hill": "Charlotte", "Cornelius NC": "Charlotte",
    "Harrisburg NC": "Charlotte", "Indian Trail": "Charlotte",
    "Kannapolis": "Charlotte", "Monroe NC": "Charlotte", "Stallings": "Charlotte",
    "Ballantyne": "Charlotte",

    # San Diego
    "San Diego": "San Diego", "Carlsbad": "San Diego", "Chula Vista": "San Diego",
    "Oceanside": "San Diego", "Escondido": "San Diego", "El Cajon": "San Diego",
    "La Mesa": "San Diego", "Santee": "San Diego", "Encinitas": "San Diego",
    "Alpine CA": "San Diego", "Alpine": "San Diego", "Bonita CA": "San Diego",
    "Lemon Grove": "San Diego", "National City": "San Diego", "Poway": "San Diego",
    "Ramona": "San Diego", "San Marcos CA": "San Diego", "San Marcos": "San Diego",
    "Spring Valley CA": "San Diego", "Vista CA": "San Diego", "Vista": "San Diego",

    # Los Angeles
    "Los Angeles": "Los Angeles", "Long Beach": "Los Angeles",
    "Glendale": "Los Angeles", "Burbank": "Los Angeles", "Pasadena": "Los Angeles",
    "West Hollywood": "Los Angeles", "Culver City": "Los Angeles",
    "El Segundo": "Los Angeles", "Torrance": "Los Angeles", "Hawthorne": "Los Angeles",
    "Inglewood": "Los Angeles", "Compton": "Los Angeles", "Carson": "Los Angeles",
    "Alhambra": "Los Angeles", "Arcadia CA": "Los Angeles", "Arcadia": "Los Angeles",
    "Azusa": "Los Angeles", "Baldwin Park": "Los Angeles", "Bell": "Los Angeles",
    "Bellflower": "Los Angeles", "Canoga Park": "Los Angeles", "Cerritos": "Los Angeles",
    "Chatsworth": "Los Angeles", "Covina": "Los Angeles", "Diamond Bar": "Los Angeles",
    "Downey": "Los Angeles", "Duarte": "Los Angeles", "El Monte": "Los Angeles",
    "Encino": "Los Angeles", "Gardena": "Los Angeles", "Glendora": "Los Angeles",
    "Granada Hills": "Los Angeles", "Hacienda Heights": "Los Angeles",
    "La Puente": "Los Angeles", "Lakewood CA": "Los Angeles", "Lancaster CA": "Los Angeles",
    "Lawndale": "Los Angeles", "Lomita": "Los Angeles", "Monrovia CA": "Los Angeles",
    "Montebello": "Los Angeles", "Monterey Park": "Los Angeles",
    "Northridge": "Los Angeles", "Norwalk CA": "Los Angeles", "Paramount CA": "Los Angeles",
    "Pomona": "Los Angeles", "Reseda": "Los Angeles", "Rowland Heights": "Los Angeles",
    "San Dimas": "Los Angeles", "San Gabriel": "Los Angeles", "San Pedro": "Los Angeles",
    "Santa Clarita": "Los Angeles", "Santa Monica": "Los Angeles",
    "Sherman Oaks": "Los Angeles", "Signal Hill": "Los Angeles",
    "Simi Valley": "Los Angeles", "South Gate": "Los Angeles",
    "Temple City": "Los Angeles", "Thousand Oaks": "Los Angeles",
    "Van Nuys": "Los Angeles", "West Covina": "Los Angeles", "Whittier": "Los Angeles",
    "Woodland Hills": "Los Angeles",

    # Austin
    "Austin": "Austin", "Round Rock": "Austin", "Cedar Park": "Austin",
    "Georgetown TX": "Austin", "Georgetown": "Austin", "Pflugerville": "Austin",
    "Kyle TX": "Austin", "Kyle": "Austin", "Buda": "Austin", "Leander": "Austin",
    "Manor TX": "Austin", "Lakeway": "Austin", "Bastrop": "Austin",
    "Waco": "Austin",

    # Nashville
    "Nashville": "Nashville", "Brentwood TN": "Nashville", "Brentwood": "Nashville",
    "Franklin TN": "Nashville", "Franklin": "Nashville",
    "Murfreesboro": "Nashville", "Smyrna TN": "Nashville",
    "Antioch TN": "Nashville", "Antioch": "Nashville",
    "Clarksville TN": "Nashville", "Columbia TN": "Nashville",
    "Gallatin": "Nashville", "Goodlettsville": "Nashville",
    "Hendersonville TN": "Nashville", "Hendersonville": "Nashville",
    "Hermitage": "Nashville", "Jackson TN": "Nashville",
    "LaVergne": "Nashville", "Mount Juliet": "Nashville",
    "Nolensville": "Nashville", "Old Hickory": "Nashville",
    "Spring Hill TN": "Nashville",

    # Houston
    "Houston": "Houston", "Sugar Land": "Houston", "The Woodlands": "Houston",
    "Katy": "Houston", "Spring TX": "Houston", "Spring": "Houston",
    "Pearland": "Houston", "Pasadena TX": "Houston", "Conroe": "Houston",
    "Humble": "Houston", "Missouri City": "Houston", "Cypress TX": "Houston",
    "Cypress": "Houston", "Deer Park TX": "Houston", "Friendswood": "Houston",
    "Galveston": "Houston", "La Marque": "Houston", "League City": "Houston",
    "Rosenberg TX": "Houston", "Stafford TX": "Houston", "Texas City": "Houston",
    "Webster TX": "Houston",

    # Washington, DC
    "Washington DC": "Washington, DC", "Washington, DC": "Washington, DC",
    "Arlington VA": "Washington, DC", "Arlington": "Washington, DC",
    "Alexandria VA": "Washington, DC", "Alexandria": "Washington, DC",
    "Bethesda MD": "Washington, DC", "Bethesda": "Washington, DC",
    "Silver Spring MD": "Washington, DC", "Silver Spring": "Washington, DC",
    "Rockville MD": "Washington, DC", "Rockville": "Washington, DC",
    "Gaithersburg MD": "Washington, DC", "Gaithersburg": "Washington, DC",
    "Reston VA": "Washington, DC", "Reston": "Washington, DC",
    "McLean VA": "Washington, DC", "McLean": "Washington, DC",
    "Falls Church VA": "Washington, DC", "Falls Church": "Washington, DC",
    "Fairfax VA": "Washington, DC", "Fairfax": "Washington, DC",
    "Herndon VA": "Washington, DC", "Herndon": "Washington, DC",
    "Sterling VA": "Washington, DC", "Sterling": "Washington, DC",
    "Tysons": "Washington, DC", "Centreville VA": "Washington, DC",
    "Centreville": "Washington, DC", "Chantilly": "Washington, DC",
    "Leesburg VA": "Washington, DC", "Leesburg": "Washington, DC",
    "Manassas": "Washington, DC", "Stafford VA": "Washington, DC",
    "Fredericksburg VA": "Washington, DC",

    # Dallas/Fort Worth
    "Dallas": "Dallas/Fort Worth", "Fort Worth": "Dallas/Fort Worth",
    "Plano": "Dallas/Fort Worth", "Irving": "Dallas/Fort Worth",
    "Frisco": "Dallas/Fort Worth", "McKinney": "Dallas/Fort Worth",
    "Garland": "Dallas/Fort Worth", "Mesquite": "Dallas/Fort Worth",
    "Arlington TX": "Dallas/Fort Worth", "Arlington": "Dallas/Fort Worth",
    "Carrollton": "Dallas/Fort Worth", "Richardson": "Dallas/Fort Worth",
    "Grand Prairie": "Dallas/Fort Worth", "Lewisville": "Dallas/Fort Worth",
    "Denton": "Dallas/Fort Worth", "Flower Mound": "Dallas/Fort Worth",
    "Allen TX": "Dallas/Fort Worth", "Allen": "Dallas/Fort Worth",
    "Murphy TX": "Dallas/Fort Worth", "Murphy": "Dallas/Fort Worth",
    "Addison TX": "Dallas/Fort Worth", "Addison": "Dallas/Fort Worth",
    "Balch Springs": "Dallas/Fort Worth", "Cedar Hill": "Dallas/Fort Worth",
    "Coppell": "Dallas/Fort Worth", "DeSoto TX": "Dallas/Fort Worth",
    "Duncanville": "Dallas/Fort Worth", "Euless": "Dallas/Fort Worth",
    "Farmers Branch": "Dallas/Fort Worth", "Grapevine": "Dallas/Fort Worth",
    "Haltom City": "Dallas/Fort Worth", "Lancaster TX": "Dallas/Fort Worth",
    "Little Elm": "Dallas/Fort Worth", "Mansfield TX": "Dallas/Fort Worth",
    "North Richland Hills": "Dallas/Fort Worth", "Prosper": "Dallas/Fort Worth",
    "Rowlett": "Dallas/Fort Worth", "Sachse": "Dallas/Fort Worth",
    "Sunnyvale TX": "Dallas/Fort Worth", "The Colony": "Dallas/Fort Worth",
    "Trophy Club": "Dallas/Fort Worth", "Waxahachie": "Dallas/Fort Worth",
    "Wylie": "Dallas/Fort Worth",

    # Phoenix
    "Phoenix": "Phoenix", "Scottsdale": "Phoenix", "Tempe": "Phoenix",
    "Chandler": "Phoenix", "Mesa": "Phoenix", "Gilbert AZ": "Phoenix",
    "Gilbert": "Phoenix", "Glendale AZ": "Phoenix",
    "Peoria AZ": "Phoenix", "Peoria": "Phoenix",
    "Surprise AZ": "Phoenix", "Surprise": "Phoenix",
    "Goodyear": "Phoenix", "Avondale": "Phoenix", "Buckeye": "Phoenix",
    "Queen Creek": "Phoenix", "Maricopa AZ": "Phoenix",
    "Maricopa": "Phoenix", "Tolleson": "Phoenix",

    # San Francisco-East Bay
    "Oakland": "San Francisco-East Bay", "Berkeley": "San Francisco-East Bay",
    "Fremont CA": "San Francisco-East Bay", "Fremont": "San Francisco-East Bay",
    "Hayward": "San Francisco-East Bay", "Concord CA": "San Francisco-East Bay",
    "Walnut Creek": "San Francisco-East Bay", "Pleasanton": "San Francisco-East Bay",
    "Livermore": "San Francisco-East Bay", "San Leandro": "San Francisco-East Bay",
    "Richmond CA": "San Francisco-East Bay", "Alameda": "San Francisco-East Bay",
    "Castro Valley": "San Francisco-East Bay", "Newark CA": "San Francisco-East Bay",
    "San Ramon": "San Francisco-East Bay", "Union City CA": "San Francisco-East Bay",
    "San Lorenzo": "San Francisco-East Bay",

    # San Francisco
    "San Francisco": "San Francisco", "Daly City": "San Francisco",
    "South San Francisco": "San Francisco", "Pacifica": "San Francisco",
    "San Bruno": "San Francisco",

    # San Jose
    "San Jose": "San Jose", "Santa Clara": "San Jose", "Sunnyvale": "San Jose",
    "Cupertino": "San Jose", "Mountain View": "San Jose", "Milpitas": "San Jose",
    "Campbell CA": "San Jose", "Campbell": "San Jose",
    "Los Altos": "San Jose", "Saratoga CA": "San Jose", "Saratoga": "San Jose",
    "Los Gatos": "San Jose",

    # Boston
    "Boston": "Boston", "Cambridge MA": "Boston", "Cambridge": "Boston",
    "Somerville": "Boston", "Quincy": "Boston", "Brookline": "Boston",
    "Newton": "Boston", "Waltham": "Boston", "Watertown MA": "Boston",
    "Watertown": "Boston", "Malden": "Boston", "Medford": "Boston",
    "Woburn": "Boston", "Burlington MA": "Boston", "Burlington": "Boston",
    "Lexington MA": "Boston", "Lexington": "Boston", "Needham": "Boston",
    "Acton": "Boston", "Andover": "Boston", "Bedford MA": "Boston",
    "Beverly": "Boston", "Canton MA": "Boston", "Chelmsford": "Boston",
    "Danvers": "Boston", "Dracut": "Boston", "Framingham": "Boston",
    "Franklin MA": "Boston", "Haverhill": "Boston", "Holliston": "Boston",
    "Hopkinton": "Boston", "Lowell": "Boston", "Lynn": "Boston",
    "Lynnfield": "Boston", "Marlborough": "Boston", "Methuen": "Boston",
    "Milford MA": "Boston", "Natick": "Boston", "North Andover": "Boston",
    "Norwood": "Boston", "Peabody": "Boston", "Reading MA": "Boston",
    "Reading": "Boston", "Stoughton": "Boston", "Sudbury": "Boston",
    "Tewksbury": "Boston", "Tyngsborough": "Boston", "Wakefield MA": "Boston",
    "Wakefield": "Boston", "Wellesley": "Boston", "Westborough": "Boston",
    "Westford": "Boston", "Wilmington MA": "Boston",

    # New York
    "New York": "New York", "Manhattan": "New York", "Brooklyn": "New York",
    "Queens": "New York", "Bronx": "New York", "Staten Island": "New York",
    "Hoboken": "New York", "Jersey City": "New York", "Newark": "New York",
    "Stamford": "New York", "White Plains": "New York", "Yonkers": "New York",

    # Orange County
    "Irvine": "Orange County", "Anaheim": "Orange County",
    "Santa Ana": "Orange County", "Huntington Beach": "Orange County",
    "Costa Mesa": "Orange County", "Garden Grove": "Orange County",
    "Fullerton": "Orange County", "Orange CA": "Orange County",
    "Orange": "Orange County", "Tustin": "Orange County",
    "Aliso Viejo": "Orange County", "Lake Forest CA": "Orange County",
    "Lake Forest": "Orange County", "Mission Viejo": "Orange County",
    "Newport Beach": "Orange County", "Brea": "Orange County",
    "Buena Park": "Orange County", "Cypress CA": "Orange County",
    "Dana Point": "Orange County", "Fountain Valley": "Orange County",
    "Laguna Hills": "Orange County", "Laguna Niguel": "Orange County",
    "Los Alamitos": "Orange County", "Placentia": "Orange County",
    "Rancho Santa Margarita": "Orange County", "San Clemente": "Orange County",
    "Seal Beach": "Orange County", "Stanton": "Orange County",
    "Westminster CA": "Orange County", "Yorba Linda": "Orange County",

    # Charleston
    "Charleston": "Charleston", "Mount Pleasant SC": "Charleston",
    "Mount Pleasant": "Charleston", "North Charleston": "Charleston",
    "Summerville SC": "Charleston", "Summerville": "Charleston",
    "Goose Creek": "Charleston", "Hanahan": "Charleston",
    "Johns Island": "Charleston", "Ladson": "Charleston",

    # Standalone metros
    "Jacksonville": "Jacksonville",
    "Philadelphia": "Philadelphia", "Wilmington DE": "Philadelphia",
    "Cherry Hill NJ": "Philadelphia", "King of Prussia": "Philadelphia",
    "Norristown": "Philadelphia",
    "Las Vegas": "Las Vegas", "Henderson NV": "Las Vegas",
    "North Las Vegas": "Las Vegas",
    "Portland": "Portland", "Beaverton": "Portland", "Gresham": "Portland",
    "Hillsboro": "Portland", "Lake Oswego": "Portland",
    "Milwaukie": "Portland", "Tigard": "Portland", "Tualatin": "Portland",
    "Vancouver WA": "Portland",
    "San Antonio": "San Antonio", "New Braunfels": "San Antonio",
    "Salt Lake City": "Salt Lake City", "Provo": "Salt Lake City",
    "Ogden": "Salt Lake City", "Sandy UT": "Salt Lake City",
    "Orem UT": "Salt Lake City", "West Valley City": "Salt Lake City",
    "Layton UT": "Salt Lake City",
    "Inland Empire": "Inland Empire", "Riverside CA": "Inland Empire",
    "San Bernardino": "Inland Empire", "Ontario CA": "Inland Empire",
    "Rancho Cucamonga": "Inland Empire", "Fontana": "Inland Empire",
    "Moreno Valley": "Inland Empire",
    "Memphis": "Memphis",
    "Richmond": "Richmond", "Richmond VA": "Richmond",
    "Henrico": "Richmond", "Chesterfield VA": "Richmond",
    "Savannah": "Savannah",
    "Birmingham": "Birmingham", "Hoover AL": "Birmingham",
    "Huntsville": "Huntsville", "Madison AL": "Huntsville",
    "Greenville SC": "Greenville", "Greenville": "Greenville",
    "Spartanburg": "Greenville", "Mauldin": "Greenville",
    "Simpsonville": "Greenville",
    "Gulf Shores": "Gulf Shores", "Pensacola": "Gulf Shores",
    "Mobile AL": "Gulf Shores",
    "Tallahassee": "Tallahassee",
    "Panama City Beach": "Panama City Beach", "Panama City FL": "Panama City Beach",
    "Kansas City": "Kansas City", "Overland Park": "Kansas City",
    "Lenexa": "Kansas City",
    "Charlottesville": "Charlottesville",
    "Hampton Roads": "Hampton Roads", "Norfolk": "Hampton Roads",
    "Norfolk VA": "Hampton Roads", "Virginia Beach": "Hampton Roads",
    "Chesapeake VA": "Hampton Roads", "Suffolk VA": "Hampton Roads",
    "Salinas": "Salinas/Monterey", "Monterey CA": "Salinas/Monterey",
    "Santa Cruz": "Salinas/Monterey",
    "Louisville": "Louisville",
    "Santa Barbara": "Santa Barbara", "Goleta": "Santa Barbara",
    "Ventura": "Santa Barbara", "Oxnard": "Santa Barbara",
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GS_NAVY   = "003B70"
WHITE     = "FFFFFF"
ALT_ROW   = "F5F5F5"
LT_GREEN  = "C6EFCE"
LT_RED    = "FFC7CE"
THIN_SIDE = Side(style="thin", color="CCCCCC")
THIN_BORDER = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=THIN_SIDE)

HEADER_FONT   = Font(name="Arial", bold=True, color=WHITE, size=10)
HEADER_FILL   = PatternFill("solid", fgColor=GS_NAVY)
HEADER_ALIGN  = Alignment(horizontal="center", vertical="center", wrap_text=True)
BODY_FONT     = Font(name="Arial", size=9)
BODY_ALIGN    = Alignment(horizontal="left", vertical="center")
CENTER_ALIGN  = Alignment(horizontal="center", vertical="center")
TITLE_FONT    = Font(name="Arial", bold=True, size=12, color=GS_NAVY)
SUBHEAD_FONT  = Font(name="Arial", bold=True, size=10, color=GS_NAVY)
NUM_CURRENCY  = '$#,##0'
NUM_PCT       = '0.0%'
NUM_COMMA     = '#,##0'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2: GITHUB DOWNLOAD + LOCAL CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _api_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def _load_manifest():
    manifest_path = os.path.join(CACHE_DIR, "cache_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest):
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest_path = os.path.join(CACHE_DIR, "cache_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _cached_path(filename):
    return os.path.join(CACHE_DIR, filename)


def list_github_files(path):
    """Return list of file metadata dicts from GitHub contents API."""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path}"
    resp = requests.get(url, headers=_api_headers(), timeout=30)
    if resp.status_code == 404:
        print(f"  [WARNING] Path not found in GitHub repo: {path}")
        return []
    resp.raise_for_status()
    return resp.json()


def download_file(download_url):
    """Download raw file bytes from GitHub."""
    resp = requests.get(download_url, headers=_api_headers(), timeout=60)
    resp.raise_for_status()
    return resp.content


def fetch_all_csvs():
    """
    Download all CSVs from DATA_PATH on GitHub, using cache to skip unchanged files.
    Returns list of (filename, bytes_content) tuples for individual CSV files
    (before part merging).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = _load_manifest()

    print(f"\n[Step 1] Listing files in GitHub repo: {GITHUB_OWNER}/{GITHUB_REPO}/{DATA_PATH}")
    all_items = list_github_files(DATA_PATH)
    csv_items = [f for f in all_items if isinstance(f, dict) and f.get("name", "").endswith(".csv")]
    print(f"  Found {len(csv_items)} CSV file(s)")

    results = []
    downloaded = 0
    cached_hits = 0

    for item in csv_items:
        fname = item["name"]
        sha   = item.get("sha", "")
        durl  = item.get("download_url", "")
        cache_file = _cached_path(fname)

        if manifest.get(fname) == sha and os.path.exists(cache_file):
            # Cache hit — read from disk
            with open(cache_file, "rb") as fh:
                content = fh.read()
            cached_hits += 1
        else:
            # Download fresh copy
            print(f"  Downloading: {fname}")
            content = download_file(durl)
            with open(cache_file, "wb") as fh:
                fh.write(content)
            manifest[fname] = sha
            downloaded += 1

        results.append((fname, content))

    _save_manifest(manifest)
    print(f"  Downloaded: {downloaded} file(s), cache hits: {cached_hits}")
    return results


def fetch_registry():
    """Download unit_registry.csv, with caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = _load_manifest()

    fname = "unit_registry.csv"
    # Get metadata for the registry file
    parent_path = os.path.dirname(REGISTRY_PATH)
    reg_filename = os.path.basename(REGISTRY_PATH)

    try:
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{REGISTRY_PATH}"
        resp = requests.get(url, headers=_api_headers(), timeout=30)
        if resp.status_code == 404:
            print(f"  [INFO] Registry file not found ({REGISTRY_PATH}), skipping.")
            return None
        resp.raise_for_status()
        item = resp.json()
        sha  = item.get("sha", "")
        durl = item.get("download_url", "")
        cache_file = _cached_path(fname)

        if manifest.get(fname) == sha and os.path.exists(cache_file):
            with open(cache_file, "rb") as fh:
                content = fh.read()
        else:
            print(f"  Downloading registry: {reg_filename}")
            content = download_file(durl)
            with open(cache_file, "wb") as fh:
                fh.write(content)
            manifest[fname] = sha
            _save_manifest(manifest)

        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        print(f"  [WARNING] Could not fetch registry: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FILENAME PARSING
# ─────────────────────────────────────────────────────────────────────────────

# Patterns: {ticker}_raw_{YYYY-MM-DD}.csv
#           {ticker}_raw_{YYYY-MM-DD}_part1.csv / _part2.csv
FNAME_RE      = re.compile(r'^(.+?)_raw_(\d{4}-\d{2}-\d{2})(?:_part(\d+))?\.csv$', re.IGNORECASE)


def parse_filename(fname):
    """Returns (ticker, scrape_date, part_num_or_None)."""
    m = FNAME_RE.match(fname)
    if not m:
        return None, None, None
    ticker     = m.group(1).upper()
    date_str   = m.group(2)
    part_num   = int(m.group(3)) if m.group(3) else None
    return ticker, date_str, part_num


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b: MERGE PART FILES
# ─────────────────────────────────────────────────────────────────────────────

def merge_parts(raw_files):
    """
    Detect paired _part1/_part2 files and concatenate them.
    Returns list of (base_filename, bytes_content) with parts merged.
    """
    # Group by (ticker, date) key
    groups = defaultdict(dict)
    singles = []

    for fname, content in raw_files:
        ticker, date_str, part_num = parse_filename(fname)
        if ticker is None:
            singles.append((fname, content))
            continue
        if part_num is not None:
            groups[(ticker, date_str)][part_num] = (fname, content)
        else:
            singles.append((fname, content))

    merged = list(singles)
    for (ticker, date_str), parts in groups.items():
        sorted_parts = sorted(parts.items())  # sort by part number
        combined_bytes = b"\n".join(
            content if i == 0 else _strip_header(content)
            for i, (_, (_, content)) in enumerate(sorted_parts)
        )
        base_fname = f"{ticker}_raw_{date_str}.csv"
        merged.append((base_fname, combined_bytes))
        print(f"  Merged {len(sorted_parts)} parts -> {base_fname}")

    return merged


def _strip_header(content_bytes):
    """Remove the first (header) line from CSV bytes."""
    lines = content_bytes.split(b"\n")
    return b"\n".join(lines[1:]) if len(lines) > 1 else b""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: COMBINE INTO PANEL
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_COLS = [
    "scrape_date", "reit", "market", "community", "unit_id", "beds", "sqft",
    "rent", "has_concession", "concession_hardness", "concession_raw",
    "concession_type", "concession_value", "effective_monthly_rent"
]


def build_panel(file_list):
    """
    Parse all CSVs, add reit/scrape_date columns, combine into a panel DataFrame.
    """
    dfs = []
    for fname, content in file_list:
        ticker, date_str, _ = parse_filename(fname)
        if ticker is None:
            print(f"  [SKIP] Unrecognised filename: {fname}")
            continue
        try:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        except Exception as e:
            print(f"  [ERROR] Could not parse {fname}: {e}")
            continue

        df["reit"]        = ticker
        df["scrape_date"] = date_str

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid CSV files found. Check GitHub token and DATA_PATH.")

    panel = pd.concat(dfs, ignore_index=True)

    # Ensure all expected columns exist (fill with NaN if absent)
    for col in EXPECTED_COLS:
        if col not in panel.columns:
            panel[col] = float("nan")

    # Normalise beds column
    bed_map = {"Studio": 0, "studio": 0, "STUDIO": 0}
    panel["beds"] = panel["beds"].apply(
        lambda x: bed_map.get(str(x), x) if pd.notna(x) else x
    )
    panel["beds"] = pd.to_numeric(panel["beds"], errors="coerce")

    # Deduplicate
    before = len(panel)
    panel = panel.drop_duplicates(subset=["unit_id", "scrape_date"], keep="last")
    after = len(panel)
    if before != after:
        print(f"  Deduplication: {before:,} -> {after:,} rows (removed {before - after:,} dupes)")

    # Sort
    panel["scrape_date"] = pd.to_datetime(panel["scrape_date"], errors="coerce")
    panel = panel.sort_values(["scrape_date", "reit", "unit_id"]).reset_index(drop=True)

    print(f"  Panel: {len(panel):,} rows, {panel['scrape_date'].nunique()} date(s), "
          f"{panel['reit'].nunique()} REIT(s)")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: MACRO-MARKET MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def apply_macro_map(df):
    if "market" not in df.columns:
        df["macro_market"] = "Other"
        return df

    df["macro_market"] = df["market"].map(MACRO_MAP).fillna("Other")

    # Report unmapped markets
    other_mask = df["macro_market"] == "Other"
    if other_mask.any():
        unmapped = df.loc[other_mask, "market"].value_counts()
        print(f"\n  [INFO] {len(unmapped)} market value(s) mapped to 'Other':")
        for mkt, cnt in unmapped.items():
            print(f"    {mkt!r}: {cnt:,} rows")
    else:
        print("  All market values mapped successfully.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SAME-PROPERTY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_same_property(df):
    """
    For each consecutive pair of scrape_dates, compute same-property metrics
    at REIT × macro_market × beds granularity.
    Returns sp_df or empty DataFrame.
    """
    dates = sorted(df["scrape_date"].dropna().unique())
    if len(dates) < 2:
        print("  [INFO] Only 1 scrape date — same-property analysis requires Week 2+")
        return pd.DataFrame()

    records = []
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev = df[df["scrape_date"] == prev_date]
        curr = df[df["scrape_date"] == curr_date]

        same_prop_ids = set(prev["unit_id"].dropna()) & set(curr["unit_id"].dropna())
        if not same_prop_ids:
            continue

        prev_sp = prev[prev["unit_id"].isin(same_prop_ids)].copy()
        curr_sp = curr[curr["unit_id"].isin(same_prop_ids)].copy()

        # Group by REIT × macro_market × beds
        key_cols = ["reit", "macro_market", "beds"]

        prev_grp = prev_sp.groupby(key_cols, dropna=False).agg(
            sp_avg_rent_prev=("rent", "mean"),
            sp_concession_rate_prev=("has_concession", "mean"),
            sp_count_prev=("unit_id", "count"),
        ).reset_index()

        curr_grp = curr_sp.groupby(key_cols, dropna=False).agg(
            sp_avg_rent_curr=("rent", "mean"),
            sp_concession_rate_curr=("has_concession", "mean"),
            sp_count_curr=("unit_id", "count"),
        ).reset_index()

        merged = pd.merge(prev_grp, curr_grp, on=key_cols, how="inner")
        merged["date_curr"]  = curr_date
        merged["date_prev"]  = prev_date
        merged["sp_count"]   = merged["sp_count_curr"]
        merged["sp_wow_pct"] = (
            (merged["sp_avg_rent_curr"] - merged["sp_avg_rent_prev"])
            / merged["sp_avg_rent_prev"]
        )

        records.append(merged)

    if not records:
        return pd.DataFrame()

    sp_df = pd.concat(records, ignore_index=True)
    out_cols = [
        "date_curr", "date_prev", "reit", "macro_market", "beds",
        "sp_count", "sp_avg_rent_curr", "sp_avg_rent_prev", "sp_wow_pct",
        "sp_concession_rate_curr", "sp_concession_rate_prev",
    ]
    for c in out_cols:
        if c not in sp_df.columns:
            sp_df[c] = float("nan")

    print(f"  Same-property analysis: {len(sp_df):,} rows across {len(dates) - 1} period(s)")
    return sp_df[out_cols]


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_col_widths(ws, widths):
    for col_letter, w in widths.items():
        ws.column_dimensions[col_letter].width = w


def write_header_row(ws, row_num, headers, fill=None):
    fill = fill or HEADER_FILL
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=row_num, column=col_idx, value=h)
        cell.font   = HEADER_FONT
        cell.fill   = fill
        cell.alignment = HEADER_ALIGN
        cell.border = THIN_BORDER


def write_data_row(ws, row_num, values, alt=False, number_formats=None):
    fill = PatternFill("solid", fgColor=ALT_ROW) if alt else None
    for col_idx, val in enumerate(values, start=1):
        cell = ws.cell(row=row_num, column=col_idx, value=val)
        cell.font      = BODY_FONT
        cell.alignment = BODY_ALIGN
        cell.border    = THIN_BORDER
        if fill:
            cell.fill = fill
        if number_formats and col_idx <= len(number_formats) and number_formats[col_idx - 1]:
            cell.number_format = number_formats[col_idx - 1]


def freeze_top_row(ws):
    ws.freeze_panes = "A2"


def add_title(ws, title, row=1):
    ws.cell(row=row, column=1, value=title).font = TITLE_FONT


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 1: Inputs
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs_sheet(wb, df, registry_df, latest_date_str):
    ws = wb.create_sheet("Inputs")
    ws.sheet_view.showGridLines = False

    add_title(ws, "REIT Rental Analysis — Configuration & Summary", row=1)

    ws.cell(row=3, column=1, value="Run Date").font = SUBHEAD_FONT
    ws.cell(row=3, column=2, value=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))

    ws.cell(row=4, column=1, value="Latest Scrape Date").font = SUBHEAD_FONT
    ws.cell(row=4, column=2, value=latest_date_str)

    ws.cell(row=5, column=1, value="Data Source").font = SUBHEAD_FONT
    ws.cell(row=5, column=2, value=f"github.com/{GITHUB_OWNER}/{GITHUB_REPO}/{DATA_PATH}")

    ws.cell(row=6, column=1, value="Total Rows").font = SUBHEAD_FONT
    ws.cell(row=6, column=2, value=len(df))

    ws.cell(row=7, column=1, value="REITs Covered").font = SUBHEAD_FONT
    ws.cell(row=7, column=2, value=", ".join(sorted(df["reit"].dropna().unique())))

    ws.cell(row=8, column=1, value="Scrape Dates").font = SUBHEAD_FONT
    dates_str = ", ".join(str(d)[:10] for d in sorted(df["scrape_date"].dropna().unique()))
    ws.cell(row=8, column=2, value=dates_str)

    ws.cell(row=10, column=1, value="REIT Coverage Summary").font = SUBHEAD_FONT
    headers = ["REIT", "Total Units", "Markets", "Communities",
               "Avg Rent ($)", "Concession Rate (%)"]
    write_header_row(ws, 11, headers)

    reit_grp = df.groupby("reit")
    for i, (reit, grp) in enumerate(reit_grp):
        alt = (i % 2 == 1)
        avg_rent = grp["rent"].mean()
        conc_rate = grp["has_concession"].mean() if "has_concession" in grp else float("nan")
        row_vals = [
            reit,
            len(grp),
            grp["market"].nunique() if "market" in grp else "",
            grp["community"].nunique() if "community" in grp else "",
            round(avg_rent, 0) if pd.notna(avg_rent) else "",
            round(conc_rate, 4) if pd.notna(conc_rate) else "",
        ]
        fmts = [None, NUM_COMMA, NUM_COMMA, NUM_COMMA, NUM_CURRENCY, NUM_PCT]
        write_data_row(ws, 12 + i, row_vals, alt=alt, number_formats=fmts)

    freeze_top_row(ws)
    set_col_widths(ws, {"A": 28, "B": 50, "C": 18, "D": 18, "E": 16, "F": 20})
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 2: Data (raw panel)
# ─────────────────────────────────────────────────────────────────────────────

def build_data_sheet(wb, df):
    ws = wb.create_sheet("Data")
    ws.sheet_view.showGridLines = False

    display_cols = [c for c in EXPECTED_COLS + ["macro_market"] if c in df.columns]
    write_header_row(ws, 1, display_cols)

    fmts = []
    for c in display_cols:
        if c in ("rent", "effective_monthly_rent", "sqft"):
            fmts.append(NUM_CURRENCY if "rent" in c else NUM_COMMA)
        elif c in ("has_concession",):
            fmts.append(None)
        else:
            fmts.append(None)

    for i, row in enumerate(df[display_cols].itertuples(index=False), start=2):
        alt = (i % 2 == 0)
        vals = list(row)
        # Convert Timestamp to string for Excel
        vals_clean = [
            v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else
            (None if (isinstance(v, float) and pd.isna(v)) else v)
            for v in vals
        ]
        write_data_row(ws, i, vals_clean, alt=alt, number_formats=fmts)

    freeze_top_row(ws)

    widths = {}
    for idx, col in enumerate(display_cols, start=1):
        ltr = get_column_letter(idx)
        widths[ltr] = max(12, len(col) + 2)
    set_col_widths(ws, widths)
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 3: Market_Calcs
# ─────────────────────────────────────────────────────────────────────────────

def build_market_calcs_sheet(wb, df):
    ws = wb.create_sheet("Market_Calcs")
    ws.sheet_view.showGridLines = False
    add_title(ws, "Market-Level Calculations", row=1)

    grp_cols = ["scrape_date", "macro_market", "beds"]
    agg = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            unit_count=("unit_id", "count"),
            avg_rent=("rent", "mean"),
            median_rent=("rent", "median"),
            avg_sqft=("sqft", "mean"),
            concession_rate=("has_concession", "mean"),
            avg_eff_rent=("effective_monthly_rent", "mean"),
        )
        .reset_index()
        .sort_values(["scrape_date", "macro_market", "beds"])
    )

    headers = ["Scrape Date", "Macro Market", "Beds", "Unit Count",
               "Avg Rent", "Median Rent", "Avg Sqft",
               "Concession Rate", "Avg Eff Rent"]
    write_header_row(ws, 2, headers)
    fmts = [None, None, None, NUM_COMMA,
            NUM_CURRENCY, NUM_CURRENCY, NUM_COMMA, NUM_PCT, NUM_CURRENCY]

    for i, row in enumerate(agg.itertuples(index=False), start=3):
        alt = (i % 2 == 0)
        vals = [
            str(row.scrape_date)[:10] if pd.notna(row.scrape_date) else "",
            row.macro_market,
            int(row.beds) if pd.notna(row.beds) else "N/A",
            row.unit_count,
            round(row.avg_rent, 0) if pd.notna(row.avg_rent) else None,
            round(row.median_rent, 0) if pd.notna(row.median_rent) else None,
            round(row.avg_sqft, 0) if pd.notna(row.avg_sqft) else None,
            round(row.concession_rate, 4) if pd.notna(row.concession_rate) else None,
            round(row.avg_eff_rent, 0) if pd.notna(row.avg_eff_rent) else None,
        ]
        write_data_row(ws, i, vals, alt=alt, number_formats=fmts)

    freeze_top_row(ws)
    set_col_widths(ws, {"A": 14, "B": 26, "C": 8, "D": 12,
                        "E": 12, "F": 14, "G": 10, "H": 18, "I": 14})
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 4: REIT_Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_reit_summary_sheet(wb, df):
    ws = wb.create_sheet("REIT_Summary")
    ws.sheet_view.showGridLines = False
    add_title(ws, "REIT-Level Summary (Latest Scrape Date)", row=1)

    latest = df["scrape_date"].max()
    df_latest = df[df["scrape_date"] == latest].copy()

    grp = (
        df_latest.groupby(["reit", "beds"], dropna=False)
        .agg(
            unit_count=("unit_id", "count"),
            avg_rent=("rent", "mean"),
            median_rent=("rent", "median"),
            avg_sqft=("sqft", "mean"),
            concession_rate=("has_concession", "mean"),
            avg_eff_rent=("effective_monthly_rent", "mean"),
            markets=("macro_market", "nunique"),
        )
        .reset_index()
        .sort_values(["reit", "beds"])
    )

    headers = ["REIT", "Beds", "Unit Count", "Avg Rent", "Median Rent",
               "Avg Sqft", "Concession Rate", "Avg Eff Rent", "# Markets"]
    write_header_row(ws, 2, headers)
    fmts = [None, None, NUM_COMMA, NUM_CURRENCY, NUM_CURRENCY,
            NUM_COMMA, NUM_PCT, NUM_CURRENCY, None]

    for i, row in enumerate(grp.itertuples(index=False), start=3):
        alt = (i % 2 == 0)
        vals = [
            row.reit,
            int(row.beds) if pd.notna(row.beds) else "N/A",
            row.unit_count,
            round(row.avg_rent, 0) if pd.notna(row.avg_rent) else None,
            round(row.median_rent, 0) if pd.notna(row.median_rent) else None,
            round(row.avg_sqft, 0) if pd.notna(row.avg_sqft) else None,
            round(row.concession_rate, 4) if pd.notna(row.concession_rate) else None,
            round(row.avg_eff_rent, 0) if pd.notna(row.avg_eff_rent) else None,
            row.markets,
        ]
        write_data_row(ws, i, vals, alt=alt, number_formats=fmts)

    freeze_top_row(ws)
    set_col_widths(ws, {"A": 12, "B": 8, "C": 12, "D": 12, "E": 14,
                        "F": 10, "G": 18, "H": 14, "I": 12})
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 5: Macro_Market_Detail
# ─────────────────────────────────────────────────────────────────────────────

def build_macro_market_detail_sheet(wb, df):
    ws = wb.create_sheet("Macro_Market_Detail")
    ws.sheet_view.showGridLines = False
    add_title(ws, "Macro Market Detail — REIT × Market × Beds", row=1)

    latest = df["scrape_date"].max()
    df_latest = df[df["scrape_date"] == latest].copy()

    grp = (
        df_latest.groupby(["macro_market", "reit", "beds"], dropna=False)
        .agg(
            unit_count=("unit_id", "count"),
            avg_rent=("rent", "mean"),
            median_rent=("rent", "median"),
            concession_rate=("has_concession", "mean"),
            avg_eff_rent=("effective_monthly_rent", "mean"),
        )
        .reset_index()
        .sort_values(["macro_market", "reit", "beds"])
    )

    headers = ["Macro Market", "REIT", "Beds", "Unit Count",
               "Avg Rent", "Median Rent", "Concession Rate", "Avg Eff Rent"]
    write_header_row(ws, 2, headers)
    fmts = [None, None, None, NUM_COMMA,
            NUM_CURRENCY, NUM_CURRENCY, NUM_PCT, NUM_CURRENCY]

    for i, row in enumerate(grp.itertuples(index=False), start=3):
        alt = (i % 2 == 0)
        vals = [
            row.macro_market,
            row.reit,
            int(row.beds) if pd.notna(row.beds) else "N/A",
            row.unit_count,
            round(row.avg_rent, 0) if pd.notna(row.avg_rent) else None,
            round(row.median_rent, 0) if pd.notna(row.median_rent) else None,
            round(row.concession_rate, 4) if pd.notna(row.concession_rate) else None,
            round(row.avg_eff_rent, 0) if pd.notna(row.avg_eff_rent) else None,
        ]
        write_data_row(ws, i, vals, alt=alt, number_formats=fmts)

    freeze_top_row(ws)
    set_col_widths(ws, {"A": 26, "B": 12, "C": 8, "D": 12,
                        "E": 12, "F": 14, "G": 18, "H": 14})
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 6: Charts_Rent
# ─────────────────────────────────────────────────────────────────────────────

def build_charts_rent_sheet(wb, df):
    ws = wb.create_sheet("Charts_Rent")
    ws.sheet_view.showGridLines = False
    add_title(ws, "Average Rent by REIT and Bed Type", row=1)

    # Build a pivot: rows = beds, cols = reit, values = avg_rent (latest date)
    latest = df["scrape_date"].max()
    df_latest = df[df["scrape_date"] == latest].copy()

    pivot = df_latest.groupby(["reit", "beds"])["rent"].mean().unstack("beds")
    pivot = pivot.round(0)

    reits = list(pivot.index)
    beds_vals = sorted([c for c in pivot.columns if pd.notna(c)])

    # Write table starting at row 3
    hdr = ["REIT"] + [f"{int(b)}BR" if b != 0 else "Studio" for b in beds_vals]
    write_header_row(ws, 3, hdr)

    for i, reit in enumerate(reits):
        vals = [reit] + [
            pivot.loc[reit, b] if b in pivot.columns and pd.notna(pivot.loc[reit, b]) else None
            for b in beds_vals
        ]
        fmts = [None] + [NUM_CURRENCY] * len(beds_vals)
        write_data_row(ws, 4 + i, vals, alt=(i % 2 == 1), number_formats=fmts)

    # Bar chart
    if reits and beds_vals:
        chart = BarChart()
        chart.type    = "col"
        chart.grouping = "clustered"
        chart.title   = f"Avg Rent by REIT & Bed Type ({str(latest)[:10]})"
        chart.y_axis.title = "Avg Monthly Rent ($)"
        chart.x_axis.title = "REIT"
        chart.style  = 10
        chart.width  = 24
        chart.height = 14

        data_ref = Reference(ws, min_col=2, max_col=1 + len(beds_vals),
                             min_row=3, max_row=3 + len(reits))
        cats_ref = Reference(ws, min_col=1, min_row=4, max_row=3 + len(reits))
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        ws.add_chart(chart, f"A{5 + len(reits)}")

    freeze_top_row(ws)
    set_col_widths(ws, {get_column_letter(i): 14 for i in range(1, 2 + len(beds_vals))})
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 7: Charts_Concessions
# ─────────────────────────────────────────────────────────────────────────────

def build_charts_concessions_sheet(wb, df):
    ws = wb.create_sheet("Charts_Concessions")
    ws.sheet_view.showGridLines = False
    add_title(ws, "Concession Rate by REIT and Date", row=1)

    grp = (
        df.groupby(["scrape_date", "reit"])["has_concession"]
        .mean()
        .reset_index()
        .rename(columns={"has_concession": "concession_rate"})
    )
    grp["scrape_date"] = grp["scrape_date"].apply(lambda x: str(x)[:10])

    dates_sorted = sorted(grp["scrape_date"].unique())
    reits_sorted = sorted(grp["reit"].unique())

    pivot = grp.pivot(index="scrape_date", columns="reit", values="concession_rate")
    pivot = pivot.round(4)

    # Write table
    hdr = ["Date"] + reits_sorted
    write_header_row(ws, 3, hdr)

    for i, dt in enumerate(dates_sorted):
        row_vals = [dt] + [
            pivot.loc[dt, r] if r in pivot.columns and dt in pivot.index and pd.notna(pivot.loc[dt, r]) else None
            for r in reits_sorted
        ]
        fmts = [None] + [NUM_PCT] * len(reits_sorted)
        write_data_row(ws, 4 + i, row_vals, alt=(i % 2 == 1), number_formats=fmts)

    # Line chart
    if dates_sorted and reits_sorted:
        chart = LineChart()
        chart.title   = "Concession Rate Over Time by REIT"
        chart.y_axis.title = "Concession Rate"
        chart.x_axis.title = "Scrape Date"
        chart.style  = 10
        chart.width  = 24
        chart.height = 14

        data_ref = Reference(ws, min_col=2, max_col=1 + len(reits_sorted),
                             min_row=3, max_row=3 + len(dates_sorted))
        cats_ref = Reference(ws, min_col=1, min_row=4, max_row=3 + len(dates_sorted))
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        ws.add_chart(chart, f"A{5 + len(dates_sorted)}")

    freeze_top_row(ws)
    set_col_widths(ws, {get_column_letter(i): 14 for i in range(1, 2 + len(reits_sorted))})
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 8: Same_Prop_Trends
# ─────────────────────────────────────────────────────────────────────────────

def build_same_prop_sheet(wb, df, sp_df):
    ws = wb.create_sheet("Same_Prop_Trends")
    ws.sheet_view.showGridLines = False

    dates = sorted(df["scrape_date"].dropna().unique())

    if len(dates) < 2 or sp_df.empty:
        add_title(ws, "Same-Property Trends", row=1)
        ws.cell(row=3, column=1,
                value="Insufficient data — run again after Week 2 scrape").font = Font(
            name="Arial", italic=True, color="888888", size=10)
        return ws

    add_title(ws, "Same-Property WoW Rent Change — REIT × Macro Market × Beds", row=1)

    # Latest period only for the table
    latest_curr = sp_df["date_curr"].max()
    sp_latest = sp_df[sp_df["date_curr"] == latest_curr].copy()

    headers = ["REIT", "Macro Market", "Beds", "Same-Prop Count",
               "Avg Rent (Curr)", "Avg Rent (Prev)", "WoW Chg (%)",
               "Concession Rate (Curr)", "Concession Rate (Prev)",
               "Period"]
    write_header_row(ws, 3, headers)

    green_fill = PatternFill("solid", fgColor=LT_GREEN)
    red_fill   = PatternFill("solid", fgColor=LT_RED)

    fmts = [None, None, None, NUM_COMMA,
            NUM_CURRENCY, NUM_CURRENCY, NUM_PCT,
            NUM_PCT, NUM_PCT, None]

    data_rows = sorted(
        sp_latest.itertuples(index=False),
        key=lambda r: (str(r.reit), str(r.macro_market), r.beds if pd.notna(r.beds) else 99)
    )

    for i, row in enumerate(data_rows):
        excel_row = 4 + i
        alt = (i % 2 == 1)
        beds_disp = int(row.beds) if pd.notna(row.beds) else "N/A"
        period_str = f"{str(row.date_prev)[:10]} -> {str(row.date_curr)[:10]}"
        vals = [
            row.reit,
            row.macro_market,
            beds_disp,
            int(row.sp_count) if pd.notna(row.sp_count) else None,
            round(row.sp_avg_rent_curr, 0) if pd.notna(row.sp_avg_rent_curr) else None,
            round(row.sp_avg_rent_prev, 0) if pd.notna(row.sp_avg_rent_prev) else None,
            round(row.sp_wow_pct, 4) if pd.notna(row.sp_wow_pct) else None,
            round(row.sp_concession_rate_curr, 4) if pd.notna(row.sp_concession_rate_curr) else None,
            round(row.sp_concession_rate_prev, 4) if pd.notna(row.sp_concession_rate_prev) else None,
            period_str,
        ]
        write_data_row(ws, excel_row, vals, alt=alt, number_formats=fmts)

        # Highlight WoW column (col 7)
        wow_val = row.sp_wow_pct
        if pd.notna(wow_val):
            cell = ws.cell(row=excel_row, column=7)
            if wow_val > 0.005:
                cell.fill = green_fill
            elif wow_val < -0.005:
                cell.fill = red_fill

    n_data = len(data_rows)

    # ── Line chart: rent index (base=100) per REIT over time ──
    # Build index table below the main table
    chart_start_row = 4 + n_data + 3
    ws.cell(row=chart_start_row - 1, column=1,
            value="Same-Property Avg Rent Index (Base=100, per REIT)").font = SUBHEAD_FONT

    reits = sorted(sp_df["reit"].dropna().unique())

    # For each REIT, compute avg rent per date across all markets/beds
    reit_date_rent = (
        sp_df.groupby(["reit", "date_curr"])["sp_avg_rent_curr"]
        .mean()
        .reset_index()
    )

    # Pivot: rows = dates, cols = reits
    all_dates_for_chart = sorted(set(
        list(sp_df["date_curr"].unique()) + [sp_df["date_prev"].min()]
    ))

    # Build a wider series: include the very first prev date
    first_prev = sp_df["date_prev"].min()
    reit_date_rent_base = (
        sp_df[sp_df["date_prev"] == first_prev]
        .groupby("reit")["sp_avg_rent_prev"]
        .mean()
        .reset_index()
        .rename(columns={"sp_avg_rent_prev": "sp_avg_rent_curr"})
    )
    reit_date_rent_base["date_curr"] = first_prev
    combined_rent = pd.concat([reit_date_rent_base, reit_date_rent], ignore_index=True)
    combined_rent = combined_rent.sort_values("date_curr")

    pivot_chart = combined_rent.pivot(index="date_curr", columns="reit", values="sp_avg_rent_curr")

    # Normalise to base 100
    for reit in pivot_chart.columns:
        first_val = pivot_chart[reit].dropna().iloc[0] if not pivot_chart[reit].dropna().empty else None
        if first_val and first_val != 0:
            pivot_chart[reit] = (pivot_chart[reit] / first_val * 100).round(2)

    chart_hdr = ["Date"] + list(pivot_chart.columns)
    write_header_row(ws, chart_start_row, chart_hdr)

    chart_dates = [str(d)[:10] for d in pivot_chart.index]
    for i, (dt, row_series) in enumerate(pivot_chart.iterrows()):
        row_vals = [str(dt)[:10]] + [
            float(row_series[r]) if pd.notna(row_series.get(r, float("nan"))) else None
            for r in pivot_chart.columns
        ]
        write_data_row(ws, chart_start_row + 1 + i, row_vals, alt=(i % 2 == 1))

    n_chart_rows = len(pivot_chart)

    if n_chart_rows >= 2 and reits:
        line_chart = LineChart()
        line_chart.title    = "Same-Property Avg Rent Index (Base = 100)"
        line_chart.y_axis.title = "Rent Index"
        line_chart.x_axis.title = "Scrape Date"
        line_chart.style   = 10
        line_chart.width   = 24
        line_chart.height  = 14

        data_ref = Reference(
            ws,
            min_col=2, max_col=1 + len(reits),
            min_row=chart_start_row,
            max_row=chart_start_row + n_chart_rows,
        )
        cats_ref = Reference(
            ws, min_col=1,
            min_row=chart_start_row + 1,
            max_row=chart_start_row + n_chart_rows,
        )
        line_chart.add_data(data_ref, titles_from_data=True)
        line_chart.set_categories(cats_ref)
        ws.add_chart(line_chart, f"A{chart_start_row + n_chart_rows + 2}")

    freeze_top_row(ws)
    set_col_widths(ws, {
        "A": 12, "B": 26, "C": 8, "D": 14,
        "E": 16, "F": 16, "G": 12, "H": 22, "I": 22, "J": 28,
    })
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  REIT Rental Analysis — Excel Workbook Builder")
    print("=" * 60)

    if not GITHUB_TOKEN:
        print("\n[WARNING] GITHUB_TOKEN is empty. Requests will be unauthenticated.")
        print("  Set GITHUB_TOKEN at the top of this script for private repos.\n")

    # Step 1+2: Download from GitHub with caching
    raw_files = fetch_all_csvs()

    if not raw_files:
        print("[ERROR] No files downloaded. Check token, owner, repo, and DATA_PATH.")
        sys.exit(1)

    # Step 2b: Merge parts
    print("\n[Step 2b] Merging part files...")
    merged_files = merge_parts(raw_files)

    # Step 3: Build panel
    print("\n[Step 3] Building panel dataset...")
    df = build_panel(merged_files)

    # Also download registry (informational — not required for core analysis)
    print("\n[Step 1b] Fetching unit registry...")
    registry_df = fetch_registry()

    # Step 4: Macro-market mapping
    print("\n[Step 4] Applying macro-market mapping...")
    df = apply_macro_map(df)

    # Step 5: Same-property analysis
    print("\n[Step 5] Computing same-property metrics...")
    sp_df = compute_same_property(df)

    # Step 6: Build Excel
    print("\n[Step 6] Building Excel workbook...")
    wb = Workbook()
    # Remove default sheet
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    latest_date = df["scrape_date"].max()
    latest_date_str = str(latest_date)[:10]

    print("  Building: Inputs")
    build_inputs_sheet(wb, df, registry_df, latest_date_str)
    print("  Building: Data")
    build_data_sheet(wb, df)
    print("  Building: Market_Calcs")
    build_market_calcs_sheet(wb, df)
    print("  Building: REIT_Summary")
    build_reit_summary_sheet(wb, df)
    print("  Building: Macro_Market_Detail")
    build_macro_market_detail_sheet(wb, df)
    print("  Building: Charts_Rent")
    build_charts_rent_sheet(wb, df)
    print("  Building: Charts_Concessions")
    build_charts_concessions_sheet(wb, df)
    print("  Building: Same_Prop_Trends")
    build_same_prop_sheet(wb, df, sp_df)

    # Step 7: Save
    out_filename = f"REIT_Rental_Analysis_{latest_date_str}.xlsx"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    out_path = os.path.normpath(out_path)
    wb.save(out_path)

    file_size_kb = os.path.getsize(out_path) / 1024
    print(f"\n{'=' * 60}")
    print(f"  Output saved: {out_path}")
    print(f"  File size:    {file_size_kb:.1f} KB")
    print(f"  Sheets:       {', '.join(wb.sheetnames)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
