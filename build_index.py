import pandas as pd
import sqlite3
import time
import os
import glob 

# --- 1. CONFIGURATION ---
# !! UPDATE THESE PATHS if your filenames are different !!
PRODUCTS_CSV_PATH = "amazondb/products.csv"
CATEGORIES_CSV_PATH = "amazondb/categories.csv"
DB_FILE = "products.db"

# --- !! Columns from your TERMINAL OUTPUT !! ---
# Products file
PROD_ID_COL = "asin"        # <-- Changed from 'product_id'
PROD_NAME_COL = "title"     # <-- Changed from 'product_name'
CAT_ID_COL_LEFT = "category_id" # <-- This is the key in the 'products' file
PRICE_COL = "price"         # <-- Changed from 'actual_price'
DESC_COL = "title"      # <-- Using 'title' as 'about_product' is missing

# Categories file
CAT_ID_COL_RIGHT = "id"     # <-- This is the KEY FIX. Was 'category_id'
CAT_NAME_COL = "category_name"

# --- Columns we will create ---
SYNTHETIC_ID_COL = "product_id_db"
SYNTHETIC_DESC_COL = "synthetic_description"

print("Starting data preparation script...")
print(f"This will create an SQLite database at: {DB_FILE}")

# --- 2. LOAD DATA (FROM 2 FILES) ---
start_time = time.time()
try:
    print(f"Loading Products CSV: {PRODUCTS_CSV_PATH}")
    df_products = pd.read_csv(PRODUCTS_CSV_PATH, encoding='latin1')
    print(f"Loading Categories CSV: {CATEGORIES_CSV_PATH}")
    df_categories = pd.read_csv(CATEGORIES_CSV_PATH, encoding='latin1')
except FileNotFoundError as e:
    print(f"ERROR: File not found. Make sure paths are correct in the script.")
    print(e)
    exit()
except Exception as e:
    print(f"Error loading CSVs: {e}")
    exit()

print(f"Loaded {len(df_products)} products and {len(df_categories)} categories.")

# --- Clean column names (just in case) ---
df_products.columns = df_products.columns.str.strip()
df_categories.columns = df_categories.columns.str.strip()

print(f"Products columns: {df_products.columns.tolist()}")
print(f"Categories columns: {df_categories.columns.tolist()}")


# --- 3. MERGE & CLEAN DATA ---
print("Merging products and categories...")
try:
    # --- !! THIS IS THE MERGE FIX !! ---
    # We merge using the two different key names
    df = pd.merge(
        df_products, 
        df_categories, 
        left_on=CAT_ID_COL_LEFT,  # 'category_id' from products.csv
        right_on=CAT_ID_COL_RIGHT, # 'id' from categories.csv
        how="left"
    )
    # --- END OF FIX ---
except KeyError:
    print(f"ERROR: Merge failed. Check the CONFIGURATION section.")
    exit()

print(f"Merged data. Total rows: {len(df)}")

print("Cleaning data...")
# Drop duplicates based on the product ID ('asin')
df = df.drop_duplicates(subset=[PROD_ID_COL])

print("Cleaning price column (aggressive)...")
# Check if price column exists before cleaning
if PRICE_COL not in df.columns:
    print(f"Warning: Price column '{PRICE_COL}' not found. Defaulting prices to 0.")
    df[PRICE_COL] = 0.0
else:
    df[PRICE_COL] = df[PRICE_COL].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors='coerce').fillna(0)


print("Creating synthetic product IDs...")
df = df.reset_index(drop=True)
df[SYNTHETIC_ID_COL] = df.index

print("Creating synthetic descriptions...")
# Fill any missing text with empty strings
df[PROD_NAME_COL] = df[PROD_NAME_COL].fillna("")
df[CAT_NAME_COL] = df[CAT_NAME_COL].fillna("")
# Use DESC_COL ('title') as 'about_product' is missing
df[DESC_COL] = df[DESC_COL].fillna("") 

# Combine all text fields into one powerful search column
df[SYNTHETIC_DESC_COL] = df[PROD_NAME_COL] + " " + df[CAT_NAME_COL] + " " + df[DESC_COL]

df = df.dropna(subset=[SYNTHETIC_DESC_COL])
print(f"Cleaned data. {len(df)} unique products remaining.")

# --- 4. PREPARE DATAFRAME FOR SQL ---
# Select the columns we want to save
df_to_save = df[[
    SYNTHETIC_ID_COL, 
    PROD_NAME_COL, 
    PRICE_COL, 
    SYNTHETIC_DESC_COL
]]
# Rename them to the simple names our main.py expects
df_to_save.columns = ['product_id', 'name', 'price', 'synthetic_description']

# --- 5. CREATE AND POPULATE SQLITE DATABASE ---
print(f"Connecting to database '{DB_FILE}'...")
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

print("Dropping old tables (if they exist)...")
c.execute('DROP TABLE IF EXISTS products')
c.execute('DROP TABLE IF EXISTS products_fts')

print("Creating 'products' table...")
c.execute('''
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    price REAL,
    synthetic_description TEXT
)
''')

print("Creating 'products_fts' search index...")
c.execute('''
CREATE VIRTUAL TABLE products_fts USING fts5(
    product_id UNINDEXED,
    synthetic_description,
    content='products',
    content_rowid='product_id'
)
''')

print(f"Populating 'products' table with {len(df_to_save)} items... (This may take a minute)")
df_to_save.to_sql('products', conn, if_exists='append', index=False)

print("Populating 'products_fts' search index...")
c.execute('''
INSERT INTO products_fts (rowid, synthetic_description)
SELECT product_id, synthetic_description FROM products
''')

print("Committing changes and closing database...")
conn.commit()
conn.close()

end_time = time.time()
print("\n--- SCRIPT COMPLETE ---")
print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print(f"Created new SQLite database: '{DB_FILE}' with merged category data.")
print("You are now ready to run main.py!")