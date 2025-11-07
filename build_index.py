import pandas as pd
import sqlite3
import time
import os
import glob 

# --- 1. CONFIGURATION ---
DATASET_FOLDER = "amazondb"
NAME_COLUMN = "name"
PRICE_COLUMN = "actual_price"
CATEGORY_COLUMN = "main_category"
SUBCATEGORY_COLUMN = "sub_category"
SYNTHETIC_ID_COLUMN = "product_id"
SYNTHETIC_DESC_COLUMN = "synthetic_description"
DB_FILE = "products.db"

print("Starting data preparation script...")
print(f"This will create an SQLite database at: {DB_FILE}")

# --- 2. LOAD DATA (FROM FOLDER) ---
start_time = time.time()
path_to_search = os.path.join(DATASET_FOLDER, "*.csv")
all_csv_files = glob.glob(path_to_search)

if not all_csv_files:
    print(f"ERROR: No CSV files found in '{DATASET_FOLDER}'.")
    exit()

print(f"Found {len(all_csv_files)} CSV files. Loading and combining...")
all_dfs = []
for csv_file in all_csv_files:
    try:
        temp_df = pd.read_csv(csv_file, encoding='latin1')
        all_dfs.append(temp_df)
    except Exception as e:
        print(f"  Warning: Could not load {csv_file}. Skipping. Error: {e}")

if not all_dfs:
    print("ERROR: No data was loaded. All files failed to load.")
    exit()
    
df = pd.concat(all_dfs, ignore_index=True)
print(f"\nSuccessfully loaded and combined all files. Total rows: {len(df)}")
    
# --- 3. CLEAN DATA ---
print("Cleaning data...")
required_cols = [NAME_COLUMN, PRICE_COLUMN, CATEGORY_COLUMN, SUBCATEGORY_COLUMN]
if not all(col in df.columns for col in required_cols):
    print(f"ERROR: Your CSVs are missing one of these columns: {required_cols}")
    exit()

print("Cleaning price column (aggressive)...")
df[PRICE_COLUMN] = df[PRICE_COLUMN].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors='coerce').fillna(0)

print("Creating synthetic product IDs...")
df = df.reset_index(drop=True)
df[SYNTHETIC_ID_COLUMN] = df.index # <-- We create the 'product_id' column

# --- !! THIS LINE IS REMOVED !! ---
# df = df.set_index(SYNTHETIC_ID_COLUMN) 
# --- END OF FIX ---

print("Creating synthetic descriptions...")
df[NAME_COLUMN] = df[NAME_COLUMN].fillna("")
df[CATEGORY_COLUMN] = df[CATEGORY_COLUMN].fillna("")
df[SUBCATEGORY_COLUMN] = df[SUBCATEGORY_COLUMN].fillna("")
df[SYNTHETIC_DESC_COLUMN] = df[NAME_COLUMN] + " " + df[CATEGORY_COLUMN] + " " + df[SUBCATEGORY_COLUMN]

df = df.dropna(subset=[SYNTHETIC_DESC_COLUMN])
print(f"Cleaned data. {len(df)} unique products remaining.")

# --- 4. PREPARE DATAFRAME FOR SQL ---
# 'product_id' is now a regular column, so we can select it.
df_to_save = df[[
    SYNTHETIC_ID_COLUMN,  # <-- This will now work
    NAME_COLUMN, 
    PRICE_COLUMN, 
    SYNTHETIC_DESC_COLUMN
]]
df_to_save.columns = ['product_id', 'name', 'price', 'synthetic_description']
# We no longer need to reset the index here.

# --- 5. CREATE AND POPULATE SQLITE DATABASE ---
print(f"Connecting to database '{DB_FILE}'...")
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

print("Dropping old tables (if they exist)...")
c.execute('DROP TABLE IF EXISTS products')
c.execute('DROP TABLE IF EXISTS products_fts')

print("Creating 'products' table...")
# product_id is now a regular column, so we define it as the PRIMARY KEY
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
print(f"Created SQLite database: '{DB_FILE}'")
print("We are now ready to build main.py!")