
import sqlite3
import re
import os

def import_quran_sql_to_sqlite(sql_path: str, db_path: str):
    """
    Parses the provided Quran SQL file and imports it into a SQLite database.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create valid SQLite table
    create_table_query = """
    CREATE TABLE quran (
        id INTEGER PRIMARY KEY,
        jozz INTEGER,
        page INTEGER,
        sura_no INTEGER,
        sura_name_en TEXT,
        sura_name_ar TEXT,
        line_start INTEGER,
        line_end INTEGER,
        aya_no INTEGER,
        aya_text TEXT,
        aya_text_emlaey TEXT
    );
    """
    cursor.execute(create_table_query)
    
    # Read the SQL file
    print(f"Reading SQL file from: {sql_path}")
    with open(sql_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()

    # The file contains INSERT INTO `` ... which is invalid for SQLite or generic SQL if table name is empty.
    # We replace `INSERT INTO `` (` with `INSERT INTO quran (`
    
    # Regex to find the insert statements
    # Pattern: INSERT INTO ``
    # Replacement: INSERT INTO quran
    
    # Also, standard SQL uses single quotes for strings. The file seems to use single quotes.
    # We need to make sure we treat it line by line or split by statement.
    
    # Let's split by ';' to get statements, but be careful about ';' inside strings.
    # Ideally, we just iterate line by line if each insert is on a new line (which seems to be the case from the prev view_file).
    
    count = 0
    
    # Split by lines and process
    for line in sql_content.splitlines():
        line = line.strip()
        if not line or line.startswith("/*") or line.startswith("--"):
            continue
            
        if line.startswith("INSERT INTO ``"):
            # Fix table name
            fixed_line = line.replace("INSERT INTO ``", "INSERT INTO quran")
            # Replace the weird symbol at the end of ayahs if it causes issues? 
            # Looking at view_file result: '...ﰀ'
            # Those are end of ayah symbols, we should keep them if possible, or we can clean them later.
            
            try:
                cursor.execute(fixed_line)
                count += 1
            except sqlite3.Error as e:
                print(f"Error executing line: {line[:50]}... -> {e}")

    conn.commit()
    print(f"Imported {count} ayahs into {db_path}")
    conn.close()

if __name__ == "__main__":
    # For standalone testing
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sql_file = os.path.join(base_dir, "data", "quran", "hafsData_v2-0.sql")
    db_file = os.path.join(base_dir, "data", "quran.db")
    import_quran_sql_to_sqlite(sql_file, db_file)
