
import sqlite3
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Ayah:
    id: int
    jozz: int
    page: int
    sura_no: int
    sura_name_en: str
    sura_name_ar: str
    line_start: int
    line_end: int
    aya_no: int
    aya_text: str
    aya_text_emlaey: str

class QuranDB:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to data/quran.db relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base_dir, "data", "quran.db")
        
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Quran database not found at {self.db_path}. Please run importer.py first.")

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_ayah(self, sura_no: int, aya_no: int) -> Optional[Ayah]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM quran WHERE sura_no = ? AND aya_no = ?", 
            (sura_no, aya_no)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Ayah(**dict(row))
        return None

    def get_ayah_text(self, sura_no: int, aya_no: int) -> Optional[str]:
        ayah = self.get_ayah(sura_no, aya_no)
        return ayah.aya_text if ayah else None

    def search_text(self, text: str) -> List[Ayah]:
        """Search for text in Emlaey script (easier for search)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM quran WHERE aya_text_emlaey LIKE ?", 
            (f"%{text}%",)
        )
        rows = cursor.fetchall()
        conn.close()
        return [Ayah(**dict(row)) for row in rows]
