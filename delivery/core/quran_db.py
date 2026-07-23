"""
Minimal Quran database access for the delivery service.
Provides ayah-range lookup for the async /api/evaluate endpoint.
"""
import os
import re
import sqlite3

# End-of-ayah ornament markers (Arabic Presentation Forms) that appear in the
# uthmani `aya_text` but are NOT part of the recited words. They must be stripped
# so the grader doesn't count them as mistakes. (Integration spec §6.3)
_MARKER_RE = re.compile(r'[ﰀ-﷿]')
_WS_RE = re.compile(r'\s+')


class QuranDB:
    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "data", "quran.db")
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Quran database not found at {self.db_path}")

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_ayah_range(self, surah_num: int, from_ayah: int, to_ayah: int) -> str:
        """Return the concatenated text of ayahs [from_ayah, to_ayah], marker-free."""
        if to_ayah < from_ayah:
            from_ayah, to_ayah = to_ayah, from_ayah
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT aya_text FROM quran
                   WHERE sura_no = ? AND aya_no BETWEEN ? AND ?
                   ORDER BY aya_no""",
                (surah_num, from_ayah, to_ayah),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            raise ValueError(
                f"No ayahs found: surah={surah_num} range={from_ayah}-{to_ayah}"
            )

        text = " ".join(row["aya_text"] for row in rows)
        text = _MARKER_RE.sub("", text)          # drop end-of-ayah ornaments
        text = _WS_RE.sub(" ", text).strip()     # collapse the gaps they leave
        return text
