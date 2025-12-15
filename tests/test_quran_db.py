
import pytest
import os
from src.quran_db.core import QuranDB

def test_db_connection():
    db = QuranDB()
    assert os.path.exists(db.db_path)

def test_get_fatiha_first_ayah():
    db = QuranDB()
    ayah = db.get_ayah(1, 1)
    assert ayah is not None
    assert ayah.sura_no == 1
    assert ayah.aya_no == 1
    assert "بِسۡمِ ٱللَّهِ" in ayah.aya_text
    assert ayah.sura_name_en == "Al-Fātiḥah"

def test_get_ayah_text_only():
    db = QuranDB()
    text = db.get_ayah_text(1, 1)
    assert text is not None
    assert "بِسۡمِ ٱللَّهِ" in text

def test_get_non_existent_ayah():
    db = QuranDB()
    ayah = db.get_ayah(115, 1) # Only 114 surahs
    assert ayah is None

def test_search_text():
    db = QuranDB()
    results = db.search_text("الحمد لله")
    assert len(results) > 0
    found = False
    for ayah in results:
        if ayah.sura_no == 1 and ayah.aya_no == 2:
            found = True
            break
    assert found
