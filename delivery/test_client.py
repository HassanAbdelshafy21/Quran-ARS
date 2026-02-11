"""
Demo Client — Tests the Quran ASR API and displays results.
Usage: python test_client.py [--url URL] [--file PATH] [--text TEXT]
"""
import requests
import json
import sys
import argparse
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def test_api(api_url, file_path, target_text, surah_num=None, ayah_num=None):
    print("=" * 60)
    print("  Quran ASR Kids — Demo Client")
    print("=" * 60)
    print(f"  API:  {api_url}")
    print(f"  File: {file_path}")
    print(f"  Text: {target_text[:60]}...")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"\n  Audio file not found: {file_path}")
        return
    
    with open(file_path, 'rb') as f:
        data = {"target_ayah": target_text}
        if surah_num: data["surah_num"] = surah_num
        if ayah_num: data["ayah_num"] = ayah_num
        
        try:
            response = requests.post(
                api_url, 
                files={"file": f}, 
                data=data
            )
        except requests.ConnectionError:
            print(f"\n  Cannot connect to {api_url}")
            print("  Start the server first: python main.py")
            return
    
    if response.status_code != 200:
        print(f"\n  ERROR {response.status_code}: {response.text}")
        return
    
    result = response.json()
    
    # --- Pretty Print ---
    score_pct = int(result['accuracy'] * 100)
    status = "PASSED" if result['passed'] else "NEEDS PRACTICE"
    
    print(f"\n  Overall: {score_pct}% ({result['raw_score']}) — {status}")
    print(f"  Feedback Audio: {result['feedback_audio']}")
    if result.get('reference_audio'):
        print(f"  Sheikh Audio:   {result['reference_audio']}")
    
    print(f"\n  Your Recitation:")
    print(f"    {result['user_recitation']}")
    print(f"  Expected:")
    print(f"    {result['expected_recitation']}")
    
    if result['mistakes']:
        print(f"\n  Mistakes ({len(result['mistakes'])}):")
        for m in result['mistakes']:
            print(f"    - {m}")
    
    print(f"\n  Per-Word Analysis:")
    print("  " + "-" * 76)
    print(f"  {'#':<3} {'Word':<14} {'Expected':<14} {'Score':<7} {'OK':<4} {'Error':<10} {'Time'}")
    print("  " + "-" * 76)
    
    for i, w in enumerate(result.get('words', []), 1):
        word = w.get('word') or '-'
        expected = w.get('expected') or '-'
        score = f"{w['score']:.1f}" if w.get('score') is not None else '-'
        ok = 'Y' if w.get('is_correct') else 'N'
        err = w.get('error_type_ar') or '-'
        ts_s = w.get('timestamp_start')
        ts_e = w.get('timestamp_end')
        ts = f"{ts_s:.2f}-{ts_e:.2f}s" if ts_s is not None else '-'
        
        print(f"  {i:<3} {word:<14} {expected:<14} {score:<7} {ok:<4} {err:<10} {ts}")
        
        for ce in w.get('char_errors', []):
            t = ce.get('type', '')
            if ce.get('got') and ce.get('expected'):
                print(f"      > {t}: '{ce['got']}' -> '{ce['expected']}'")
            elif ce.get('expected'):
                print(f"      > {t}: missing '{ce['expected']}'")
            elif ce.get('got'):
                print(f"      > {t}: extra '{ce['got']}'")
    
    print("  " + "-" * 76)
    
    # Save JSON
    log_path = "last_response.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Full JSON saved: {log_path}")
    
    # Print raw JSON
    print(f"\n  Raw JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quran ASR Demo Client")
    parser.add_argument("--url", default="http://localhost:8000/grade_recitation")
    parser.add_argument("--file", default="test_audio.ogg", help="Path to audio file")
    parser.add_argument("--text", default="بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", help="Expected text")
    parser.add_argument("--surah", type=int, default=None)
    parser.add_argument("--ayah", type=int, default=None)
    args = parser.parse_args()
    
    test_api(args.url, args.file, args.text, args.surah, args.ayah)
