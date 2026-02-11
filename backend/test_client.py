import requests
import json
import sys

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Configuration
API_URL = "http://localhost:8000/grade_recitation"
FILE_PATH = "finetuning/test_samples/test 3.ogg" # Updated to Test 3
# Correct Text for Test 3 (Surah Al-Hadid 57:1-2)
TARGET_TEXT = "سَبَّحَ لِلَّهِ مَا فِي السَّمَاوَاتِ وَالْأَرْضِ وَهُوَ الْعَزِيزُ الْحَكِيمُ لَهُ مُلْكُ السَّمَاوَاتِ وَالْأَرْضِ يُحْيِي وَيُمِيتُ وَهُوَ عَلَى كُلِّ شَيْءٍ قَدِيرٌ"

def test_api():
    print("=" * 60)
    print("🎙️  Quran ASR Kids - Demo Client")
    print("=" * 60)
    print(f"\n📁 File: {FILE_PATH}")
    print(f"🔗 API:  {API_URL}")
    print(f"📖 Expected: {TARGET_TEXT[:50]}...")
    print()
    
    with open(FILE_PATH, 'rb') as f:
        
        try:
            response = requests.post(
                API_URL, 
                files={"file": f}, 
                data={
                    "target_ayah": TARGET_TEXT,
                    "surah_num": 57
                    # "ayah_num": 1 # Full Surah mode
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # --- Pretty Print Results ---
                print("✅ [SUCCESS] Response received!\n")
                
                # Overall Score
                score_pct = int(data['accuracy'] * 100)
                status = "✅ PASSED" if data['passed'] else "❌ NEEDS PRACTICE"
                print(f"📊 Overall Score: {score_pct}% ({data['raw_score']}) — {status}")
                print(f"🔊 Feedback Audio: {data['feedback_audio']}")
                if data.get('reference_audio'):
                    print(f"🎧 Sheikh Audio:   {data['reference_audio']}")
                print()
                
                # Transcription Comparison
                print("🎤 Your Recitation:")
                print(f"   {data['user_recitation']}")
                print(f"📖 Expected:")
                print(f"   {data['expected_recitation']}")
                print()
                
                # Mistakes Summary
                if data['mistakes']:
                    print(f"❌ Mistakes ({len(data['mistakes'])}):")
                    for m in data['mistakes']:
                        print(f"   • {m}")
                else:
                    print("🎉 No mistakes! Perfect recitation!")
                print()
                
                # Per-Word Detail
                print("📝 Per-Word Analysis:")
                print("-" * 80)
                print(f"{'#':<4} {'Word':<15} {'Expected':<15} {'Score':<8} {'Error':<12} {'Timestamps'}")
                print("-" * 80)
                
                for i, w in enumerate(data.get('words', []), 1):
                    word = w.get('word') or '—'
                    expected = w.get('expected') or '—'
                    score = f"{w['score']:.1f}" if w.get('score') is not None else '—'
                    error = w.get('error_type_ar') or '✅'
                    ts_start = w.get('timestamp_start')
                    ts_end = w.get('timestamp_end')
                    ts = f"{ts_start:.2f}s - {ts_end:.2f}s" if ts_start is not None else '—'
                    
                    correct_mark = "✅" if w.get('is_correct') else "❌"
                    print(f"{i:<4} {word:<15} {expected:<15} {score:<8} {correct_mark} {error:<10} {ts}")
                    
                    # Show character-level errors if any
                    if w.get('char_errors'):
                        for ce in w['char_errors']:
                            ce_type = ce.get('type', '')
                            if ce.get('got') and ce.get('expected'):
                                print(f"     └─ {ce_type}: '{ce.get('got')}' → '{ce.get('expected')}'")
                            elif ce.get('expected'):
                                print(f"     └─ {ce_type}: missing '{ce.get('expected')}'")
                            elif ce.get('got'):
                                print(f"     └─ {ce_type}: extra '{ce.get('got')}'")
                
                print("-" * 80)
                print()
                
                # Save full JSON response
                import os
                os.makedirs("data/logs", exist_ok=True)
                log_path = "data/logs/response_log.json"
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"💾 Full JSON saved to: {log_path}")
                
                # Also print raw JSON
                print(f"\n📋 Raw JSON Response:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                
            else:
                print(f"\n❌ [ERROR] {response.status_code}: {response.text}")
                
        except requests.ConnectionError:
            print(f"\n❌ Cannot connect to {API_URL}")
            print("   Make sure the server is running: python backend/main.py")
        except FileNotFoundError:
            print(f"\n❌ Audio file not found: {FILE_PATH}")
        except Exception as e:
            print(f"\n❌ Failed: {e}")

if __name__ == "__main__":
    test_api()
