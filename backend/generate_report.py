import json
import sys

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def generate_arabic_report(json_path):
    print("generating report...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Header
    report = []
    report.append("📄 **التقرير اليومي للمراجعة**")
    report.append(f"**التاريخ:** 2025-12-31")
    report.append("-" * 30)
    
    # 2. Result Summary
    score = int(data['accuracy'] * 100)
    if data['passed']:
        status = "✅ **ممتاز! (Passed)**"
        msg = "أحسنت! تلاوتك ممتازة."
    else:
        status = "⚠️ **يحتاج إلى تدريب (Needs Practice)**"
        msg = "محاولة جيدة، ولكن هناك بعض الملاحظات."
        
    report.append(f"**الحالة:** {status}")
    report.append(f"**الدرجة:** {score}%")
    report.append(f"**تعليق المعلم:** {msg}")
    report.append("-" * 30)

    # 3. Details (The "Full" Transcript requested)
    report.append("🔊 **ما سمعه التطبيق (Your Recitation):**")
    report.append(f"> {data['user_recitation']}")
    report.append("")
    report.append("📖 **الصواب (Correct Verse):**")
    report.append(f"> {data['expected_recitation']}")
    report.append("-" * 30)
    
    # 4. Mistakes Analysis
    if data['mistakes']:
        report.append("❌ **الملاحظات (Mistakes):**")
        for mistake in data['mistakes']:
            # Translate technical errors to friendly Arabic
            if "Mistake: Said" in mistake:
                # Format: Mistake: Said 'X' instead of 'Y'
                parts = mistake.split("'")
                wrong_word = parts[1]
                right_word = parts[3]
                report.append(f"- **خطأ في الكلمة:** قلتَ ( {wrong_word} ) والصواب ( {right_word} )")
            elif "Added Word" in mistake:
                 parts = mistake.split("'")
                 added_word = parts[1]
                 report.append(f"- **كلمة زائدة:** أضفت ( {added_word} )")
            elif "Missed Word" in mistake:
                 parts = mistake.split("'")
                 missed_word = parts[1]
                 report.append(f"- **كلمة ناقصة:** نسيت أن تقرأ ( {missed_word} )")
            else:
                report.append(f"- {mistake}")
    else:
        report.append("🎉 **لا توجد أخطاء! واصل التقدم.**")

    # Save Report
    with open("daily_report_arabic.md", "w", encoding="utf-8") as out:
        out.write("\n".join(report))
    
    print("\n".join(report))

if __name__ == "__main__":
    generate_arabic_report("response_log.json")
