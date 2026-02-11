import asyncio
import edge_tts
import os

# Voice Options:
# ar-EG-SalmaNeural (Egyptian, Female) - Friendly
# ar-SA-HamedNeural (Saudi, Male) - Formal/Sheikh-like
# ar-AE-FatimaNeural (UAE, Female) - Neutral

VOICE = "ar-EG-SalmaNeural" 

async def generate_feedback_audio(mistakes, output_path):
    """
    Converts a list of mistakes into a friendly spoken Arabic message.
    """
    if not mistakes:
        text = "أحسنت! تلاوتك ممتازة، وليس لديك أي أخطاء. بارك الله فيك."
    else:
        # Construct the detailed message
        text_parts = ["انتبه يا بطل، لديك بعض الملاحظات."]
        
        for mistake in mistakes:
            # Mistake format: "خطأ: قلت 'X' بدلاً من 'Y'"
            # We want to make it sound natural.
            clean_mistake = mistake.replace("خطأ:", "").replace("كلمة ناقصة:", "نسيت أن تقرأ").replace("كلمة زائدة:", "أضفت كلمة")
            text_parts.append(clean_mistake)
            
        text_parts.append("حاول مرة أخرى، وأنا واثقة أنك ستحفظها.")
        text = " ... ".join(text_parts)

    # print(f"TTS Generating: {text}") # Disabled to avoid Windows encoding errors
    print("TTS Generating audio...")
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_path)
    
    return output_path

# Wrapper for synchronous calls if needed
def create_feedback_file(mistakes, output_filename):
    try:
        asyncio.run(generate_feedback_audio(mistakes, output_filename))
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

if __name__ == "__main__":
    # Test
    test_mistakes = [
        "خطأ: قلت 'هِيُهَا' بدلاً من 'أَيُّهَا'",
        "كلمة ناقصة: 'لَا'"
    ]
    create_feedback_file(test_mistakes, "test_feedback.mp3")
    print("Generated test_feedback.mp3")
