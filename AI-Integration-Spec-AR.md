# مواصفات الربط بين خدمة الـ AI و باك إند "قرآن يُتلى"

**موجّه إلى:** مهندس الـ AI — Quran-ASR  
**من:** فريق الباك إند — Quran Yutla  
**الإصدار:** 1.0

---

## 1. الملخص التنفيذي

الباك إند بتاعنا (NestJS) مبني من الأساس على إن خدمة الـ AI تشتغل **بشكل غير متزامن (Async)**:

> نبعتلك الطلب → ترد علينا فوراً بـ `jobId` → تشتغل في الخلفية → تبعتلنا النتيجة على Webhook.

الخدمة الحالية عندك (`POST /grade_recitation`) شغالة **متزامنة (Sync)** — بتستنى لحد ما تخلص وترد بالنتيجة في نفس الـ response.

**المطلوب منك:** إضافة **endpoint واحد جديد** اسمه `POST /api/evaluate` جنب اللي عندك، من غير ما تلمس `/grade_recitation` الحالي (هيفضل شغال زي ما هو للاختبار والـ demo).

| البند | القيمة |
| :--- | :--- |
| **حجم الشغل** | ~80–120 سطر Python |
| **ملفات هتتعدل** | `backend/main.py` + إضافة دالة في `src/quran_db/core.py` |
| **حاجة جديدة تتسطّب** | `httpx` (لو مش متسطّب) |
| **منطق الـ Grading** | ❌ **مش هيتغير خالص** — نفس الـ grader والموديل بالظبط |

**ليه Async مش Sync؟** لأن الـ Whisper على تسجيل طويل ممكن ياخد 10–60 ثانية. لو خليناها sync، الطفل هيقعد على شاشة انتظار، والـ request هيعمل timeout على الـ Nginx / Kubernetes ingress قبل ما تخلص. كمان لو الخدمة وقعت لحظة الرفع، التسميع هيضيع بدل ما يتعاد.

---

## 2. الفرق بين الموجود والمطلوب

| | خدمتك الحالية (`/grade_recitation`) | المطلوب (`/api/evaluate`) |
| :--- | :--- | :--- |
| **نوع الطلب** | `multipart/form-data` | `application/json` |
| **الصوت** | `file` (الملف نفسه) | `audioUrl` (رابط تنزّله بنفسك) |
| **النص المستهدف** | `target_ayah` (نص جاهز) | تجيبه إنت من `surahNumber` + `fromAyah` + `toAyah` |
| **النطاق** | آية واحدة | **نطاق آيات** (من آية ٣ لآية ٧ مثلاً) |
| **المصادقة** | مفيش | `Authorization: Bearer {AI_API_KEY}` |
| **الرد** | النتيجة كاملة (بعد الانتظار) | `{ status, jobId }` **فوراً** |
| **النتيجة النهائية** | نفس الـ response | Webhook على سيرفرنا |
| **مقياس الدرجة** | `accuracy` من 0.0 لـ 1.0 | `overallScore` من **0 لـ 100** |

---

## 3. الـ Endpoint الجديد: `POST /api/evaluate`

### 3.1 الطلب اللي هيوصلك

```http
POST /api/evaluate
Content-Type: application/json
Authorization: Bearer {AI_API_KEY}
```

```json
{
  "audioUrl": "https://quran-yutla-container.s3.gra.io.cloud.ovh.net/recitations/a1b2-c3d4-recording.webm",
  "surahNumber": 1,
  "surahName": "الفاتحة",
  "fromAyah": 1,
  "toAyah": 7,
  "userId": 42,
  "recitationId": 1337,
  "webhookUrl": "https://api.yutlaquran.com/api/v1/recitations/webhook/ai-evaluation",
  "webhookSecret": "<سلسلة سرية هنبعتهالك بشكل منفصل وآمن>"
}
```

| الحقل | النوع | الوصف |
| :--- | :--- | :--- |
| `audioUrl` | string | رابط مباشر للملف الصوتي على تخزين OVH S3 |
| `surahNumber` | int | رقم السورة (1–114) |
| `surahName` | string | اسم السورة بالعربي (للمعلومية فقط) |
| `fromAyah` | int | أول آية في النطاق |
| `toAyah` | int | آخر آية في النطاق (**ممكن تساوي `fromAyah`**) |
| `userId` | int | معرّف المستخدم (رجّعه لنا زي ما هو) |
| `recitationId` | int | معرّف التسميع (رجّعه لنا زي ما هو) |
| `webhookUrl` | string | العنوان اللي هتبعتلنا عليه النتيجة |
| `webhookSecret` | string | السر اللي هتحطه في هيدر الـ Webhook |

> **مهم:** الـ `webhookUrl` و `webhookSecret` بيوصلوك **في كل طلب**. متخزنهمش hardcoded في الكود — استخدم اللي جايلك في الطلب نفسه، لأنهم بيختلفوا بين بيئة التطوير والإنتاج.

### 3.2 الرد المطلوب منك — **فوراً وقبل ما تشتغل**

**عند النجاح (HTTP 200 أو 202):**

```json
{
  "status": "processing",
  "jobId": "8f14e45f-ceea-467a-9b8c-1f0c1e5b4a21",
  "estimatedTime": 30
}
```

| الحقل | النوع | مطلوب | ملاحظات |
| :--- | :--- | :---: | :--- |
| `status` | string | ✅ | لازم تكون بالضبط `"processing"` |
| `jobId` | string | ✅ | UUID فريد — **احفظه، لأنك هترجّعه في الـ Webhook** |
| `estimatedTime` | int | ❌ | ثواني تقديرية |

**عند الرفض (مثلاً مفتاح غلط أو بيانات ناقصة):**

```json
{
  "status": "error",
  "message": "Invalid API key",
  "code": "AUTH_FAILED"
}
```

> ⚠️ الرد ده لازم يرجع **في أقل من ثانيتين**. متستناش الموديل يخلص. استخدم `BackgroundTasks` بتاعة FastAPI.

### 3.3 المصادقة

كل طلب جاي منّا فيه هيدر:

```
Authorization: Bearer {AI_API_KEY}
```

الـ `AI_API_KEY` ده سلسلة عشوائية هنتفق عليها ونبعتهالك. حطها عندك في متغير بيئة وقارنها. لو مش مطابقة → `401`.

---

## 4. الـ Webhook: إرسال النتيجة لنا

بعد ما تخلص المعالجة، ابعت `POST` على الـ `webhookUrl` اللي وصلك في الطلب.

```http
POST {webhookUrl}
Content-Type: application/json
Authorization: Bearer {webhookSecret}
```

### 4.1 ⚠️ تحذير حرج: الـ Body بيتحقق منه بصرامة

الباك إند بتاعنا شغال بـ `forbidNonWhitelisted: true`. يعني:

> **أي حقل زيادة في المستوى الأول (top-level) غير الحقول الستة دي → الطلب هيترفض بـ `400` والنتيجة هتضيع.**

الحقول المسموح بيها في المستوى الأول — **ستة فقط لا غير**:

| الحقل | النوع | مطلوب | القيم |
| :--- | :--- | :---: | :--- |
| `jobId` | string | ✅ | نفس الـ `jobId` اللي رجّعته بالظبط |
| `recitationId` | int | ✅ | نفس اللي وصلك |
| `userId` | int | ✅ | نفس اللي وصلك |
| `status` | string | ✅ | `"success"` أو `"error"` فقط |
| `data` | object | عند النجاح | **جوّاه حط اللي إنت عايزه بحرية** |
| `message` | string | عند الخطأ | وصف الخطأ |

✅ **جوّه `data` مفيش أي قيود** — حط أي حقول بأي شكل. بس **برّه `data` ممنوع أي حاجة زيادة**.

```json
// ❌ غلط — هيترفض 400
{
  "jobId": "...",
  "recitationId": 1337,
  "userId": 42,
  "status": "success",
  "request_id": "abc",        // ← حقل زيادة، هيفشّل الطلب كله
  "processing_time": 12.4,    // ← حقل زيادة، هيفشّل الطلب كله
  "data": { }
}

// ✅ صح — نفس المعلومات بس جوّه data
{
  "jobId": "...",
  "recitationId": 1337,
  "userId": 42,
  "status": "success",
  "data": {
    "requestId": "abc",
    "processingTime": 12.4
  }
}
```

### 4.2 مثال النجاح الكامل

```json
{
  "jobId": "8f14e45f-ceea-467a-9b8c-1f0c1e5b4a21",
  "recitationId": 1337,
  "userId": 42,
  "status": "success",
  "data": {
    "overallScore": 87.5,
    "passed": true,
    "totalWords": 8,
    "correctWords": 7,
    "incorrectWords": 1,
    "userRecitation": "بسم الله الرحمن الرحيم الحمد لله رب",
    "expectedRecitation": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ",
    "words": [
      {
        "word": "بسم",
        "expected": "بسم",
        "isCorrect": true,
        "score": 1.0,
        "errorType": null,
        "errorTypeAr": null,
        "charErrors": [],
        "timestampStart": 0.50,
        "timestampEnd": 0.92
      },
      {
        "word": "اللا",
        "expected": "الله",
        "isCorrect": false,
        "score": 0.667,
        "errorType": "substitution",
        "errorTypeAr": "استبدال",
        "charErrors": [
          {
            "type": "استبدال_حرف",
            "typeEn": "char_substitution",
            "position": 3,
            "got": "ا",
            "expected": "ه"
          }
        ],
        "timestampStart": 0.92,
        "timestampEnd": 1.40
      },
      {
        "word": null,
        "expected": "العالمين",
        "isCorrect": false,
        "score": 0.0,
        "errorType": "deletion",
        "errorTypeAr": "حذف",
        "charErrors": [],
        "timestampStart": null,
        "timestampEnd": null
      }
    ],
    "errors": [
      "خطأ: قلت 'اللا' بدلاً من 'الله'",
      "كلمة ناقصة: 'العالمين'"
    ],
    "errorSummary": {
      "substitution": 1,
      "deletion": 1,
      "insertion": 0
    },
    "suggestions": [
      "ركّز على نطق حرف الهاء في نهاية كلمة 'الله'"
    ],
    "feedbackAudio": "https://ai.yutlaquran.com/audio/feedback_8f14e45f.mp3",
    "referenceAudio": "https://everyayah.com/data/Minshawy_Mujawwad_192kbps/001001.mp3",
    "segmentsProcessed": 2,
    "modelVersion": "v5-30k"
  }
}
```

### 4.3 مثال الفشل

```json
{
  "jobId": "8f14e45f-ceea-467a-9b8c-1f0c1e5b4a21",
  "recitationId": 1337,
  "userId": 42,
  "status": "error",
  "message": "Failed to download audio file: 404 Not Found"
}
```

### 4.4 الرد اللي هيرجعلك منّا

```json
{ "success": true, "message": "Evaluation received and saved" }
```

| كود الرد | المعنى | تعمل إيه |
| :---: | :--- | :--- |
| `200` | اتسجّلت | خلاص، متبعتش تاني |
| `400` | الـ body شكله غلط | **متعملش retry** — صلّح الشكل |
| `401` | الـ `webhookSecret` غلط | **متعملش retry** — راجع السر |
| `404` | مفيش تسميع بالـ `recitationId` + `jobId` دول | **متعملش retry** — راجع إنك بترجّع نفس القيم |
| `5xx` أو timeout | مشكلة عندنا | ✅ **اعمل retry** |

### 4.5 سياسة إعادة المحاولة (مطلوبة)

الشبكة بتقع، وسيرفرنا ممكن يعيد التشغيل لحظة الـ deploy. لو ضاعت الرسالة، التسميع هيفضل عالق عند الطالب.

**المطلوب:** 3 محاولات بـ exponential backoff (1s → 2s → 4s)، **وبس** على أخطاء `5xx` والـ timeout. مع `4xx` متعيدش.

---

## 5. جدول تحويل المخرجات

مخرجات الـ grader بتاعك ممتازة وكاملة — كل اللي محتاجينه إعادة تسمية بسيطة و **تحويل مقياس الدرجة**.

| عندك دلوقتي | المطلوب في `data` | التحويل |
| :--- | :--- | :--- |
| `accuracy` (0.0–1.0) | `overallScore` (0–100) | 🔴 **`round(accuracy * 100, 2)`** |
| `passed` | `passed` | نفسه |
| `user_recitation` | `userRecitation` | إعادة تسمية |
| `expected_recitation` | `expectedRecitation` | إعادة تسمية |
| `mistakes[]` | `errors[]` | إعادة تسمية |
| `words[]` | `words[]` | نفسه (بالتسميات تحت) |
| `words[].is_correct` | `words[].isCorrect` | إعادة تسمية |
| `words[].error_type` | `words[].errorType` | إعادة تسمية |
| `words[].error_type_ar` | `words[].errorTypeAr` | إعادة تسمية |
| `words[].char_errors` | `words[].charErrors` | إعادة تسمية |
| `words[].timestamp_start` | `words[].timestampStart` | إعادة تسمية |
| `words[].timestamp_end` | `words[].timestampEnd` | إعادة تسمية |
| `char_errors[].type_en` | `charErrors[].typeEn` | إعادة تسمية |
| `feedback_audio` | `feedbackAudio` | إعادة تسمية + انظر §6.4 |
| `reference_audio` | `referenceAudio` | إعادة تسمية |
| `segments_processed` | `segmentsProcessed` | إعادة تسمية |
| `raw_score` ("3/4") | `correctWords` / `totalWords` | فكّها لرقمين |
| `request_id` | `data.requestId` | لازم تبقى **جوّه** `data` |

> **ملاحظة عن الـ camelCase:** التسمية دي مش إجبارية تقنياً (الحقول جوّه `data` مش بتتحقق) — بس تطبيق الموبايل بتاعنا شغال بـ camelCase، فلو مشيت عليها هتوفّر علينا طبقة تحويل زيادة. **الحقول الستة اللي برّه `data` إجبارية بالأسماء دي بالضبط.**

---

## 6. ⚠️ خمس نقاط حرجة — راجعها كويس

### 6.1 🔴 مقياس الدرجة: 0–100 مش 0.0–1.0

إحنا بنخزن الدرجة في عمود `evaluation_score` من نوع `numeric(5,2)` — يعني بيقبل أي رقم لحد `999.99`، **ومفيش أي قيد بيرفض الأرقام الصغيرة**.

يعني لو بعتلنا `0.95`:
- الداتابيز هتقبلها من غير أي خطأ ✅
- بس هتتعرض للطالب على إنها **٠.٩٥٪** بدل ٩٥٪ ❌
- وكل الإحصائيات والمتوسطات والتقارير هتبقى غلط
- **ومحدش هيلاحظ** لأن مفيش رسالة خطأ

ده **أخطر بند في الوثيقة كلها** لأن الغلط بيعدّي صامت. تأكد من `overallScore` قبل ما تبعت.

### 6.2 🔴 نطاق آيات — مش آية واحدة

`/grade_recitation` الحالي بياخد `target_ayah` (آية واحدة). الطلب الجديد بيدي `fromAyah` و `toAyah`، والفرق بينهم ممكن يكون **٢٠ آية أو أكتر**.

لازم تضيف دالة في `src/quran_db/core.py` تجمع نص النطاق كله:

```python
def get_ayah_range(self, surah_num: int, from_ayah: int, to_ayah: int) -> str:
    """يرجّع نص كل الآيات من from_ayah لـ to_ayah متصلة بمسافة."""
    cur = self.conn.cursor()
    cur.execute(
        """SELECT aya_text FROM ayahs
           WHERE sura_no = ? AND aya_no BETWEEN ? AND ?
           ORDER BY aya_no""",
        (surah_num, from_ayah, to_ayah),
    )
    rows = cur.fetchall()
    if not rows:
        raise ValueError(f"No ayahs found: surah={surah_num} {from_ayah}-{to_ayah}")
    return " ".join(r[0] for r in rows)
```

*(عدّل أسماء الجداول/الأعمدة حسب الـ schema الفعلي عندك.)*

لو بعتّ الآية الأولى بس، أي تسميع أكتر من آية هيرجع درجة منخفضة جداً وغلط تماماً.

### 6.3 🟡 تأكد إن نص القرآن عندك "نظيف"

إحنا الاتنين مصدرنا نفس الملف `hafsData_v2-0.sql`. **بس** إحنا بنستخدم نسخة منظّفة (`hafsData_v2-0-clean.sql`) شيلنا منها رموز نهاية الآية (النطاق `U+FC00`–`U+FDFF` — الرموز زي `ﰀ ﰁ ﰂ`) لأنها مش جزء من النص.

**لو الـ `quran.db` بتاعك متبني من الملف الخام**، النص المستهدف هيبقى فيه الرموز دي، والـ grader هيحاول يقارنها بكلام الطفل ويحسبها أخطاء وهمية.

**التحقق:**
```python
import re
text = db.get_ayah_range(1, 1, 7)
markers = re.findall(r'[ﰀ-﷿]', text)
print(f"Found {len(markers)} markers — لازم تكون 0")
```

لو طلعت أكبر من صفر، نظّف الداتابيز بنفس المنطق:
```python
text = re.sub(r'[ﰀ-﷿]', '', text)
```

### 6.4 🟡 روابط الصوت لازم تكون عامة ومطلقة

`feedback_audio` عندك دلوقتي بيرجع `http://localhost:8000/audio/feedback_xxx.mp3`.

الرابط ده **هيتخزن عندنا زي ما هو** وتطبيق الموبايل هيحاول يفتحه — وهيفشل، لأن `localhost` بالنسبة للموبايل يعني الموبايل نفسه.

الحل: متغير بيئة عندك بالـ domain الحقيقي:

```python
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
feedback_url = f"{PUBLIC_BASE_URL}/audio/feedback_{job_id}.mp3"
```

**كمان مهم:** ملفات الـ TTS دي محتاجة تفضل موجودة على الأقل ٣٠ يوم (ده عمر التسميع عندنا قبل الحذف التلقائي). لو بتمسحها بعد الرد، الطالب مش هيقدر يسمع التقييم الصوتي لما يفتح تسميعه القديم.

### 6.5 🟡 صيغة `webm` مدعومة؟

التطبيق بيسجّل مباشرة من المتصفح/الموبايل باستخدام `MediaRecorder`، واللي بيطلع **`audio/webm`** أو `video/webm` في الغالب — مش MP3.

الـ README بتاعك بيقول `MP3, WAV, OGG, M4A`. `ffmpeg` بيدعم `webm` عادي، بس **اختبرها فعلياً** قبل ما نربط. لو `librosa` وقعت عليها، حوّلها بـ `ffmpeg` قبل ما تدخل الموديل.

الصيغ اللي ممكن توصلك: `webm`, `ogg`, `mp4`, `m4a`, `mp3`, `wav`.

---

## 7. كود جاهز للنسخ

```python
# ==== backend/main.py ====
import os, uuid, asyncio, tempfile, logging
import httpx
from fastapi import BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

log = logging.getLogger("evaluate")

AI_API_KEY      = os.getenv("AI_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")


class EvaluateRequest(BaseModel):
    audioUrl: str
    surahNumber: int
    surahName: str | None = None
    fromAyah: int
    toAyah: int
    userId: int
    recitationId: int
    webhookUrl: str
    webhookSecret: str


@app.post("/api/evaluate")
async def evaluate_async(
    req: EvaluateRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None),
):
    if not AI_API_KEY or authorization != f"Bearer {AI_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    job_id = str(uuid.uuid4())
    background_tasks.add_task(_process_and_callback, job_id, req)

    # ← بيرجع فوراً، من غير ما ينتظر الموديل
    return {"status": "processing", "jobId": job_id, "estimatedTime": 30}


async def _process_and_callback(job_id: str, req: EvaluateRequest):
    audio_path = None
    try:
        # 1) نزّل الصوت من التخزين السحابي
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            resp = await c.get(req.audioUrl)
            resp.raise_for_status()

        suffix = os.path.splitext(req.audioUrl.split("?")[0])[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(resp.content)
            audio_path = f.name

        # 2) هات نص النطاق كله من quran.db  ← راجع §6.2
        target_text = quran_db.get_ayah_range(
            req.surahNumber, req.fromAyah, req.toAyah
        )

        # 3) نفس منطق /grade_recitation — من غير أي تغيير
        result = grade_recitation_internal(
            audio_path=audio_path,
            target_ayah=target_text,
            surah_num=req.surahNumber,
            ayah_num=req.fromAyah,
        )

        words = result["words"]
        payload = {
            "jobId": job_id,
            "recitationId": req.recitationId,
            "userId": req.userId,
            "status": "success",
            "data": {
                # 🔴 التحويل الحرج: 0.0–1.0  →  0–100
                "overallScore": round(result["accuracy"] * 100, 2),
                "passed": result["passed"],
                "totalWords": len(words),
                "correctWords": sum(1 for w in words if w["is_correct"]),
                "incorrectWords": sum(1 for w in words if not w["is_correct"]),
                "userRecitation": result["user_recitation"],
                "expectedRecitation": result["expected_recitation"],
                "words": [
                    {
                        "word": w["word"],
                        "expected": w["expected"],
                        "isCorrect": w["is_correct"],
                        "score": w["score"],
                        "errorType": w["error_type"],
                        "errorTypeAr": w["error_type_ar"],
                        "charErrors": [
                            {
                                "type": ce["type"],
                                "typeEn": ce["type_en"],
                                "position": ce["position"],
                                "got": ce.get("got"),
                                "expected": ce.get("expected"),
                            }
                            for ce in w.get("char_errors", [])
                        ],
                        "timestampStart": w["timestamp_start"],
                        "timestampEnd": w["timestamp_end"],
                    }
                    for w in words
                ],
                "errors": result["mistakes"],
                "feedbackAudio": (
                    f"{PUBLIC_BASE_URL}/audio/feedback_{job_id}.mp3"
                    if result.get("feedback_audio") else None
                ),
                "referenceAudio": result.get("reference_audio"),
                "segmentsProcessed": result.get("segments_processed", 1),
                "requestId": job_id,
                "modelVersion": "v5-30k",
            },
        }

    except Exception as e:
        log.exception("evaluation failed for job %s", job_id)
        payload = {
            "jobId": job_id,
            "recitationId": req.recitationId,
            "userId": req.userId,
            "status": "error",
            "message": str(e)[:500],
        }
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

    await _send_webhook(req.webhookUrl, req.webhookSecret, payload)


async def _send_webhook(url: str, secret: str, payload: dict, attempts: int = 3):
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as c:
        for i in range(attempts):
            try:
                r = await c.post(url, json=payload, headers=headers)
                if r.status_code < 400:
                    log.info("webhook delivered: %s", payload["jobId"])
                    return
                if 400 <= r.status_code < 500:
                    # غلط في البيانات مش في الشبكة — الإعادة مش هتنفع
                    log.error("webhook rejected %s: %s", r.status_code, r.text[:300])
                    return
            except Exception as e:
                log.warning("webhook attempt %d failed: %s", i + 1, e)
            await asyncio.sleep(2 ** i)
    log.error("webhook FAILED after %d attempts: %s", attempts, payload["jobId"])
```

**متغيرات البيئة المطلوبة عندك:**

```bash
AI_API_KEY=<اللي هنتفق عليه>
PUBLIC_BASE_URL=https://ai.yutlaquran.com
```

---

## 8. قائمة القبول — اختبرها قبل ما تقولنا جاهز

اشتغل على الجدول ده بالترتيب:

- [ ] `POST /api/evaluate` بيرجع في **أقل من ثانيتين** بـ `{ status:"processing", jobId }`
- [ ] الطلب من غير `Authorization` أو بمفتاح غلط → `401`
- [ ] بينزّل الملف من `audioUrl` بنجاح (اختبر برابط OVH حقيقي هنبعتهولك)
- [ ] بيشتغل صح مع `webm` (مش MP3 بس)
- [ ] `get_ayah_range(1, 1, 7)` بترجّع **الفاتحة كاملة** مش آية واحدة
- [ ] النص الراجع **مفيهوش** رموز `U+FC00`–`U+FDFF` (راجع §6.3)
- [ ] `overallScore` في الـ webhook رقم **بين 0 و 100** (مش كسر عشري ≤ 1) 🔴
- [ ] الـ `jobId` في الـ webhook **مطابق حرفياً** للـ `jobId` اللي رجّعته أول مرة
- [ ] الـ body برّه `data` فيه **الستة حقول دول بس** ولا حاجة زيادة 🔴
- [ ] `feedbackAudio` رابط مطلق بـ domain حقيقي، مش `localhost`
- [ ] حالة الفشل بتبعت `status:"error"` + `message` (**من غير** `data`)
- [ ] لو الـ webhook رجّع `500`، بيعيد المحاولة ٣ مرات؛ لو `400`/`401` مبيعيدش
- [ ] الملف المؤقت بيتمسح في كل الحالات (نجاح وفشل)

### طريقة اختبار سريعة من غير ما تستنانا

استقبل الـ webhook بنفسك عشان تشوف شكل الـ payload:

```bash
pip install fastapi uvicorn
```

```python
# mock_backend.py  →  uvicorn mock_backend:app --port 3777
from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/api/v1/recitations/webhook/ai-evaluation")
async def hook(request: Request):
    body = await request.json()
    print("AUTH:", request.headers.get("authorization"))
    print("TOP-LEVEL KEYS:", list(body.keys()))   # لازم تكون 6 بالظبط
    print("SCORE:", body.get("data", {}).get("overallScore"))  # لازم 0-100
    return {"success": True, "message": "ok"}
```

وبعدين:

```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "audioUrl": "https://example.com/sample.mp3",
    "surahNumber": 1, "surahName": "الفاتحة",
    "fromAyah": 1, "toAyah": 7,
    "userId": 1, "recitationId": 1,
    "webhookUrl": "http://localhost:3777/api/v1/recitations/webhook/ai-evaluation",
    "webhookSecret": "test-secret"
  }'
```

---

## 9. اللي **مش** مطلوب منك

عشان نوفر وقت — الحاجات دي كلها مش في نطاق الشغل ده:

- ❌ **متغيّرش** `POST /grade_recitation` — سيبه شغال زي ما هو
- ❌ **متغيّرش** منطق الـ grader ولا الموديل ولا الـ VAD ولا الـ TTS
- ❌ مش محتاج تتعامل مع S3 credentials — الرابط اللي هيوصلك بيفتح مباشرة
- ❌ مش محتاج CORS — الاتصال server-to-server مش من المتصفح
- ❌ مش محتاج تخزّن حالة الـ jobs في داتابيز — `BackgroundTasks` كفاية
- ❌ مش محتاج تعمل endpoint للاستعلام عن حالة الـ job — إحنا بنعتمد على الـ webhook
- ✅ `POST /report_issue` سيبه زي ما هو — هنربطه في مرحلة تانية

---

## 10. الخطوات الجاية

| # | المهمة | المسؤول |
| :---: | :--- | :--- |
| 1 | نبعت `AI_API_KEY` و `AI_WEBHOOK_SECRET` بقناة آمنة | باك إند |
| 2 | نبعت رابط OVH حقيقي لملف اختبار | باك إند |
| 3 | تنفيذ `/api/evaluate` + `get_ayah_range` | **AI** |
| 4 | اجتياز قائمة القبول (§8) | **AI** |
| 5 | تصليح `app.url` و `audioUrl` في الباك | باك إند |
| 6 | اختبار متكامل بصوت طفل حقيقي | الاتنين |
| 7 | نشر الخدمة داخل نفس الـ Kubernetes cluster | DevOps |

**ملاحظة للنشر:** لو الخدمة هتتنشر جوّه نفس الـ cluster بتاعنا (namespace `quran-yutla`)، هنكلّمك على العنوان الداخلي مباشرة ومش هتحتاج تتعرّض للإنترنت العام أصلاً — أأمن وأسرع.

---

## 11. للتواصل حول أي بند

أي حقل في الوثيقة دي **قابل للنقاش قبل التنفيذ** — لو في حاجة أصعب من المتوقع أو ليها طريقة أحسن، اتكلم قبل ما تبدأ. اللي **مش** قابل للتفاوض بند واحد بس:

> `overallScore` لازم يكون **0–100**، والحقول برّه `data` لازم تكون **ستة بالظبط**.

الاتنين دول لو اتكسروا، هيفشلوا بصمت من غير أي رسالة خطأ — وده أسوأ نوع من الأعطال.
