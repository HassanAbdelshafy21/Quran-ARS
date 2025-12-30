# Dataset Normalization Audit

## 1. Character Check
**Unique Characters found in Training Data (First 500 lines):**
`    ء آ أ ؤ إ ئ ا ب ة ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ـ ف ق ك ل م ن ه و ى ي ً ٌ ٍ َ ُ ِ ّ ْ ٓ ٔ ٖ ٗ ٞ ٰ ٱ ۖ ۗ ۚ ۛ ۞ ۠ ۡ ۢ ۥ ۦ ۧ ۭ ﰀ ﰁ ﰂ ﰃ ﰄ ﰅ ﰆ ﰇ ﰈ ﰉ ﰊ ﰋ ﰌ ﰍ ﰎ ﰏ ﰐ ﰑ ﰒ ﰓ ﰔ ﰕ ﰖ ﰗ ﰘ ﰙ ﰚ ﰛ ﰜ ﰝ ﰟ ﰠ ﰡ ﰢ ﰤ ﰥ ﰦ ﰧ ﰨ ﰩ ﰪ ﰫ ﰬ ﰭ ﰮ ﰯ ﰰ ﰱ ﰲ ﰳ ﰴ ﰵ ﰶ ﰷ ﰸ ﰹ ﰺ ﰻ ﰼ ﰾ ﰿ ﱀ ﱁ ﱂ ﱃ ﱄ ﱅ ﱆ ﱇ ﱈ ﱉ ﱊ ﱋ ﱌ ﱍ ﱎ ﱏ ﱐ ﱑ ﱒ ﱓ ﱔ ﱕ ﱖ ﱗ ﱘ ﱙ ﱚ ﱛ ﱜ ﱝ ﱞ ﱟ ﱠ ﱡ ﱢ ﱣ ﱤ ﱥ ﱦ ﱧ ﱨ ﱩ ﱪ ﱫ ﱬ ﱮ ﱰ ﱱ ﱲ ﱳ ﱴ ﱵ ﱶ ﱷ ﱸ ﱹ ﱺ ﱻ ﱼ ﱽ ﱾ ﱿ ﲁ ﲄ ﲈ ﲉ ﲎ ﲏ ﲑ ﲒ ﲔ ﲗ ﲘ ﲙ ﲚ ﲜ ﲝ ﲟ ﲠ ﲡ ﲢ ﲧ ﲲ ﲳ ﲴ ﲵ ﲶ ﲷ ﲺ ﲼ ﲽ ﳂ ﳃ ﳆ ﳇ ﳈ ﳉ ﳟ ﳠ ﳭ ﳮ ﳻ ﴄ ﴑ �`

| Char | Hex | Status in Normalizer |
|---|---|---|
|   | 0020 | KEPT |
|   | 00a0 | REMOVED |
| ء | 0621 | REMOVED |
| آ | 0622 | KEPT |
| أ | 0623 | KEPT |
| ؤ | 0624 | REMOVED |
| إ | 0625 | KEPT |
| ئ | 0626 | REMOVED |
| ا | 0627 | KEPT |
| ب | 0628 | KEPT |
| ة | 0629 | KEPT |
| ت | 062a | KEPT |
| ث | 062b | KEPT |
| ج | 062c | KEPT |
| ح | 062d | KEPT |
| خ | 062e | KEPT |
| د | 062f | KEPT |
| ذ | 0630 | KEPT |
| ر | 0631 | KEPT |
| ز | 0632 | KEPT |
| س | 0633 | KEPT |
| ش | 0634 | KEPT |
| ص | 0635 | KEPT |
| ض | 0636 | KEPT |
| ط | 0637 | KEPT |
| ظ | 0638 | KEPT |
| ع | 0639 | KEPT |
| غ | 063a | KEPT |
| ـ | 0640 | REMOVED |
| ف | 0641 | KEPT |
| ق | 0642 | KEPT |
| ك | 0643 | KEPT |
| ل | 0644 | KEPT |
| م | 0645 | KEPT |
| ن | 0646 | KEPT |
| ه | 0647 | KEPT |
| و | 0648 | KEPT |
| ى | 0649 | KEPT |
| ي | 064a | KEPT |
| ً | 064b | REMOVED |
| ٌ | 064c | REMOVED |
| ٍ | 064d | REMOVED |
| َ | 064e | REMOVED |
| ُ | 064f | REMOVED |
| ِ | 0650 | REMOVED |
| ّ | 0651 | REMOVED |
| ْ | 0652 | REMOVED |
| ٓ | 0653 | REMOVED |
| ٔ | 0654 | REMOVED |
| ٖ | 0656 | REMOVED |
| ٗ | 0657 | REMOVED |
| ٞ | 065e | REMOVED |
| ٰ | 0670 | REMOVED |
| ٱ | 0671 | KEPT |
| ۖ | 06d6 | REMOVED |
| ۗ | 06d7 | REMOVED |
| ۚ | 06da | REMOVED |
| ۛ | 06db | REMOVED |
| ۞ | 06de | REMOVED |
| ۠ | 06e0 | REMOVED |
| ۡ | 06e1 | REMOVED |
| ۢ | 06e2 | REMOVED |
| ۥ | 06e5 | REMOVED |
| ۦ | 06e6 | REMOVED |
| ۧ | 06e7 | REMOVED |
| ۭ | 06ed | REMOVED |
| ﰀ | fc00 | REMOVED |
| ﰁ | fc01 | REMOVED |
| ﰂ | fc02 | REMOVED |
| ﰃ | fc03 | REMOVED |
| ﰄ | fc04 | REMOVED |
| ﰅ | fc05 | REMOVED |
| ﰆ | fc06 | REMOVED |
| ﰇ | fc07 | REMOVED |
| ﰈ | fc08 | REMOVED |
| ﰉ | fc09 | REMOVED |
| ﰊ | fc0a | REMOVED |
| ﰋ | fc0b | REMOVED |
| ﰌ | fc0c | REMOVED |
| ﰍ | fc0d | REMOVED |
| ﰎ | fc0e | REMOVED |
| ﰏ | fc0f | REMOVED |
| ﰐ | fc10 | REMOVED |
| ﰑ | fc11 | REMOVED |
| ﰒ | fc12 | REMOVED |
| ﰓ | fc13 | REMOVED |
| ﰔ | fc14 | REMOVED |
| ﰕ | fc15 | REMOVED |
| ﰖ | fc16 | REMOVED |
| ﰗ | fc17 | REMOVED |
| ﰘ | fc18 | REMOVED |
| ﰙ | fc19 | REMOVED |
| ﰚ | fc1a | REMOVED |
| ﰛ | fc1b | REMOVED |
| ﰜ | fc1c | REMOVED |
| ﰝ | fc1d | REMOVED |
| ﰟ | fc1f | REMOVED |
| ﰠ | fc20 | REMOVED |
| ﰡ | fc21 | REMOVED |
| ﰢ | fc22 | REMOVED |
| ﰤ | fc24 | REMOVED |
| ﰥ | fc25 | REMOVED |
| ﰦ | fc26 | REMOVED |
| ﰧ | fc27 | REMOVED |
| ﰨ | fc28 | REMOVED |
| ﰩ | fc29 | REMOVED |
| ﰪ | fc2a | REMOVED |
| ﰫ | fc2b | REMOVED |
| ﰬ | fc2c | REMOVED |
| ﰭ | fc2d | REMOVED |
| ﰮ | fc2e | REMOVED |
| ﰯ | fc2f | REMOVED |
| ﰰ | fc30 | REMOVED |
| ﰱ | fc31 | REMOVED |
| ﰲ | fc32 | REMOVED |
| ﰳ | fc33 | REMOVED |
| ﰴ | fc34 | REMOVED |
| ﰵ | fc35 | REMOVED |
| ﰶ | fc36 | REMOVED |
| ﰷ | fc37 | REMOVED |
| ﰸ | fc38 | REMOVED |
| ﰹ | fc39 | REMOVED |
| ﰺ | fc3a | REMOVED |
| ﰻ | fc3b | REMOVED |
| ﰼ | fc3c | REMOVED |
| ﰾ | fc3e | REMOVED |
| ﰿ | fc3f | REMOVED |
| ﱀ | fc40 | REMOVED |
| ﱁ | fc41 | REMOVED |
| ﱂ | fc42 | REMOVED |
| ﱃ | fc43 | REMOVED |
| ﱄ | fc44 | REMOVED |
| ﱅ | fc45 | REMOVED |
| ﱆ | fc46 | REMOVED |
| ﱇ | fc47 | REMOVED |
| ﱈ | fc48 | REMOVED |
| ﱉ | fc49 | REMOVED |
| ﱊ | fc4a | REMOVED |
| ﱋ | fc4b | REMOVED |
| ﱌ | fc4c | REMOVED |
| ﱍ | fc4d | REMOVED |
| ﱎ | fc4e | REMOVED |
| ﱏ | fc4f | REMOVED |
| ﱐ | fc50 | REMOVED |
| ﱑ | fc51 | REMOVED |
| ﱒ | fc52 | REMOVED |
| ﱓ | fc53 | REMOVED |
| ﱔ | fc54 | REMOVED |
| ﱕ | fc55 | REMOVED |
| ﱖ | fc56 | REMOVED |
| ﱗ | fc57 | REMOVED |
| ﱘ | fc58 | REMOVED |
| ﱙ | fc59 | REMOVED |
| ﱚ | fc5a | REMOVED |
| ﱛ | fc5b | REMOVED |
| ﱜ | fc5c | REMOVED |
| ﱝ | fc5d | REMOVED |
| ﱞ | fc5e | REMOVED |
| ﱟ | fc5f | REMOVED |
| ﱠ | fc60 | REMOVED |
| ﱡ | fc61 | REMOVED |
| ﱢ | fc62 | REMOVED |
| ﱣ | fc63 | REMOVED |
| ﱤ | fc64 | REMOVED |
| ﱥ | fc65 | REMOVED |
| ﱦ | fc66 | REMOVED |
| ﱧ | fc67 | REMOVED |
| ﱨ | fc68 | REMOVED |
| ﱩ | fc69 | REMOVED |
| ﱪ | fc6a | REMOVED |
| ﱫ | fc6b | REMOVED |
| ﱬ | fc6c | REMOVED |
| ﱮ | fc6e | REMOVED |
| ﱰ | fc70 | REMOVED |
| ﱱ | fc71 | REMOVED |
| ﱲ | fc72 | REMOVED |
| ﱳ | fc73 | REMOVED |
| ﱴ | fc74 | REMOVED |
| ﱵ | fc75 | REMOVED |
| ﱶ | fc76 | REMOVED |
| ﱷ | fc77 | REMOVED |
| ﱸ | fc78 | REMOVED |
| ﱹ | fc79 | REMOVED |
| ﱺ | fc7a | REMOVED |
| ﱻ | fc7b | REMOVED |
| ﱼ | fc7c | REMOVED |
| ﱽ | fc7d | REMOVED |
| ﱾ | fc7e | REMOVED |
| ﱿ | fc7f | REMOVED |
| ﲁ | fc81 | REMOVED |
| ﲄ | fc84 | REMOVED |
| ﲈ | fc88 | REMOVED |
| ﲉ | fc89 | REMOVED |
| ﲎ | fc8e | REMOVED |
| ﲏ | fc8f | REMOVED |
| ﲑ | fc91 | REMOVED |
| ﲒ | fc92 | REMOVED |
| ﲔ | fc94 | REMOVED |
| ﲗ | fc97 | REMOVED |
| ﲘ | fc98 | REMOVED |
| ﲙ | fc99 | REMOVED |
| ﲚ | fc9a | REMOVED |
| ﲜ | fc9c | REMOVED |
| ﲝ | fc9d | REMOVED |
| ﲟ | fc9f | REMOVED |
| ﲠ | fca0 | REMOVED |
| ﲡ | fca1 | REMOVED |
| ﲢ | fca2 | REMOVED |
| ﲧ | fca7 | REMOVED |
| ﲲ | fcb2 | REMOVED |
| ﲳ | fcb3 | REMOVED |
| ﲴ | fcb4 | REMOVED |
| ﲵ | fcb5 | REMOVED |
| ﲶ | fcb6 | REMOVED |
| ﲷ | fcb7 | REMOVED |
| ﲺ | fcba | REMOVED |
| ﲼ | fcbc | REMOVED |
| ﲽ | fcbd | REMOVED |
| ﳂ | fcc2 | REMOVED |
| ﳃ | fcc3 | REMOVED |
| ﳆ | fcc6 | REMOVED |
| ﳇ | fcc7 | REMOVED |
| ﳈ | fcc8 | REMOVED |
| ﳉ | fcc9 | REMOVED |
| ﳟ | fcdf | REMOVED |
| ﳠ | fce0 | REMOVED |
| ﳭ | fced | REMOVED |
| ﳮ | fcee | REMOVED |
| ﳻ | fcfb | REMOVED |
| ﴄ | fd04 | REMOVED |
| ﴑ | fd11 | REMOVED |
| � | fffd | REMOVED |

## 2. Sample Normalization
### Sample 1
**Raw:** `بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ ﰀ ٱلۡحَمۡدُ لِلَّهِ رَبِّ ٱلۡعَٰلَمِينَ ﰁ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ ﰂ مَٰلِكِ يَوۡمِ ٱلدِّينِ ﰃ إِيَّاكَ نَعۡبُدُ وَإِيَّاكَ نَسۡتَعِينُ ﰄ ٱهۡدِنَا ٱلصِّرَٰطَ ٱلۡمُسۡتَقِيمَ ﰅ`
**Normalized:** `بسم الله الرحمن الرحيم الحمد لله رب العلمين الرحمن الرحيم ملك يوم الدين اياك نعبد واياك نستعين اهدنا الصرط المستقيم`

---
### Sample 2
**Raw:** `صِرَٰطَ ٱلَّذِينَ أَنۡعَمۡتَ عَلَيۡهِمۡ غَيۡرِ ٱلۡمَغۡضُوبِ عَلَيۡهِمۡ وَلَا ٱلضَّآلِّينَ ﰆ الٓمٓ ﰀ ذَٰلِكَ ٱلۡكِتَٰبُ لَا رَيۡبَۛ فِيهِۛ هُدٗى لِّلۡمُتَّقِينَ ﰁ`
**Normalized:** `صرط الذين انعمت عليهم غير المغضوب عليهم ولا الضالين الم ذلك الكتب لا ريب فيه هدي للمتقين`

---
### Sample 3
**Raw:** `ٱلَّذِينَ يُؤۡمِنُونَ بِٱلۡغَيۡبِ وَيُقِيمُونَ ٱلصَّلَوٰةَ وَمِمَّا رَزَقۡنَٰهُمۡ يُنفِقُونَ ﰂ وَٱلَّذِينَ يُؤۡمِنُونَ بِمَآ أُنزِلَ إِلَيۡكَ وَمَآ أُنزِلَ مِن قَبۡلِكَ وَبِٱلۡأٓخِرَةِ هُمۡ يُوقِنُونَ ﰃ`
**Normalized:** `الذين يمنون بالغيب ويقيمون الصلوه ومما رزقنهم ينفقون والذين يمنون بما انزل اليك وما انزل من قبلك وبالاخره هم يوقنون`

---
### Sample 4
**Raw:** `أُوْلَٰٓئِكَ عَلَىٰ هُدٗى مِّن رَّبِّهِمۡۖ وَأُوْلَٰٓئِكَ هُمُ ٱلۡمُفۡلِحُونَ ﰄ  إِنَّ ٱلَّذِينَ كَفَرُواْ سَوَآءٌ عَلَيۡهِمۡ ءَأَنذَرۡتَهُمۡ أَمۡ لَمۡ تُنذِرۡهُمۡ لَا يُؤۡمِنُونَ ﰅ`
**Normalized:** `اولك علي هدي من ربهم واولك هم المفلحون ان الذين كفروا سوا عليهم انذرتهم ام لم تنذرهم لا يمنون`

---
### Sample 5
**Raw:** `خَتَمَ ٱللَّهُ عَلَىٰ قُلُوبِهِمۡ وَعَلَىٰ سَمۡعِهِمۡۖ وَعَلَىٰٓ أَبۡصَٰرِهِمۡ غِشَٰوَةٞۖ وَلَهُمۡ عَذَابٌ عَظِيمٞ ﰆ وَمِنَ ٱلنَّاسِ مَن يَقُولُ ءَامَنَّا بِٱللَّهِ وَبِٱلۡيَوۡمِ ٱلۡأٓخِرِ وَمَا هُم بِمُؤۡمِنِينَ ﰇ`
**Normalized:** `ختم الله علي قلوبهم وعلي سمعهم وعلي ابصرهم غشوه ولهم عذاب عظيم ومن الناس من يقول امنا بالله وباليوم الاخر وما هم بممنين`

---
### Sample 6
**Raw:** `لَّا مَقۡطُوعَةٖ وَلَا مَمۡنُوعَةٖ ﰠ وَفُرُشٖ مَّرۡفُوعَةٍ ﰡ إِنَّآ أَنشَأۡنَٰهُنَّ إِنشَآءٗ ﰢ فَجَعَلۡنَٰهُنَّ أَبۡكَارًا ﰣ`
**Normalized:** `لا مقطوعه ولا ممنوعه وفرش مرفوعه انا انشانهن انشا فجعلنهن ابكارا`

---
### Sample 7
**Raw:** `عُرُبًا أَتۡرَابٗا ﰤ لِّأَصۡحَٰبِ ٱلۡيَمِينِ ﰥ ثُلَّةٞ مِّنَ ٱلۡأَوَّلِينَ ﰦ وَثُلَّةٞ مِّنَ ٱلۡأٓخِرِينَ ﰧ وَأَصۡحَٰبُ ٱلشِّمَالِ مَآ أَصۡحَٰبُ ٱلشِّمَالِ ﰨ فِي سَمُومٖ وَحَمِيمٖ ﰩ`
**Normalized:** `عربا اترابا لاصحب اليمين ثله من الاولين وثله من الاخرين واصحب الشمال ما اصحب الشمال في سموم وحميم`

---
### Sample 8
**Raw:** `وَظِلّٖ مِّن يَحۡمُومٖ ﰪ لَّا بَارِدٖ وَلَا كَرِيمٍ ﰫ إِنَّهُمۡ كَانُواْ قَبۡلَ ذَٰلِكَ مُتۡرَفِينَ ﰬ وَكَانُواْ يُصِرُّونَ عَلَى ٱلۡحِنثِ ٱلۡعَظِيمِ ﰭ`
**Normalized:** `وظل من يحموم لا بارد ولا كريم انهم كانوا قبل ذلك مترفين وكانوا يصرون علي الحنث العظيم`

---
### Sample 9
**Raw:** `وَكَانُواْ يَقُولُونَ أَئِذَا مِتۡنَا وَكُنَّا تُرَابٗا وَعِظَٰمًا أَءِنَّا لَمَبۡعُوثُونَ ﰮ أَوَءَابَآؤُنَا ٱلۡأَوَّلُونَ ﰯ قُلۡ إِنَّ ٱلۡأَوَّلِينَ وَٱلۡأٓخِرِينَ ﰰ`
**Normalized:** `وكانوا يقولون اذا متنا وكنا ترابا وعظما انا لمبعوثون اوابانا الاولون قل ان الاولين والاخرين`

---
### Sample 10
**Raw:** `لَمَجۡمُوعُونَ إِلَىٰ مِيقَٰتِ يَوۡمٖ مَّعۡلُومٖ ﰱ ثُمَّ إِنَّكُمۡ أَيُّهَا ٱلضَّآلُّونَ ٱلۡمُكَذِّبُونَ ﰲ لَأٓكِلُونَ مِن شَجَرٖ مِّن زَقُّومٖ ﰳ`
**Normalized:** `لمجموعون الي ميقت يوم معلوم ثم انكم ايها الضالون المكذبون لاكلون من شجر من زقوم`

---
