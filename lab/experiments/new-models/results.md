# New Models Benchmark Results

Benchmark of ASR models for offline Quran verse identification.

## Overall Ranking

| Rank | Model | Type | Size (MB) | Accuracy | Avg Score | Avg Time (s) | Load Time (s) |
|------|-------|------|-----------|----------|-----------|--------------|---------------|
| 1 | Moonshine Tiny Arabic | moonshine | 103.4 | 3/7 | 0.751 | 1.26 | 11.9 |
| 2 | Whisper Large-v3-Turbo (HF) | whisper-hf | 3085.6 | 3/7 | 0.748 | 2.04 | 13.2 |
| 3 | Tarteel Whisper Base (mlx-whisper) | whisper-hf-tarteel | 276.9 | 3/7 | 0.727 | 2.08 | 13.3 |
| 4 | MMS-1B-All (Arabic) | wav2vec2-mms | 3680.4 | 3/7 | 0.724 | 4.01 | 36.0 |
| 5 | HamzaSidhu Quran ASR | wav2vec2-quran | 360.2 | 2/7 | 0.480 | 3.64 | 9.4 |
| 6 | Distil-Whisper Large-v3 | whisper-hf | 2885.5 | 0/7 | 0.000 | 1.66 | 14.2 |
| 7 | Nuwaisir Quran Recognizer | wav2vec2-quran | 1203.5 | 0/7 | 0.000 | 3.08 | 13.7 |
| 8 | SeamlessM4T-v2 Large | seamless-m4t | 5729.1 | 0/7 | 0.000 | 17.32 | 36.3 |

## Per-Test-Case Results

### An-Nisa:51 (user recording)

| Model | Correct? | Score | Time (s) | Transcription (clean) |
|-------|----------|-------|----------|----------------------|
| Whisper Large-v3-Turbo (HF) | No | 0.790 | 3.49 | الم تر الي الذين اوتوا نصيبا من الكتاب يدعون الي كتاب الله نحكم بينهم |
| Tarteel Whisper Base (mlx-whisper) | No | 0.747 | 4.00 | الم توان الذين اوتوا نصيبا من الكتاب يدعون الي كتاب الله ومحكم بينهم |
| Distil-Whisper Large-v3 | No | 0.000 | 1.99 | I'm tauten thou'em to the same t'em'en't'en'en'en'n't'n't |
| Moonshine Tiny Arabic | No | 0.734 | 1.73 | الم تر الي الذين اوتوا النصيب من الكتاب ودعونا الي كتاب الله معكم وبينه |
| SeamlessM4T-v2 Large | No | 0.000 | 16.64 |  |
| MMS-1B-All (Arabic) | No | 0.679 | 12.05 | لم ان الذين اوتون صلبا من الكتاب يدعون الي كتاب لا نحكم بينهم |
| Nuwaisir Quran Recognizer | No | 0.000 | 7.05 | >alamo/tara/<ila/Al~a*iynap/>ytuw/naSiybFA/mila/AlokitaAbi/mudoEFAla/<ilaY/kitaA |
| HamzaSidhu Quran ASR | No | 0.418 | 11.39 | لمكهنمفمصممملنكتفدعمئككلككمليم |

### Al-Ikhlas:2-3 (user, multi-ayah)

| Model | Correct? | Score | Time (s) | Transcription (clean) |
|-------|----------|-------|----------|----------------------|
| Whisper Large-v3-Turbo (HF) | No | 0.652 | 1.85 | الله الصمد لم يند ولم يوند |
| Tarteel Whisper Base (mlx-whisper) | No | 0.634 | 0.80 | الله الصمد لم يلت ولم يولت |
| Distil-Whisper Large-v3 | No | 0.000 | 1.66 | Allah islamad. |
| Moonshine Tiny Arabic | No | 0.634 | 0.19 | الله اصطمد لم ينج ولم يولد |
| SeamlessM4T-v2 Large | No | 0.000 | 1.51 |  |
| MMS-1B-All (Arabic) | No | 0.652 | 0.48 | الله الصمد لم يند ولم يوند |
| Nuwaisir Quran Recognizer | No | 0.000 | 0.20 | All~ahu/AlS~amato/lamo/yalido/walamo/yuwnado |
| HamzaSidhu Quran ASR | No | 0.514 | 0.27 | اللاظصمنللينتنمونت |

### Al-Fatiha:1 (ref)

| Model | Correct? | Score | Time (s) | Transcription (clean) |
|-------|----------|-------|----------|----------------------|
| Whisper Large-v3-Turbo (HF) | Yes | 0.978 | 1.68 | بسم الله الرحمن الرحيم |
| Tarteel Whisper Base (mlx-whisper) | Yes | 0.978 | 0.51 | بسم الله الرحمن الرحيم |
| Distil-Whisper Large-v3 | No | 0.000 | 1.62 | In the name of Allah, the |
| Moonshine Tiny Arabic | Yes | 0.978 | 0.21 | بسم الله الرحمن الرحيم |
| SeamlessM4T-v2 Large | No | 0.000 | 1.13 |  |
| MMS-1B-All (Arabic) | Yes | 0.936 | 0.35 | بباسم الله الرحمن الرحيم |
| Nuwaisir Quran Recognizer | No | 0.000 | 0.16 | bisomi/All~ahi/Alr~aHomani/Alr~aHiymm |
| HamzaSidhu Quran ASR | Yes | 0.571 | 0.12 | بسملانحمانرح |

### Al-Fatiha:2 (ref)

| Model | Correct? | Score | Time (s) | Transcription (clean) |
|-------|----------|-------|----------|----------------------|
| Whisper Large-v3-Turbo (HF) | Yes | 0.976 | 1.70 | الحمد لله رب العالمين |
| Tarteel Whisper Base (mlx-whisper) | Yes | 0.976 | 0.55 | الحمد لله رب العالمين |
| Distil-Whisper Large-v3 | No | 0.000 | 1.68 | Al-Lahmdhav-Lah-N-N-N-N-N- |
| Moonshine Tiny Arabic | Yes | 0.976 | 0.21 | الحمد لله رب العالمين |
| SeamlessM4T-v2 Large | No | 0.000 | 0.88 |  |
| MMS-1B-All (Arabic) | Yes | 0.884 | 0.39 | احالحم لله رب العا لمين |
| Nuwaisir Quran Recognizer | No | 0.000 | 0.13 | AloHamodu/lil~ahi/rab~i/AloEaAlamiy |
| HamzaSidhu Quran ASR | No | 0.581 | 0.10 | الحنرلارهلعلنين |

### Ayat al-Kursi (ref)

| Model | Correct? | Score | Time (s) | Transcription (clean) |
|-------|----------|-------|----------|----------------------|
| Whisper Large-v3-Turbo (HF) | Yes | 0.735 | 2.33 | الله لا اله الا هو الحي القيوم لا تاخذه سنه ولا نوم له ما في السماوات وما في الا |
| Tarteel Whisper Base (mlx-whisper) | Yes | 0.748 | 8.04 | الله لا اله الا هو الحي القيوم لا تاخذه سنه ولا نوم له ما في السماوات وما في الا |
| Distil-Whisper Large-v3 | No | 0.000 | 1.60 | Allah, Allah, Allah, Allah, Allah, |
| Moonshine Tiny Arabic | Yes | 0.744 | 6.16 | الله لا اله الا هو الحي القيوم لا تاخذه سنه ولا نوم له ما في السماوات وما في الا |
| SeamlessM4T-v2 Large | No | 0.000 | 59.31 |  |
| MMS-1B-All (Arabic) | Yes | 0.945 | 13.88 | الل لا الها الا هو الحي القي لا تاخذه سنه ولانوو لهما في السماوات وما في الارض م |
| Nuwaisir Quran Recognizer | No | 0.000 | 11.19 | ll~ahu/laA/<ilahaA/<i/l~ahua/AloHay~u/Aloqay~umK/laA/ta>oxu*uw/sinatuwwalaA/lamK |
| HamzaSidhu Quran ASR | Yes | 0.430 | 12.88 | الخلالاكائاحيلقلنتونتوننمفسماتوماملرييشعدكلابريعلنمابندنماخفلاحقونبشكننعلنئلابنش |

### Al-Ikhlas:1 (ref)

| Model | Correct? | Score | Time (s) | Transcription (clean) |
|-------|----------|-------|----------|----------------------|
| Whisper Large-v3-Turbo (HF) | No | 0.606 | 1.65 | قل هو الله احد |
| Tarteel Whisper Base (mlx-whisper) | No | 0.606 | 0.47 | قل هو الله احد |
| Distil-Whisper Large-v3 | No | 0.000 | 1.52 | Tell, |
| Moonshine Tiny Arabic | No | 0.606 | 0.18 | قل هو الله احد |
| SeamlessM4T-v2 Large | No | 0.000 | 31.59 |  |
| MMS-1B-All (Arabic) | No | 0.609 | 0.61 | قل ه الله احد |
| Nuwaisir Quran Recognizer | No | 0.000 | 2.65 | qulo/huwa/All~ahu/>aHadN |
| HamzaSidhu Quran ASR | No | 0.385 | 0.61 | قلظرؤحد |

### Ya-Sin:1 (ref)

| Model | Correct? | Score | Time (s) | Transcription (clean) |
|-------|----------|-------|----------|----------------------|
| Whisper Large-v3-Turbo (HF) | No | 0.500 | 1.59 | ياسين |
| Tarteel Whisper Base (mlx-whisper) | No | 0.400 | 0.19 | يس |
| Distil-Whisper Large-v3 | No | 0.000 | 1.54 | Yesin. |
| Moonshine Tiny Arabic | No | 0.588 | 0.11 | يا سين |
| SeamlessM4T-v2 Large | No | 0.000 | 10.20 |  |
| MMS-1B-All (Arabic) | No | 0.364 | 0.29 | يا |
| Nuwaisir Quran Recognizer | No | 0.000 | 0.15 | yaA/siy |
| HamzaSidhu Quran ASR | No | 0.462 | 0.12 | سين |

## Speed Comparison

| Model | Avg Inference (s) | Load Time (s) | Size (MB) |
|-------|-------------------|---------------|-----------|
| Moonshine Tiny Arabic | 1.26 | 11.9 | 103.4 |
| Distil-Whisper Large-v3 | 1.66 | 14.2 | 2885.5 |
| Whisper Large-v3-Turbo (HF) | 2.04 | 13.2 | 3085.6 |
| Tarteel Whisper Base (mlx-whisper) | 2.08 | 13.3 | 276.9 |
| Nuwaisir Quran Recognizer | 3.08 | 13.7 | 1203.5 |
| HamzaSidhu Quran ASR | 3.64 | 9.4 | 360.2 |
| MMS-1B-All (Arabic) | 4.01 | 36.0 | 3680.4 |
| SeamlessM4T-v2 Large | 17.32 | 36.3 | 5729.1 |
