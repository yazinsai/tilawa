# QUD alignment benchmark v1: decoded

Report v1, reference ref-v2, commit aa67a15. Posterior reranking: off. Phrases: phrases-v1. Streamed in 1 s chunks.

16 recordings, 357 min, 39 passage starts.

| metric | value |
|---|---|
| correct ayah within 5 s | 30/39 76.9% |
| starts with a false lock first | 2/39 5.1% |
| unresolved starts | 0/39 0.0% |
| locked time on a wrong ayah | 1.72% |
| false jumps | 8/91 8.8% |
| shortlist recall | 53.6% |
| median time to first lock | 4.11 s |
| shared-opening starts | 36 |
| median lock delay after the distinguishing word | 0.08 s |
| real-time factor (mean / max) | 0.0037 / 0.0086 |

| recording | style | min | starts in 5 s | false-lock starts | unresolved | wrong locked time | false jumps | rtf |
|---|---|---|---|---|---|---|---|---|
| abdulbasit-raad | mujawwad | 38 | 0/1 | 0 | 0 | 0.6% | 0/0 | 0.0010 |
| afasy-baqarah | hadr | 80 | 0/1 | 0 | 0 | 0.1% | 0/1 | 0.0039 |
| ayyub-juz-amma | murattal | 58 | 1/1 | 0 | 0 | 2.3% | 0/1 | 0.0020 |
| badri-juz-tabarak | hadr | 24 | 1/1 | 0 | 0 | 0.0% | 0/0 | 0.0037 |
| diban-sad | hadr | 4 | 1/1 | 0 | 0 | 0.7% | 1/3 | 0.0086 |
| husary-fath | mujawwad | 25 | 1/1 | 0 | 0 | 1.0% | 0/0 | 0.0012 |
| jaber-inshiqaq | murattal | 2 | 1/1 | 0 | 0 | 0.0% | 0/0 | 0.0018 |
| jaber-nisa | hadr | 1 | 1/1 | 0 | 0 | 0.4% | 0/0 | 0.0057 |
| luhaidan-fatir | murattal, noisy | 18 | 1/1 | 1 | 0 | 9.8% | 6/14 | 0.0059 |
| luhaidan-haqqah | murattal, noisy | 2 | 0/1 | 0 | 0 | 16.5% | 0/1 | 0.0041 |
| luhaidan-qiyamah | murattal, noisy | 4 | 1/2 | 0 | 0 | 1.4% | 0/1 | 0.0031 |
| mandour-taha | murattal, noisy | 21 | 0/1 | 0 | 0 | 0.7% | 0/0 | 0.0021 |
| minshawi-luqman | muallim, noisy | 19 | 0/1 | 0 | 0 | 1.6% | 0/0 | 0.0025 |
| prayer-baqarah-yasin | murattal | 13 | 4/4 | 0 | 0 | 2.5% | 0/3 | 0.0032 |
| taraweeh-maryam | murattal | 34 | 12/15 | 1 | 0 | 3.8% | 1/32 | 0.0033 |
| witr-muawwidhat | murattal | 12 | 6/6 | 0 | 0 | 5.8% | 0/35 | 0.0079 |
