# QUD alignment benchmark v1: posteriors

Report v1, reference ref-v2, commit 26acd8f. Posterior reranking: on. Streamed in 1 s chunks.

16 recordings, 357 min, 39 passage starts.

| metric | value |
|---|---|
| correct ayah within 5 s | 27/39 69.2% |
| starts with a false lock first | 4/39 10.3% |
| unresolved starts | 0/39 0.0% |
| locked time on a wrong ayah | 1.76% |
| false jumps | 9/100 9.0% |
| shortlist recall | 52.5% |
| median time to first lock | 4.17 s |
| shared-opening starts | 36 |
| median lock delay after the distinguishing word | 0.65 s |
| real-time factor (mean / max) | 0.0023 / 0.0065 |

| recording | style | min | starts in 5 s | false-lock starts | unresolved | wrong locked time | false jumps | rtf |
|---|---|---|---|---|---|---|---|---|
| abdulbasit-raad | mujawwad | 38 | 0/1 | 0 | 0 | 0.6% | 0/0 | 0.0004 |
| afasy-baqarah | hadr | 80 | 0/1 | 1 | 0 | 0.4% | 1/2 | 0.0014 |
| ayyub-juz-amma | murattal | 58 | 1/1 | 0 | 0 | 1.2% | 1/2 | 0.0018 |
| badri-juz-tabarak | hadr | 24 | 1/1 | 0 | 0 | 0.3% | 0/0 | 0.0013 |
| diban-sad | hadr | 4 | 1/1 | 0 | 0 | 0.7% | 1/3 | 0.0053 |
| husary-fath | mujawwad | 25 | 1/1 | 0 | 0 | 1.0% | 0/0 | 0.0004 |
| jaber-inshiqaq | murattal | 2 | 1/1 | 0 | 0 | 0.0% | 0/0 | 0.0011 |
| jaber-nisa | hadr | 1 | 1/1 | 0 | 0 | 0.4% | 0/0 | 0.0024 |
| luhaidan-fatir | murattal, noisy | 18 | 1/1 | 1 | 0 | 10.1% | 6/14 | 0.0045 |
| luhaidan-haqqah | murattal, noisy | 2 | 0/1 | 0 | 0 | 16.5% | 0/1 | 0.0031 |
| luhaidan-qiyamah | murattal, noisy | 4 | 0/2 | 0 | 0 | 4.8% | 0/1 | 0.0026 |
| mandour-taha | murattal, noisy | 21 | 1/1 | 0 | 0 | 0.7% | 0/0 | 0.0006 |
| minshawi-luqman | muallim, noisy | 19 | 0/1 | 1 | 0 | 2.3% | 0/1 | 0.0008 |
| prayer-baqarah-yasin | murattal | 13 | 4/4 | 0 | 0 | 2.7% | 0/6 | 0.0027 |
| taraweeh-maryam | murattal | 34 | 9/15 | 0 | 0 | 3.1% | 0/33 | 0.0023 |
| witr-muawwidhat | murattal | 12 | 6/6 | 1 | 0 | 15.7% | 0/37 | 0.0065 |
