# Embedding Search Benchmark Results

Generated: 2026-02-24 10:49:45

## Summary

| Model | Index Size | Category | Top-1 | Top-3 | Top-5 | Surah@1 | Avg Time |
|-------|-----------|----------|-------|-------|-------|---------|----------|
| hubert | 6236 | All | 71.4% | 71.4% | 71.4% | 71.4% | 376.7ms |
| | | Reference | 100.0% | 100.0% | 100.0% | - | - |
| | | User | 0.0% | 0.0% | 0.0% | 0.0% | - |

## Detailed Results

### hubert / alafasy

- **An-Nisa:51 (user recording)** (expected 4:51): MISS
  - Time: 1233.2ms (embed: 1228.9ms, search: 4.259ms)
  - #1: Al-Qalam 68:39 (score: 0.8605)
  - #2: Al-Qasas 28:2 (score: 0.8597)
  - #3: Al-Anbiyaa 21:1 (score: 0.8592)
  - #4: Al-Hijr 15:55 (score: 0.8585)
  - #5: Al-Maaida 5:26 (score: 0.8577)
- **Al-Ikhlas:2-3 (user recording, multi-verse)** (expected 112:2): MISS
  - Time: 133.3ms (embed: 132.7ms, search: 0.545ms)
  - #1: An-Najm 53:62 (score: 0.7208)
  - #2: Al-Ikhlaas 112:4 (score: 0.6744)
  - #3: Al-Qadr 97:5 (score: 0.6731)
  - #4: Al-Alaq 96:19 (score: 0.6715)
  - #5: Al-Qaari'a 101:11 (score: 0.6691)
- **Al-Fatiha:1 (ref)** (expected 1:1): TOP-1
  - Time: 97.1ms (embed: 96.6ms, search: 0.457ms)
  - #1: Al-Faatiha 1:1 (score: 1.0) **<<<**
  - #2: Az-Zumar 39:66 (score: 0.9379)
  - #3: As-Saaffaat 37:182 (score: 0.9355)
  - #4: Yaseen 36:58 (score: 0.9326)
  - #5: Az-Zumar 39:40 (score: 0.9315)
- **Al-Fatiha:2 (ref)** (expected 1:2): TOP-1
  - Time: 215.2ms (embed: 214.6ms, search: 0.62ms)
  - #1: Al-Faatiha 1:2 (score: 1.0) **<<<**
  - #2: As-Saaffaat 37:76 (score: 0.9453)
  - #3: Al-Hijr 15:17 (score: 0.9419)
  - #4: As-Saaffaat 37:40 (score: 0.9417)
  - #5: Al-Anbiyaa 21:106 (score: 0.94)
- **Ayat al-Kursi (ref)** (expected 2:255): TOP-1
  - Time: 673.5ms (embed: 672.8ms, search: 0.701ms)
  - #1: Al-Baqara 2:255 (score: 0.974) **<<<**
  - #2: At-Talaaq 65:5 (score: 0.9679)
  - #3: At-Tawba 9:82 (score: 0.9671)
  - #4: Hud 11:19 (score: 0.9669)
  - #5: Ar-Room 30:9 (score: 0.9662)
- **Al-Ikhlas:1 (ref)** (expected 112:1): TOP-1
  - Time: 168.4ms (embed: 167.5ms, search: 0.9ms)
  - #1: Al-Ikhlaas 112:1 (score: 1.0) **<<<**
  - #2: Al-Humaza 104:6 (score: 0.9291)
  - #3: Al-Ikhlaas 112:2 (score: 0.9251)
  - #4: Al-Muddaththir 74:17 (score: 0.8981)
  - #5: An-Naazi'aat 79:24 (score: 0.8884)
- **Ya-Sin:1 (ref)** (expected 36:1): TOP-1
  - Time: 116.5ms (embed: 115.9ms, search: 0.615ms)
  - #1: Yaseen 36:1 (score: 1.0) **<<<**
  - #2: Ad-Dukhaan 44:46 (score: 0.9144)
  - #3: Al-Mursalaat 77:45 (score: 0.9113)
  - #4: Fussilat 41:1 (score: 0.9005)
  - #5: Al-Mursalaat 77:37 (score: 0.8968)
