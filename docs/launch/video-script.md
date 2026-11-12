# Launch video: shot list (30–60 s)

**Draft.** A short screen recording of the real product, for X and the release page.

## Ground rules

- **Real audio only.** The recitation must be the presenter's own, or recorded with the reciter's written permission. No recordings from apps, broadcasts or datasets without permission.
- **Real output only.** Record the live demo and the live /help page on the release candidate. No edited or mocked UI.
- **A genuine correction.** The mistake in shot 3 has to be caught by the engine during that take. If you make the slip on purpose (scripted), label the video "Demonstration: the reader slips on purpose". Retake until the engine catches it; don't splice in a catch from another take.
- **No numbers** on screen or in captions unless they come from the model card's Evaluation section.
- **Say "preview"** (the release is labelled Preview).
- Show nothing personal: hide bookmarks, tabs, notifications and other people's recordings. Blur the delete code on the success screen.
- Pronunciation and tajweed are not supported. Don't show or imply them.

## Setup

- Phone (iOS Safari or Android Chrome), recording the screen with the microphone on, in a quiet room. Or a laptop, if the phone recording drops audio.
- Pick a passage the reader knows well, for example the start of Surah Al-Mulk (67:1–5).
- Load tilawa.dev once before recording, so the model is cached and the take starts quickly.
- A code editor with the quickstart open for shot 5.

## Shots

| # | Time | Screen | Audio | Caption |
|---|---|---|---|---|
| 1 | 0–6 s | tilawa.dev on a phone. Tap **Start**. | Room tone. | Tilawa follows your Quran recitation, on your device. |
| 2 | 6–18 s | The reader starts reciting. After a few words the ayah appears, and the current word is highlighted and moves along with the recitation. | The reader's recitation. | Start anywhere. It finds your place and follows word by word. |
| 3 | 18–30 s | The reader slips: drops or changes a word. A few seconds later the note appears on that word, for example "Missed word". | Recitation, with the slip. | When you slip, it notices. |
| 4 | 30–42 s | tilawa.dev/help: tap **Stop**, play back a marked moment, answer "Did you make a mistake here?", tap **Send**. Show the thank-you screen (delete code blurred). | Short playback clip, then quiet. | Tell us where it got it wrong. That's how it learns. |
| 5 | 42–55 s | A code editor: the quickstart (`createTilawa()`, `startFreeRecitation()`, `subscribe`, `listen`). Cut to the npm install line. | Quiet, or soft room tone. No music over the recitation. | For developers: npm i @tilawa/core@next |
| 6 | 55–60 s | End card: tilawa.dev · tilawa.dev/help · github.com/yazinsai/tilawa | None. | Preview release. Try it, and help it learn. |

If the take runs short, drop shot 5 to a 5-second cut of the install line and keep the end card.

## Captions (`.srt` draft)

Burn them in for X (most people watch muted), and also upload the file. Adjust the times to the final edit. The Arabic recitation itself isn't captioned; if you add the ayah text, take it verbatim from a Uthmani mushaf text (Tanzil) and credit it.

```srt
1
00:00:00,000 --> 00:00:06,000
Tilawa follows your Quran recitation, on your device.

2
00:00:06,000 --> 00:00:18,000
Start anywhere. It finds your place
and follows word by word.

3
00:00:18,000 --> 00:00:30,000
When you slip, it notices.

4
00:00:30,000 --> 00:00:42,000
Tell us where it got it wrong.
That's how it learns.

5
00:00:42,000 --> 00:00:55,000
For developers: npm i @tilawa/core@next

6
00:00:55,000 --> 00:01:00,000
Preview release. Try it, and help it learn.
tilawa.dev · tilawa.dev/help
```

If shot 3 is scripted, add this caption for the length of the shot:

```srt
3
00:00:18,000 --> 00:00:30,000
Demonstration: the reader slips on purpose.
When you slip, it notices.
```

## Deliverables

- `tilawa-launch.mp4`, 1080×1920 (vertical) and 1920×1080, 30 to 60 seconds.
- `tilawa-launch.en.srt`.
- Optional thumbnail: the demo with an ayah highlighted.
- A note of the reciter's permission, kept with the source files.
