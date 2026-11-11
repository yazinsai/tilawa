# X post drafts

**Draft only. Not for posting** without the account owner's explicit go-ahead.

Status: the release label is pending ({{PENDING: decision 6}}). These drafts say "preview" where it matters. If 1.0.0 ships as stable, drop "preview" from C.

Rules these follow:

- No accuracy or latency numbers. If we add any, they must come from the release report ({{RELEASE_NUMBERS}}).
- Only claims the release supports: following recitation, and memorization feedback (wrong, missed and added words, skipped passages). Nothing about pronunciation or tajweed.
- "On your device" refers to the engine. /help sends a recording only when the reader chooses to.

Character counts use X's rule that every link counts as 23 characters. Check them again in the composer before posting.

Before posting, check in the composer that `npmjs.com/package/@tilawa/core` stays one link: the `@` may break it. If it does, use `npmjs.com/package/%40tilawa/core`. Don't write `@tilawa` outside a link: it would tag an X account.

Attach the launch video ([video-script.md](video-script.md)) or a screenshot of the demo following a recitation.

## A: the product, all links (279/280)

```text
Tilawa follows your Quran recitation word by word, on your device, and flags memorization slips: wrong, missed or added words, and skipped passages.

Try it: tilawa.dev
Help it learn: tilawa.dev/help
npm: npmjs.com/package/@tilawa/core
Docs: github.com/yazinsai/tilawa
```

## B: the contribution loop (276/280)

```text
Recite any part of the Quran. Tilawa finds your place and follows you, word by word, on your device.

Help it learn: recite for a few minutes at tilawa.dev/help and tell us where it got it wrong.

Try it: tilawa.dev
For developers: github.com/yazinsai/tilawa
```

## C: for developers (259/280)

```text
The Tilawa SDK 1.0 preview: on-device Quran recitation tracking and memorization feedback, for web and Node. Audio isn't uploaded.

Demo: tilawa.dev
Docs: github.com/yazinsai/tilawa/tree/main/docs
npm: npmjs.com/package/@tilawa/core
Help it learn: tilawa.dev/help
```

## Optional reply (thread)

```text
How it works: a small speech model turns your voice into sounds (phonemes) in the browser, and Tilawa matches them to the Quran to find your place. At tilawa.dev/help, readers tell us where it got things wrong. That is how it improves.
```

Check that this reply is still accurate for the shipped demo and /help before using it.
