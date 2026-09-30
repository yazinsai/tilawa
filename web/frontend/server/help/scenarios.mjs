// Research-informed examples, not measured population frequencies.
// See docs/recitation-mistake-prompts.md for evidence and review limitations.
// Word indices address loadQuran's waqf-marker-free Uthmani words.
export const SCENARIOS = [
  // A familiar parallel passage supplies the wrong, otherwise plausible word.
  { id: 'parallel-send', kind: 'substitution', mechanism: 'similar_passage', ref: '7:111', word: 4, source: '26:36', sourceWord: 4 },
  { id: 'parallel-dispatch', kind: 'substitution', mechanism: 'similar_passage', ref: '26:36', word: 4, source: '7:111', sourceWord: 4 },
  { id: 'parallel-revealed', kind: 'substitution', mechanism: 'similar_passage', ref: '12:2', word: 2, source: '43:3', sourceWord: 2 },
  { id: 'parallel-made', kind: 'substitution', mechanism: 'similar_passage', ref: '43:3', word: 2, source: '12:2', sourceWord: 2 },
  { id: 'parallel-prophet-nooh', kind: 'substitution', mechanism: 'similar_passage', ref: '26:106', word: 5, source: '26:124', sourceWord: 5 },
  { id: 'parallel-prophet-hud', kind: 'substitution', mechanism: 'similar_passage', ref: '26:124', word: 5, source: '26:142', sourceWord: 5 },
  { id: 'parallel-fire', kind: 'substitution', mechanism: 'similar_passage', ref: '79:39', word: 2, source: '79:41', sourceWord: 2 },
  { id: 'parallel-garden', kind: 'substitution', mechanism: 'similar_passage', ref: '79:41', word: 2, source: '79:39', sourceWord: 2 },
  // Qaf/kaf confusion has direct (small-sample) evidence in Al-Falaq learners.
  { id: 'qaf-kaf-falaq', kind: 'substitution', mechanism: 'letter_confusion', ref: '113:1', word: 8, fromLetter: 'ق', toLetter: 'ك' },
  { id: 'qaf-kaf-created', kind: 'substitution', mechanism: 'letter_confusion', ref: '113:2', word: 4, fromLetter: 'ق', toLetter: 'ك' },
  // Short grammatical words / repeated phrase positions, not arbitrary nouns.
  { id: 'omit-second-iyyaka', kind: 'skip_word', mechanism: 'repeated_phrase_omission', ref: '1:5', word: 3 },
  { id: 'omit-negation-rayb', kind: 'skip_word', mechanism: 'short_word_omission', ref: '2:2', word: 3 },
  { id: 'omit-negation-warning', kind: 'skip_word', mechanism: 'short_word_omission', ref: '2:6', word: 8 },
  { id: 'omit-with-hardship', kind: 'skip_word', mechanism: 'short_word_omission', ref: '94:5', word: 2 },
  // Fluency controls: a repetition is not automatically a tajweed violation.
  { id: 'hesitate-iyyaka', kind: 'repeat', mechanism: 'hesitation_repeat', ref: '1:5', word: 3 },
  { id: 'hesitate-hardship', kind: 'repeat', mechanism: 'hesitation_repeat', ref: '94:5', word: 2 },
  { id: 'hesitate-counsel', kind: 'repeat', mechanism: 'hesitation_repeat', ref: '103:3', word: 6 },
  { id: 'hesitate-lord', kind: 'repeat', mechanism: 'hesitation_repeat', ref: '93:3', word: 3 },
  // Specify the exact letter. Final taa here is audible in continuous reading.
  { id: 'vowel-anamta', kind: 'vowel', mechanism: 'person_ending', ref: '1:7', word: 3, letter: 'ت', from: 'fatha', to: 'damma', continue: true },
  { id: 'vowel-nabudu', kind: 'vowel', mechanism: 'short_vowel_confusion', ref: '1:5', word: 2, letter: 'ن', from: 'fatha', to: 'damma' },
  { id: 'vowel-rayb', kind: 'vowel', mechanism: 'short_vowel_confusion', ref: '2:2', word: 4, letter: 'ر', from: 'fatha', to: 'kasra' },
  { id: 'vowel-sharr', kind: 'vowel', mechanism: 'short_vowel_confusion', ref: '114:4', word: 2, letter: 'ش', from: 'fatha', to: 'kasra' },
  // Only unambiguous internal madd lazim, not variable verse-final lengthening.
  { id: 'madd-dalleen', kind: 'tajweed', mechanism: 'madd_duration', ref: '1:7', word: 9, rule: 'madd_short' },
  { id: 'ghunna-inna', kind: 'tajweed', mechanism: 'ghunna_duration', ref: '2:6', word: 1, rule: 'no_ghunna', letter: 'ن' },
  { id: 'ghunna-amma', kind: 'tajweed', mechanism: 'ghunna_duration', ref: '78:1', word: 5, rule: 'no_ghunna', letter: 'م' },
  { id: 'ghunna-khannas', kind: 'tajweed', mechanism: 'ghunna_duration', ref: '114:4', word: 4, rule: 'no_ghunna', letter: 'ن' },
  { id: 'light-sad-samad', kind: 'tajweed', mechanism: 'emphasis_loss', ref: '112:2', word: 2, rule: 'light_emphatic', letter: 'ص' },
  { id: 'light-sad-sudur', kind: 'tajweed', mechanism: 'emphasis_loss', ref: '114:5', word: 4, rule: 'light_emphatic', letter: 'ص' },
  { id: 'shadda-khannas', kind: 'tajweed', mechanism: 'gemination_loss', ref: '114:4', word: 4, rule: 'no_shadda', letter: 'ن' },
  // Repeated endings / parallel neighbouring verses create position confusion.
  { id: 'skip-nas-title', kind: 'skip_ayah', mechanism: 'similar_ending_omission', ref: '114:1', skip: '114:2', end: '114:3' },
  { id: 'skip-kafirun-parallel', kind: 'skip_ayah', mechanism: 'parallel_verse_omission', ref: '109:2', skip: '109:3', end: '109:4' },
  { id: 'skip-kafirun-middle', kind: 'skip_ayah', mechanism: 'parallel_verse_omission', ref: '109:3', skip: '109:4', end: '109:5' },
  { id: 'skip-hardship-parallel', kind: 'skip_ayah', mechanism: 'parallel_verse_omission', ref: '94:4', skip: '94:5', end: '94:6' },
];
