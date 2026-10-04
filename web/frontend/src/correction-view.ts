import type { CorrectionAction, CorrectionState } from '@tilawa/core';

export interface PracticeVerse {
  words: string[];
  wordOffset?: number;
  name: string;
  nameEn: string;
  ayahCount: number;
}

/** Presentation only. Every state comes from the SDK's acoustic evidence. */
export class CorrectionView {
  readonly dialog = document.createElement('dialog');
  private previousFocus: HTMLElement | null = null;
  private state: CorrectionState | null = null;
  private verse: PracticeVerse | null = null;
  private arabic = false;
  private action: (action: CorrectionAction) => void;
  constructor(action: (action: CorrectionAction) => void) {
    this.action = action;
    this.dialog.className = 'correction-focus';
    this.dialog.setAttribute('aria-labelledby', 'practice-surah');
    this.dialog.innerHTML = `
      <header class="practice-nav"><span class="practice-brand">tilawa</span><button class="text-btn" data-action="close"></button></header>
      <div class="practice-body">
        <header class="practice-header"><p id="practice-status" role="status"></p><h2 id="practice-surah"></h2><p id="practice-meta"></p></header>
        <p id="practice-verse" dir="rtl" lang="ar"></p>
        <section class="practice-panel" aria-live="polite"><h3 id="practice-title"></h3><p id="practice-description"></p>
          <div class="practice-actions"><button class="primary-btn" id="practice-primary"></button><button class="text-btn" id="practice-secondary"></button></div>
        </section>
      </div>`;
    document.body.append(this.dialog);
    this.dialog.addEventListener('cancel', e => { e.preventDefault(); this.action('close'); });
    this.dialog.addEventListener('click', e => {
      const button = (e.target as Element).closest<HTMLButtonElement>('button[data-action]');
      if (button?.dataset.action) this.action(button.dataset.action as CorrectionAction);
    });
  }
  get open(): boolean { return this.dialog.open; }
  show(state: CorrectionState, verse: PracticeVerse, arabic: boolean): void {
    this.state = state; this.verse = verse; this.arabic = arabic;
    if (state.phase === 'idle') { this.close(); return; }
    this.render();
    if (!this.dialog.open) {
      this.previousFocus = document.activeElement as HTMLElement;
      this.dialog.showModal();
      document.body.classList.add('practicing');
      this.dialog.querySelector<HTMLButtonElement>('#practice-primary')!.focus();
    }
  }
  language(arabic: boolean): void { this.arabic = arabic; if (this.state && this.verse) this.render(); }
  close(): void {
    if (!this.dialog.open) return;
    this.dialog.close();
    document.body.classList.remove('practicing');
    this.previousFocus?.focus({ preventScroll: true });
  }
  private render(): void {
    const { phase, issue } = this.state!;
    if (!issue) return;
    const verse = this.verse!;
    const tr = (en: string, ar: string) => this.arabic ? ar : en;
    const num = (n: number) => this.arabic ? String(n).replace(/\d/g, d => '٠١٢٣٤٥٦٧٨٩'[Number(d)]) : String(n);
    const set = (id: string, text: string) => { this.dialog.querySelector<HTMLElement>(`#${id}`)!.textContent = text; };
    this.dialog.dir = this.arabic ? 'rtl' : 'ltr';
    this.dialog.dataset.phase = phase;
    this.dialog.dataset.kind = issue.kind;
    const vowel = issue.kind === 'possible_vowel';
    const repeated = issue.kind === 'possible_repetition';
    const skippedAyah = issue.kind === 'possible_skipped_ayah';
    const unclearAyah = issue.kind === 'unclear_ayah';
    const wholeAyah = skippedAyah || unclearAyah;
    this.dialog.querySelector('[data-action="close"]')!.textContent = tr('Close practice', 'إغلاق التدريب');
    set('practice-status', phase === 'retrying' ? tr('RETRYING · MICROPHONE ON', 'نستمع لمحاولتك · الميكروفون يعمل')
      : phase === 'corrected' ? tr('RETRY COMPLETE', 'اكتملت المحاولة')
      : vowel ? tr('FOCUSED PRACTICE · POSSIBLE VOWEL SLIP', 'تدريب مركّز · خطأ محتمل في الحركة')
      : repeated ? tr('FOCUSED PRACTICE · WORD REPEATED', 'تدريب مركّز · كلمة مكرّرة')
      : skippedAyah ? tr('FOCUSED PRACTICE · POSSIBLE SKIPPED AYAH', 'تدريب مركّز · آية ربما سقطت')
      : unclearAyah ? tr('FOCUSED PRACTICE · AYAH NOT FOLLOWED', 'تدريب مركّز · لم نتمكّن من متابعة الآية')
      : tr('FOCUSED PRACTICE · POSSIBLE MISTAKE', 'تدريب مركّز · خطأ محتمل'));
    set('practice-surah', this.arabic ? verse.name : verse.nameEn);
    set('practice-meta', tr(`Surah ${issue.surah} · Ayah ${issue.ayah} of ${verse.ayahCount}`, `سورة ${num(issue.surah)} · الآية ${num(issue.ayah)} من ${num(verse.ayahCount)}`));
    const phrase = this.dialog.querySelector('#practice-verse')!;
    phrase.replaceChildren();
    // Display the original full ayah, including all diacritics and stop marks.
    verse.words.forEach((word, index) => {
      const span = document.createElement('span'); span.textContent = word;
      // Ayah-level issues highlight the whole ayah (the bismillah prefix stays plain).
      if (wholeAyah ? index >= (verse.wordOffset ?? 0) : index === issue.word + (verse.wordOffset ?? 0)) span.className = 'practice-word';
      phrase.append(span, document.createTextNode(index < verse.words.length - 1 ? ' ' : ''));
    });
    set('practice-title', phase === 'retrying' ? tr('Take your time.', 'خذ وقتك.') : phase === 'corrected'
      ? tr('That’s corrected.', 'تمّ التصحيح.')
      : vowel ? tr('Check the vowel on this word.', 'راجع حركة هذه الكلمة.')
      : repeated ? tr('This word was said twice.', 'تكرّرت هذه الكلمة.')
      : skippedAyah ? tr(`Ayah ${issue.ayah} may have been skipped.`, `ربما سقطت الآية ${num(issue.ayah)}.`)
      : unclearAyah ? tr(`We couldn't follow ayah ${issue.ayah}. Recite it again.`, `لم نتمكّن من متابعة الآية ${num(issue.ayah)}. أعد تلاوتها.`)
      : tr('One word. Try again.', 'كلمة واحدة. حاول مجددًا.'));
    set('practice-description', phase === 'retrying'
      ? (wholeAyah ? tr(`Repeat ayah ${issue.ayah} from the beginning. We’ll listen for the whole ayah.`, `أعد الآية ${num(issue.ayah)} من بدايتها. سنستمع إلى الآية كاملة.`)
        : tr(`Repeat ayah ${issue.ayah} from the beginning. We’ll check the highlighted word again.`, `أعد الآية ${num(issue.ayah)} من بدايتها. سنتحقّق من الكلمة المظلّلة مجددًا.`))
      : phase === 'corrected' ? (wholeAyah ? tr('The ayah was detected in your retry. Continue from your saved place.', 'تعرّفنا على الآية في محاولتك. تابع من موضعك المحفوظ.')
        : tr('The word was detected in your retry. Continue from your saved place.', 'تعرّفنا على الكلمة في محاولتك. تابع من موضعك المحفوظ.'))
      : skippedAyah ? tr(`We heard ayah ${issue.ayah - 1} and then ayah ${issue.ayah + 1}, but not ayah ${issue.ayah}. Recite it before you continue.`, `سمعنا الآية ${num(issue.ayah - 1)} ثم الآية ${num(issue.ayah + 1)}، ولم نسمع الآية ${num(issue.ayah)}. اتلُها قبل أن تتابع.`)
      : unclearAyah ? tr(`We heard you recite, but could not match ayah ${issue.ayah}. Recite it from the beginning at a steady pace.`, `سمعنا تلاوتك، لكن لم نتمكّن من مطابقة الآية ${num(issue.ayah)}. اتلُها من بدايتها بوتيرة ثابتة.`)
      : vowel ? tr(`Recite ayah ${issue.ayah} again and listen for the highlighted word's harakah.`, `أعد الآية ${num(issue.ayah)} وانتبه لحركة الكلمة المظلّلة.`)
      : repeated ? tr(`Recite ayah ${issue.ayah} again, saying the highlighted word once.`, `أعد الآية ${num(issue.ayah)} وانطق الكلمة المظلّلة مرة واحدة.`)
      : tr('Recite the ayah above, including the highlighted word.', 'اتلُ الآية أعلاه، بما فيها الكلمة المظلّلة.'));
    const primary = this.dialog.querySelector<HTMLButtonElement>('#practice-primary')!;
    const secondary = this.dialog.querySelector<HTMLButtonElement>('#practice-secondary')!;
    primary.dataset.action = phase === 'retrying' ? 'stop_retry' : phase === 'corrected' ? 'continue' : 'retry';
    secondary.dataset.action = phase === 'retrying' ? 'review_later' : phase === 'corrected' ? 'retry' : 'dismiss';
    primary.textContent = phase === 'retrying' ? tr('Stop retry', 'إيقاف المحاولة') : phase === 'corrected' ? tr('Continue reciting', 'متابعة التلاوة') : tr('Retry ayah', 'إعادة الآية');
    secondary.textContent = phase === 'retrying' ? tr('Review later', 'المراجعة لاحقًا') : phase === 'corrected' ? tr('Practice once more', 'التدرّب مجددًا') : tr('I recited it correctly', 'تلوتُها بشكل صحيح');
  }
}
