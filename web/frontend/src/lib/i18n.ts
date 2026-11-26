/**
 * English and Arabic wording for the demo. The static English text is index.html itself; Arabic replaces it by
 * `data-i18n` key (STATIC_AR). Text the script builds comes from `tr()`.
 */
export type Lang = "en" | "ar";

/** Chosen before first paint by the inline script in index.html: ?lang=, the saved choice, then the browser's. */
export const lang: Lang = typeof document !== "undefined" && document.documentElement.lang === "ar" ? "ar" : "en";

const en = {
  langOther: "العربية",
  onDevice: "On-device",
  onDeviceGpu: "On-device · WebGPU",
  loadCached: "Loading from this device.",
  loadAlmost: "Almost there.",
  loadDownloading: "Downloading once. Next time Tilawa starts straight away, even offline.",
  listening: "Listening",
  hintFree: "Recite a few words. Tilawa will find your place.",
  hintPassage: "Begin the passage when you're ready.",
  hintLost: "Lost your place. Keep reciting and Tilawa will find it again.",
  hintFinding: "Finding your place…",
  hintAmbiguous: "A few places match. Keep going.",
  strayHead: "Outside your passage · {ref}",
  strayBody: "This sounds like {name} {ref}, which isn't in the passage you chose ({chosen}).",
  movedHead: "Changed surah · {ref}",
  movedBody: "It sounds like you moved from {from} to {to}.",
  wrongLabel: "Different word",
  wrongText: "It sounded like a different word here.",
  omittedLabel: "Missed word",
  omittedText: "This word seems to have been left out.",
  addedLabel: "Extra word",
  addedText: "An extra word seems to have been added here.",
  skippedLabel: "Skipped part",
  skippedText: "Part of the passage seems to have been skipped from here.",
  sumNone: "Tilawa didn't find your place this time. Try a little longer, a little louder, or closer to the microphone.",
  sumRecited: "You recited {runs}.",
  sumClean: "Nothing to look at again.",
  sumOne: "One word is worth a second look.",
  sumMany: "{n} words are worth a second look.",
  sumStray: "Part of it was outside the passage you chose.",
  listSep: ", ",
  surahN: "Surah {n}",
  micDeniedTitle: "Microphone access is blocked.",
  micDeniedBody:
    "Tilawa needs the microphone to hear you. Allow it from the icon beside the address bar (or in your browser's site settings), then try again. On iPhone: Settings › Safari › Microphone.",
  micNoneTitle: "No microphone found.",
  micNoneBody: "Connect a microphone, or close other apps that might be using it, then try again.",
  micInsecureTitle: "This page can't use the microphone.",
  micInsecureBody: "Browsers only allow the microphone on secure (https) pages. Open https://tilawa.dev directly.",
  techRunsOn: "Runs on",
  techGpu: "WebGPU (your graphics chip)",
  techWasm: "WebAssembly (WASM, your processor)",
  techSkipped: "Not used",
  techSkippedWhy: "{backends} isn't available in this browser, so Tilawa uses WebAssembly instead. Nothing is wrong.",
  techModel: "Model",
  techTier: "{model} ({tier} tier)",
  techEngine: "Engine",
  techReference: "Reference",
  techIsolated: "Cross-origin isolated",
  techAssets: "Assets",
  yes: "yes",
  no: "no",
  surahListFailed: "Couldn't load the surah list",
};

export type Key = keyof typeof en;

const ar: Record<Key, string> = {
  langOther: "English",
  onDevice: "على الجهاز",
  onDeviceGpu: "على الجهاز · WebGPU",
  loadCached: "جارٍ التحميل من هذا الجهاز.",
  loadAlmost: "أوشكنا على الانتهاء.",
  loadDownloading: "تنزيل لمرة واحدة. في المرة القادمة يبدأ تلاوة فورًا، حتى بلا اتصال.",
  listening: "جارٍ الاستماع",
  hintFree: "اقرأ بضع كلمات وسيعرف تلاوة موضعك.",
  hintPassage: "ابدأ المقطع متى كنت جاهزًا.",
  hintLost: "فُقد موضعك. واصل القراءة وسيجده تلاوة من جديد.",
  hintFinding: "جارٍ تحديد موضعك…",
  hintAmbiguous: "أكثر من موضع يطابق ما قرأت. واصل القراءة.",
  strayHead: "خارج المقطع المحدّد · {ref}",
  strayBody: "يبدو أن هذه {name} {ref}، وهي ليست ضمن المقطع الذي اخترته ({chosen}).",
  movedHead: "انتقال إلى سورة أخرى · {ref}",
  movedBody: "يبدو أنك انتقلت من {from} إلى {to}.",
  wrongLabel: "كلمة مختلفة",
  wrongText: "بدت الكلمة هنا مختلفة عمّا في المصحف.",
  omittedLabel: "كلمة ساقطة",
  omittedText: "يبدو أن هذه الكلمة سقطت.",
  addedLabel: "كلمة زائدة",
  addedText: "يبدو أن كلمة زيدت هنا.",
  skippedLabel: "مقطع متروك",
  skippedText: "يبدو أن جزءًا من المقطع تُرك ابتداءً من هنا.",
  sumNone: "لم يتمكن تلاوة من تحديد موضعك هذه المرة. جرّب قراءة أطول، أو بصوت أعلى، أو أقرب إلى الميكروفون.",
  sumRecited: "قرأت {runs}.",
  sumClean: "لا شيء يستدعي المراجعة.",
  sumOne: "كلمة واحدة تستحق المراجعة.",
  sumMany: "عدد الكلمات التي تستحق المراجعة: {n}.",
  sumStray: "جزء من التلاوة كان خارج المقطع الذي اخترته.",
  listSep: "، ",
  surahN: "سورة {n}",
  micDeniedTitle: "استخدام الميكروفون محظور.",
  micDeniedBody:
    "يحتاج تلاوة إلى الميكروفون ليسمعك. اسمح به من الأيقونة بجانب شريط العنوان (أو من إعدادات الموقع في المتصفح) ثم حاول مرة أخرى. على الآيفون: الإعدادات › Safari › الميكروفون.",
  micNoneTitle: "لم يُعثر على ميكروفون.",
  micNoneBody: "صِل ميكروفونًا، أو أغلق التطبيقات الأخرى التي قد تستخدمه، ثم حاول مرة أخرى.",
  micInsecureTitle: "هذه الصفحة لا تستطيع استخدام الميكروفون.",
  micInsecureBody: "المتصفحات لا تسمح بالميكروفون إلا في الصفحات الآمنة (https). افتح https://tilawa.dev مباشرة.",
  techRunsOn: "يعمل على",
  techGpu: "WebGPU (معالج الرسوميات)",
  techWasm: "WebAssembly (المعالج)",
  techSkipped: "غير مستخدَم",
  techSkippedWhy: "{backends} غير متاح في هذا المتصفح، لذلك يستخدم تلاوة WebAssembly بدلًا منه. لا يوجد خطأ.",
  techModel: "النموذج",
  techTier: "{model} (فئة {tier})",
  techEngine: "المحرّك",
  techReference: "المرجع",
  techIsolated: "عزل المصادر",
  techAssets: "الملفات",
  yes: "نعم",
  no: "لا",
  surahListFailed: "تعذّر تحميل قائمة السور",
};

const STRINGS = { en, ar };

export function tr(key: Key, vars: Record<string, string | number> = {}, to: Lang = lang): string {
  return STRINGS[to][key].replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? ""));
}

/** Arabic for index.html's `data-i18n` (content) and `data-i18n-label` (aria-label) keys. */
export const STATIC_AR: Record<string, string> = {
  title: "تلاوة — رتّل القرآن وشاهد الآية",
  navLabel: "التنقل الرئيسي",
  themeLabel: "الوضع الداكن",
  navDevelopers: "للمطوّرين",
  h1: "رتّل القرآن. وشاهد الآية.",
  lede: "ابدأ من أي موضع. تلاوة يعرف مكانك ويتابعك كلمة بكلمة، وينبّهك بلطف إلى ما سقط أو زاد أو اختلف من الكلمات.",
  previewNote: "نسخة تجريبية: قد تفوته بعض الأخطاء، وقد ينبّه أحيانًا على تلاوة صحيحة. لا يُفحص النطق ولا التجويد بعد.",
  demoTitle: "جرّب تلاوة",
  status: "على الجهاز",
  exampleLabel: "مثال · الإخلاص 112:1",
  idleTitle: "ابدأ من أي موضع في القرآن.",
  idleCopy: "اضغط «ابدأ التلاوة»، واسمح باستخدام الميكروفون، ثم اقرأ. لا يحتاج أي إعداد.",
  start: "ابدأ التلاوة",
  firstNote: "أول تشغيل ينزّل نحو 70 ميغابايت، مرة واحدة.<br />صوتك يبقى على هذا الجهاز.",
  loadingTitle: "جارٍ التجهيز…",
  loadingDetail: "تنزيل لمرة واحدة. في المرة القادمة يبدأ تلاوة فورًا، حتى بلا اتصال.",
  progressLabel: "جارٍ التجهيز",
  cancel: "إلغاء",
  loadErrTitle: "تعذّر تجهيز تلاوة.",
  loadErrCopy: "تحقّق من اتصالك ثم حاول مرة أخرى. ما نُزّل سابقًا محفوظ.",
  retry: "حاول مرة أخرى",
  micTitle: "اسمح باستخدام الميكروفون",
  micCopy: "اختر «سماح» في نافذة المتصفح. إن لم تظهر، انظر إلى الأيقونة بجانب شريط العنوان.",
  notesTitle: "مواضع تستحق المراجعة",
  stop: "إيقاف",
  micOn: "الميكروفون يعمل · الصوت يبقى على هذا الجهاز",
  again: "اقرأ مرة أخرى",
  cueLabel: "تنبيه صوتي خفيف عند الإشارة إلى كلمة",
  passageSummary: "مراجعة مقطع محدّد",
  passageCopy: "اختر ما ستقرؤه. يبقى تلاوة ضمن المقطع وينبّهك إلى ما سقط أو زاد أو اختلف.",
  surahLabel: "السورة",
  fromLabel: "من الآية",
  toLabel: "إلى الآية",
  passageBtn: "ابدأ هذا المقطع",
  loadingOption: "جارٍ التحميل…",
  scopeNote: "ينبّه تلاوة إلى الكلمات المختلفة والساقطة والزائدة والمقاطع المتروكة. لا يفحص النطق ولا التجويد بعد.",
  details: "تفاصيل",
  runsOn: "يعمل على",
  runsOnIdle: "يُحدَّد عند الضغط على «ابدأ التلاوة»: WebGPU إن توفّر في جهازك، وإلا WebAssembly.",
  devEyebrow: "للمطوّرين",
  devTitle: "المحرّك نفسه. في تطبيقك.",
  devCopy:
    'هذه الصفحة هي <code dir="ltr">@tilawa/core</code> بلا أي خيارات. مكتوب بلغة TypeScript للمتصفح و Node، برخصة MIT، ولا يتصل بالشبكة إلا لتنزيل نموذجه.',
  docs: "التوثيق",
  footerNote: "يعمل على جهازك. صوتك لا يُرفع أبدًا.",
  source: "المصدر ↗",
};

export function applyStatic(root: ParentNode = document) {
  if (lang !== "ar") return;
  // innerHTML: these strings are ours (two carry <br> or <code>), never user input.
  for (const el of root.querySelectorAll<HTMLElement>("[data-i18n]")) el.innerHTML = STATIC_AR[el.dataset.i18n!] ?? el.innerHTML;
  for (const el of root.querySelectorAll<HTMLElement>("[data-i18n-label]"))
    if (STATIC_AR[el.dataset.i18nLabel!]) el.setAttribute("aria-label", STATIC_AR[el.dataset.i18nLabel!]);
  document.title = STATIC_AR.title;
}
