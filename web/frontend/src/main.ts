import "@fontsource/amiri/400.css";
import "./arabic-font.css";
import "./style.css";
import { ayahsRecited, type CheckerEvent, type EngineEvent, type MemorizationPassage, type PositionEvent, type TilawaSession, type WordRef } from "@tilawa/core";
// The engine (and onnxruntime-web, most of the JavaScript) loads when the user presses Start.
import type { LoadProgress, Microphone, Tilawa } from "@tilawa/core/web";
import { armCue, cue, cueEnabled, setCueEnabled } from "./lib/cue";
import { EXPLAIN, isShown } from "./lib/explain";
import { applyStatic, lang, tr } from "./lib/i18n";
import { ayahKey, changedSurah, displayIndex, indexQuran, surahList, type DisplayAyah, type QuranText, type QuranVerse } from "./lib/quran-words";
import { prepareMic } from "./lib/mic";
import { cleanupOldClient } from "./migrate";

// ---------------------------------------------------------------------------------------------------------------
// Setup

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

applyStatic();
/** The surah's name in the interface language. */
const nameOf = (a: DisplayAyah | undefined, surah: number) => (a ? (lang === "ar" ? a.surahName : a.surahNameEn) : tr("surahN", { n: surah }));

/** Same-origin mirror of https://tilawa.dev/v1/ (served by server/assets.ts from dist/v1). */
const ASSET_BASE = new URL(import.meta.env.VITE_TILAWA_ASSET_BASE ?? "/v1/", location.href).href;
const params = new URLSearchParams(location.search);
/**
 * ?app: the page is the Android app (web/android). A phone's main thread cannot resample the microphone, run the
 * model and draw in real time, so the browser captures at 16 kHz and the lite model runs on WASM in onnxruntime-web's
 * worker. The models are in the package, so the engine loads at launch instead of on Start.
 */
const APP = params.has("app");
// Not in the UI: lets tests and developers pin a tier (?tier=lite).
const TIER = params.get("tier") === "lite" || params.get("tier") === "full" ? (params.get("tier") as "lite" | "full") : APP ? "lite" : "auto";

void cleanupOldClient().then((r) => {
  if (r.databases.length || r.caches.length || r.serviceWorkers)
    console.info("tilawa: removed data from the previous demo", r);
});

type View = "idle" | "loading" | "load-error" | "mic-pending" | "mic-error" | "listening" | "stopped";
const VIEWS: View[] = ["idle", "loading", "load-error", "mic-pending", "mic-error", "listening", "stopped"];

function show(view: View) {
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== view;
  $("recitation").hidden = view !== "listening" && view !== "stopped";
  document.body.dataset.view = view;
  const passage = $<HTMLButtonElement>("btn-passage");
  passage.disabled = view === "loading" || view === "mic-pending" || view === "listening";
}

function setStatus(text: string) {
  $("status").textContent = text;
}

// ---------------------------------------------------------------------------------------------------------------
// Loading: the Quran display text and the engine

let quranPromise: Promise<QuranText> | null = null;
function loadQuran(): Promise<QuranText> {
  quranPromise ??= fetch("/quran.json")
    .then((r) => {
      if (!r.ok) throw new Error(`${r.status} fetching the Quran text`);
      return r.json() as Promise<QuranVerse[]>;
    })
    .then(indexQuran);
  quranPromise.catch(() => (quranPromise = null));
  return quranPromise;
}

let tilawaPromise: Promise<Tilawa> | null = null;
let loadAbort: AbortController | null = null;

const STAGE_WEIGHT = { model: 0.75, reference: 0.2, warmup: 0.05 } as const;

let ortVersion = "";

function loadTilawa(): Promise<Tilawa> {
  if (tilawaPromise) return tilawaPromise;
  const abort = (loadAbort = new AbortController());
  const done = { model: 0, reference: 0, warmup: 0 };
  let cached = true;
  tilawaPromise = Promise.all([import("@tilawa/core/web"), import("onnxruntime-web/all")]).then(([web, ort]) => {
    ortVersion = ort.env.versions?.web ?? "";
    if (APP) ort.env.wasm.proxy = true;
    return web.createTilawa({
      assetBase: ASSET_BASE,
      tier: TIER,
      // The app reads its models from the package each launch: a second copy in Cache Storage only adds a way to fail.
      ...(APP && { providers: ["wasm"] as const, storage: { get: async () => null, put: async () => {}, delete: async () => {}, keys: async () => [] } }),
      signal: abort.signal,
      onProgress: (p: LoadProgress) => {
        done[p.stage] = p.fraction;
        if (!p.fromCache && p.stage !== "warmup" && p.fraction < 1) cached = false;
        const overall = Object.entries(done).reduce((n, [k, f]) => n + f * STAGE_WEIGHT[k as keyof typeof done], 0);
        const pct = Math.floor(overall * 100);
        $("progress").style.setProperty("--progress", String(overall));
        $("progress-track").setAttribute("aria-valuenow", String(pct));
        $("progress-label").textContent = `${pct}%`;
        $("loading-detail").textContent = cached
          ? tr("loadCached")
          : p.stage === "warmup"
            ? tr("loadAlmost")
            : tr("loadDownloading");
      },
    });
  });
  tilawaPromise.then(renderTech, () => (tilawaPromise = null));
  return tilawaPromise;
}

function renderTech(t: Tilawa) {
  const backend = t.model.backend;
  const rows: [label: string, value: string, tooltip?: string][] = [
    [tr("techRunsOn"), tr(backend.includes("webgpu") ? "techGpu" : "techWasm")],
    [tr("techModel"), tr("techTier", { model: `${t.model.id}@${t.model.version}`, tier: t.model.tier ?? "custom" })],
    [tr("techEngine"), `@tilawa/core ${t.capabilities.sdkVersion ?? ""}, onnxruntime-web ${ortVersion}`],
    [tr("techReference"), t.capabilities.referenceVersion],
    [tr("techIsolated"), tr(globalThis.crossOriginIsolated ? "yes" : "no")],
    [tr("techAssets"), ASSET_BASE],
  ];
  if (t.model.fallbacks.length) {
    // Each entry is "<provider>: <runtime error>". The raw error reads like a failure, and it isn't one: say what
    // happened, and keep the error in the tooltip for developers.
    const backends = t.model.fallbacks.map((f) => f.split(":")[0].replace("webgpu", "WebGPU")).join(", ");
    rows.splice(1, 0, [tr("techSkipped"), tr("techSkippedWhy", { backends }), t.model.fallbacks.join("; ")]);
  }
  const dl = $("tech-list");
  dl.replaceChildren(
    ...rows.flatMap(([k, v, tooltip]) => {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      if (tooltip) dd.title = tooltip;
      return [dt, dd];
    }),
  );
  dl.dataset.backend = backend;
  setStatus(tr(backend.includes("webgpu") ? "onDeviceGpu" : "onDevice"));
}

// ---------------------------------------------------------------------------------------------------------------
// A take: one session, one microphone

interface Take {
  session: TilawaSession;
  mic: Microphone | null;
  events: EngineEvent[];
  view: RecitationView;
}

let take: Take | null = null;
let starting = false;

async function start(passage?: MemorizationPassage) {
  if (starting || take) return;
  starting = true;
  // Before any await: keeps the tap's user gesture for iOS audio (listen() takes ownership).
  let prepared = prepareMic(APP ? 16000 : undefined);
  armCue();
  try {
    show("loading");
    const [tilawa, quran] = await Promise.all([loadTilawa(), loadQuran()]);
    show("mic-pending");
    const session = passage
      ? tilawa.startMemorization(passage, { confirm: { after: "next-word" } })
      : tilawa.startFreeRecitation();
    const current: Take = { session, mic: null, events: [], view: new RecitationView(quran, passage ?? null) };
    session.subscribe((e) => {
      current.events.push(e);
      current.view.handle(e);
    });
    try {
      const mic = prepared ? { onLevel: meter, prepared } : { onLevel: meter };
      prepared = null;
      current.mic = await tilawa.listen(session, mic);
    } catch (e) {
      session.dispose();
      showMicError(e);
      return;
    }
    take = current;
    show("listening");
    window.scrollTo({ top: 0 }); // the passage form sits lower down; the bench is now the whole page
    void holdWake();
  } catch (e) {
    prepared?.release();
    if (e instanceof Error && e.name === "AbortError") return show("idle");
    console.error(e);
    $("load-error-detail").textContent = e instanceof Error ? e.message : String(e);
    show("load-error");
  } finally {
    starting = false;
  }
}

async function stop() {
  const t = take;
  if (!t) return;
  $<HTMLButtonElement>("btn-stop").disabled = true;
  t.view.quiet = true; // end() confirms every open flag at once: the summary reports those
  void wake?.release().catch(() => {});
  wake = null;
  try {
    await t.mic?.stop(); // ends the session: its final events arrive through subscribe
  } catch (e) {
    console.error(e);
  }
  take = null;
  $<HTMLButtonElement>("btn-stop").disabled = false;
  const mistakes = t.session.mistakes().filter(isShown);
  $("summary").textContent =
    summary(t.view.quran, ayahsRecited(t.events), mistakes.length) + (t.view.strays ? ` ${tr("sumStray")}` : "");
  t.session.dispose();
  meter(-100);
  show("stopped");
}

function summary(quran: QuranText, recited: { surah: number; ayah: number }[], mistakes: number): string {
  if (!recited.length) return tr("sumNone");
  const runs: string[] = [];
  let i = 0;
  while (i < recited.length) {
    const first = recited[i];
    let last = first;
    while (i + 1 < recited.length && recited[i + 1].surah === first.surah && recited[i + 1].ayah === last.ayah + 1) last = recited[++i];
    i++;
    const name = nameOf(quran.get(ayahKey(first.surah, first.ayah)), first.surah);
    runs.push(`${name} ${first.surah}:${first.ayah}${last.ayah !== first.ayah ? `–${last.ayah}` : ""}`);
  }
  const notes = mistakes === 0 ? tr("sumClean") : mistakes === 1 ? tr("sumOne") : tr("sumMany", { n: mistakes });
  return `${tr("sumRecited", { runs: runs.join(tr("listSep")) })} ${notes}`;
}

const MIC_HELP = {
  denied: ["micDeniedTitle", "micDeniedBody"],
  unavailable: ["micNoneTitle", "micNoneBody"],
  "insecure-context": ["micInsecureTitle", "micInsecureBody"],
} as const;

function showMicError(e: unknown) {
  const reason = (e as { reason?: keyof typeof MIC_HELP }).reason;
  const [title, detail] = MIC_HELP[reason && reason in MIC_HELP ? reason : "unavailable"];
  $("mic-error-title").textContent = tr(title);
  $("mic-error-detail").textContent = tr(detail);
  show("mic-error");
}

// Keeps the screen on while reciting. The browser drops the lock when the tab is hidden, so take it again on return.
let wake: WakeLockSentinel | null = null;
async function holdWake() {
  try {
    wake = (await navigator.wakeLock?.request("screen")) ?? null;
  } catch {
    /* unsupported, or refused (battery saver) */
  }
}
document.addEventListener("visibilitychange", () => {
  if (take && document.visibilityState === "visible") void holdWake();
});

const levelBar = $("level").firstElementChild as HTMLElement;
function meter(db: number) {
  levelBar.style.setProperty("--level", String(Math.max(0, Math.min(1, (db + 60) / 50))));
}

// ---------------------------------------------------------------------------------------------------------------
// The recitation display: the current ayah, the current word, and confirmed notes

class RecitationView {
  private shown: string | null = null;
  private spans: HTMLSpanElement[] = [];
  private readonly marks = new Map<string, CheckerEvent>();
  private readonly ayah = $("ayah");
  readonly quran: QuranText;
  /** Set when the take is stopping: no more sound cues. */
  quiet = false;
  /** Times the reciter left the declared passage. */
  strays = 0;
  private readonly passage: MemorizationPassage | null;
  /** The last locked word, and whether it was outside the declared passage. */
  private at: WordRef | null = null;
  private away = false;

  constructor(quran: QuranText, passage: MemorizationPassage | null) {
    this.quran = quran;
    this.passage = passage;
    $("place").classList.remove("away");
    this.ayah.replaceChildren();
    delete this.ayah.dataset.ref;
    delete this.ayah.dataset.word;
    $("notes").replaceChildren();
    $("notes-wrap").hidden = true;
    $("place").textContent = passage ? `${this.surahName(passage.surah)} ${passage.surah}:${passage.fromAyah}–${passage.toAyah}` : tr("listening");
    $("hint").textContent = tr(passage ? "hintPassage" : "hintFree");
    $("hint").hidden = false;
  }

  handle(e: EngineEvent) {
    if (e.type === "position") this.position(e);
    else if (e.type === "checker") this.checker(e);
  }

  private surahName(surah: number) {
    return nameOf(this.quran.get(ayahKey(surah, 1)), surah);
  }

  private position(e: PositionEvent) {
    const top = e.observed[0];
    if (e.state === "lost") $("hint").textContent = tr("hintLost");
    else if (e.state === "acquiring" && !this.shown) $("hint").textContent = tr("hintFinding");
    if (e.state === "ambiguous") {
      $("hint").hidden = false;
      $("hint").textContent = tr("hintAmbiguous");
    }
    // Only a locked position moves the display: ambiguous candidates would flicker between ayahs.
    if (!top || e.state !== "locked") return;
    $("hint").hidden = true;
    this.place(top.word);
    this.showWord(top.word);
  }

  /**
   * The engine follows a reciter who leaves the declared passage (a deliberate jump) and, in free recitation, one who
   * carries on in another surah, without a checker event for either. Say so here.
   */
  private place(w: WordRef) {
    const from = this.at;
    this.at = w;
    const p = this.passage;
    const name = this.surahName(w.surah);
    const ref = `${w.surah}:${w.ayah}`;
    if (!p) {
      if (from && changedSurah(this.quran, from, w))
        this.note("moved", tr("movedHead", { ref }), tr("movedBody", { from: this.surahName(from.surah), to: name }));
      return;
    }
    const away = w.surah !== p.surah || w.ayah < p.fromAyah || w.ayah > p.toAyah;
    if (away && !this.away) {
      this.strays++;
      const chosen = `${this.surahName(p.surah)} ${p.surah}:${p.fromAyah}–${p.toAyah}`;
      this.note("stray", tr("strayHead", { ref }), tr("strayBody", { name, ref, chosen }));
      if (!this.quiet) cue();
    }
    this.away = away;
    $("place").classList.toggle("away", away);
  }

  private showWord(w: WordRef) {
    const key = ayahKey(w.surah, w.ayah);
    const ayah = this.quran.get(key);
    if (!ayah) return;
    if (this.shown !== key) {
      this.shown = key;
      this.spans = ayah.words.map((text, i) => {
        const span = document.createElement("span");
        span.className = "word";
        span.textContent = text;
        span.dataset.index = String(i);
        return span;
      });
      this.ayah.replaceChildren(...this.spans.flatMap((s, i) => (i ? [document.createTextNode(" "), s] : [s])));
      const marker = document.createElement("span");
      marker.className = "ayah-number";
      marker.textContent = ` ${toArabicDigits(w.ayah)}`;
      this.ayah.append(marker);
      this.ayah.dataset.ref = key;
      $("place").textContent = `${nameOf(ayah, w.surah)} ${key}`;
      for (const m of this.marks.values()) this.applyMark(m);
    }
    const at = displayIndex(w.surah, w.ayah, w.word);
    this.spans.forEach((s, i) => {
      s.classList.toggle("said", i < at);
      s.classList.toggle("current", i === at);
    });
    this.ayah.dataset.word = String(w.word);
    // A long ayah runs past a phone screen: keep the word being recited in view.
    this.spans[at]?.scrollIntoView({ block: "nearest" });
  }

  private checker(e: CheckerEvent) {
    if (!isShown(e)) return;
    if (e.state === "confirmed") {
      this.marks.set(e.id, e);
      this.applyMark(e);
      this.addNote(e);
      this.spanFor(e.location)?.classList.add("flagged");
      if (!this.quiet) cue();
    } else if (e.state === "resolved" && this.marks.delete(e.id)) {
      // A self-correction: take the note back.
      document.querySelector(`[data-note="${CSS.escape(e.id)}"]`)?.remove();
      this.spanFor(e.location)?.classList.remove("mark");
      $("notes-wrap").hidden = !$("notes").childElementCount;
    }
  }

  private spanFor(w: WordRef): HTMLSpanElement | undefined {
    if (this.shown !== ayahKey(w.surah, w.ayah)) return undefined;
    return this.spans[displayIndex(w.surah, w.ayah, w.word)];
  }

  private applyMark(e: CheckerEvent) {
    const span = this.spanFor(e.location);
    if (!span || !isShown(e)) return;
    span.classList.add("mark");
    span.dataset.mark = e.errorClass;
    span.title = EXPLAIN[e.errorClass].text;
  }

  private addNote(e: CheckerEvent & { errorClass: keyof typeof EXPLAIN }) {
    const { surah, ayah, word } = e.location;
    const text = this.quran.get(ayahKey(surah, ayah))?.words[displayIndex(surah, ayah, word)];
    const li = this.note(e.errorClass, `${EXPLAIN[e.errorClass].label} · ${surah}:${ayah}`, EXPLAIN[e.errorClass].text, text);
    li.dataset.note = e.id;
  }

  private note(cls: string, heading: string, text: string, word?: string): HTMLLIElement {
    const li = document.createElement("li");
    li.className = "note";
    li.dataset.class = cls;
    const head = document.createElement("span");
    head.className = "note-head";
    head.textContent = heading;
    if (word) {
      const ar = document.createElement("span");
      ar.className = "note-word";
      ar.lang = "ar";
      ar.dir = "rtl";
      ar.textContent = word;
      head.append(" ", ar);
    }
    const body = document.createElement("span");
    body.className = "note-text";
    body.textContent = text;
    li.append(head, body);
    $("notes").append(li);
    $("notes-wrap").hidden = false;
    return li;
  }
}

const toArabicDigits = (n: number) => String(n).replace(/\d/g, (d) => "٠١٢٣٤٥٦٧٨٩"[Number(d)]);

// ---------------------------------------------------------------------------------------------------------------
// Declared passage picker

async function fillPassagePicker() {
  const select = $<HTMLSelectElement>("passage-surah");
  if (select.dataset.ready) return;
  try {
    const list = surahList(await loadQuran());
    select.replaceChildren(
      ...list.map((s) => {
        const o = document.createElement("option");
        o.value = String(s.surah);
        o.textContent = `${s.surah}. ${lang === "ar" ? s.name : s.nameEn}`;
        o.dataset.ayahs = String(s.ayahs);
        return o;
      }),
    );
    select.dataset.ready = "1";
    syncRange();
  } catch {
    select.firstElementChild!.textContent = tr("surahListFailed");
  }
}

function syncRange() {
  const select = $<HTMLSelectElement>("passage-surah");
  const max = Number(select.selectedOptions[0]?.dataset.ayahs ?? 286);
  const from = $<HTMLInputElement>("passage-from");
  const to = $<HTMLInputElement>("passage-to");
  from.max = to.max = String(max);
  if (Number(from.value) > max) from.value = "1";
  if (Number(to.value) > max || Number(to.value) < Number(from.value)) to.value = String(Math.min(max, Number(from.value) + 6));
}

$("passage").addEventListener("toggle", () => {
  if (($("passage") as HTMLDetailsElement).open) void fillPassagePicker();
});
$("passage-surah").addEventListener("change", syncRange);
$("passage-from").addEventListener("change", syncRange);
$<HTMLFormElement>("passage-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const surah = Number($<HTMLSelectElement>("passage-surah").value);
  const fromAyah = Number($<HTMLInputElement>("passage-from").value);
  const toAyah = Number($<HTMLInputElement>("passage-to").value);
  if (!surah || !fromAyah || toAyah < fromAyah) return;
  void start({ surah, fromAyah, toAyah });
});

// ---------------------------------------------------------------------------------------------------------------
// Buttons

// Language: the button offers the other one. A reload applies it (the inline script in index.html reads the choice).
const langBtn = $("btn-lang");
const otherLang = lang === "ar" ? "en" : "ar";
langBtn.textContent = tr("langOther");
langBtn.lang = otherLang;
langBtn.addEventListener("click", () => {
  const url = new URL(location.href);
  url.searchParams.delete("lang");
  try {
    localStorage.setItem("lang", otherLang);
  } catch {
    url.searchParams.set("lang", otherLang); // storage blocked: carry the choice in the address
  }
  location.replace(url);
});

// Theme: follows the system until the visitor picks one (the inline script in index.html applies the saved choice).
const themeBtn = $("btn-theme");
const systemDark = matchMedia("(prefers-color-scheme: dark)");
const isDark = () => (document.documentElement.dataset.theme ?? (systemDark.matches ? "dark" : "light")) === "dark";
const syncTheme = () => themeBtn.setAttribute("aria-pressed", String(isDark()));
themeBtn.addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch {
    /* not remembered */
  }
  syncTheme();
});
systemDark.addEventListener("change", syncTheme);
syncTheme();

const cueToggle = $<HTMLInputElement>("cue-toggle");
cueToggle.checked = cueEnabled();
cueToggle.addEventListener("change", () => {
  setCueEnabled(cueToggle.checked);
  armCue(); // switched on mid-take: this change is the gesture that unlocks audio
});

$("btn-start").addEventListener("click", () => void start());
$("btn-again").addEventListener("click", () => void start());
$("btn-retry-load").addEventListener("click", () => void start());
$("btn-retry-mic").addEventListener("click", () => void start());
$("btn-stop").addEventListener("click", () => void stop());
$("btn-cancel").addEventListener("click", () => loadAbort?.abort());
window.addEventListener("pagehide", () => void take?.mic?.stop());

show("idle");

if (APP) {
  document.documentElement.classList.add("bundled");
  // Frees the model copy that app builds before this one kept in Cache Storage.
  void globalThis.caches?.keys().then((keys) => keys.forEach((k) => void caches.delete(k))).catch(() => {});
  void loadTilawa().catch(() => {});
  void loadQuran().catch(() => {});
}
