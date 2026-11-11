import "@fontsource/amiri/400.css";
import "./arabic-font.css";
import "./style.css";
import { ayahsRecited, type CheckerEvent, type EngineEvent, type MemorizationPassage, type PositionEvent, type TilawaSession, type WordRef } from "@tilawa/core";
// The engine (and onnxruntime-web, most of the JavaScript) loads when the user presses Start.
import type { LoadProgress, Microphone, Tilawa } from "@tilawa/core/web";
import { EXPLAIN, isShown } from "./lib/explain";
import { ayahKey, displayIndex, indexQuran, surahList, type QuranText, type QuranVerse } from "./lib/quran-words";
import { cleanupOldClient } from "./migrate";

// ---------------------------------------------------------------------------------------------------------------
// Setup

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

/** Same-origin mirror of https://tilawa.dev/v1/ (served by server/assets.ts from dist/v1). */
const ASSET_BASE = new URL(import.meta.env.VITE_TILAWA_ASSET_BASE ?? "/v1/", location.href).href;
const params = new URLSearchParams(location.search);
// Not in the UI: lets tests and developers pin a tier (?tier=lite).
const TIER = params.get("tier") === "lite" || params.get("tier") === "full" ? (params.get("tier") as "lite" | "full") : "auto";

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
    return web.createTilawa({
      assetBase: ASSET_BASE,
      tier: TIER,
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
          ? "Loading from this device."
          : p.stage === "warmup"
            ? "Almost there."
            : "Downloading once. Next time Tilawa starts straight away, even offline.";
      },
    });
  });
  tilawaPromise.then(renderTech, () => (tilawaPromise = null));
  return tilawaPromise;
}

function renderTech(t: Tilawa) {
  const backend = t.model.backend;
  const runsOn = backend.includes("webgpu") ? "WebGPU (your graphics chip)" : "WebAssembly (WASM, your processor)";
  const rows: [string, string][] = [
    ["Runs on", runsOn],
    ["Model", `${t.model.id}@${t.model.version} (${t.model.tier} tier)`],
    ["Engine", `@tilawa/core ${t.capabilities.sdkVersion ?? ""}, onnxruntime-web ${ortVersion}`],
    ["Reference", t.capabilities.referenceVersion],
    ["Cross-origin isolated", globalThis.crossOriginIsolated ? "yes" : "no"],
    ["Assets", ASSET_BASE],
  ];
  if (t.model.fallbacks.length) rows.splice(1, 0, ["Skipped", t.model.fallbacks.join("; ")]);
  const dl = $("tech-list");
  dl.replaceChildren(
    ...rows.flatMap(([k, v]) => {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      return [dt, dd];
    }),
  );
  dl.dataset.backend = backend;
  setStatus(backend.includes("webgpu") ? "On-device · WebGPU" : "On-device");
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
      current.mic = await tilawa.listen(session, { onLevel: meter });
    } catch (e) {
      session.dispose();
      showMicError(e);
      return;
    }
    take = current;
    show("listening");
  } catch (e) {
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
  try {
    await t.mic?.stop(); // ends the session: its final events arrive through subscribe
  } catch (e) {
    console.error(e);
  }
  take = null;
  $<HTMLButtonElement>("btn-stop").disabled = false;
  const mistakes = t.session.mistakes().filter(isShown);
  $("summary").textContent = summary(t.view.quran, ayahsRecited(t.events), mistakes.length);
  t.session.dispose();
  meter(-100);
  show("stopped");
}

function summary(quran: QuranText, recited: { surah: number; ayah: number }[], mistakes: number): string {
  if (!recited.length)
    return "Tilawa didn't find your place this time. Try a little longer, a little louder, or closer to the microphone.";
  const runs: string[] = [];
  let i = 0;
  while (i < recited.length) {
    const first = recited[i];
    let last = first;
    while (i + 1 < recited.length && recited[i + 1].surah === first.surah && recited[i + 1].ayah === last.ayah + 1) last = recited[++i];
    i++;
    const name = quran.get(ayahKey(first.surah, first.ayah))?.surahNameEn ?? `Surah ${first.surah}`;
    runs.push(`${name} ${first.surah}:${first.ayah}${last.ayah !== first.ayah ? `–${last.ayah}` : ""}`);
  }
  const notes = mistakes === 0 ? "Nothing to look at again." : mistakes === 1 ? "One word is worth a second look." : `${mistakes} words are worth a second look.`;
  return `You recited ${runs.join(", ")}. ${notes}`;
}

const MIC_HELP = {
  denied: [
    "Microphone access is blocked.",
    "Tilawa needs the microphone to hear you. Allow it from the icon beside the address bar (or in your browser's site settings), then try again. On iPhone: Settings › Safari › Microphone.",
  ],
  unavailable: [
    "No microphone found.",
    "Connect a microphone, or close other apps that might be using it, then try again.",
  ],
  "insecure-context": [
    "This page can't use the microphone.",
    "Browsers only allow the microphone on secure (https) pages. Open https://tilawa.dev directly.",
  ],
} as const;

function showMicError(e: unknown) {
  const reason = (e as { reason?: keyof typeof MIC_HELP }).reason;
  const [title, detail] = MIC_HELP[reason && reason in MIC_HELP ? reason : "unavailable"];
  $("mic-error-title").textContent = title;
  $("mic-error-detail").textContent = detail;
  show("mic-error");
}

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

  constructor(quran: QuranText, passage: MemorizationPassage | null) {
    this.quran = quran;
    this.ayah.replaceChildren();
    delete this.ayah.dataset.ref;
    delete this.ayah.dataset.word;
    $("notes").replaceChildren();
    $("notes-wrap").hidden = true;
    $("place").textContent = passage ? `${this.surahName(passage.surah)} ${passage.surah}:${passage.fromAyah}–${passage.toAyah}` : "Listening";
    $("hint").textContent = passage ? "Begin the passage when you're ready." : "Recite a few words. Tilawa will find your place.";
    $("hint").hidden = false;
  }

  handle(e: EngineEvent) {
    if (e.type === "position") this.position(e);
    else if (e.type === "checker") this.checker(e);
  }

  private surahName(surah: number) {
    return this.quran.get(ayahKey(surah, 1))?.surahNameEn ?? `Surah ${surah}`;
  }

  private position(e: PositionEvent) {
    const top = e.observed[0];
    if (e.state === "lost") $("hint").textContent = "Lost your place. Keep reciting and Tilawa will find it again.";
    else if (e.state === "acquiring" && !this.shown) $("hint").textContent = "Finding your place…";
    if (e.state === "ambiguous") {
      $("hint").hidden = false;
      $("hint").textContent = "A few places match. Keep going.";
    }
    // Only a locked position moves the display: ambiguous candidates would flicker between ayahs.
    if (!top || e.state !== "locked") return;
    $("hint").hidden = true;
    this.showWord(top.word);
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
      $("place").textContent = `${ayah.surahNameEn} ${key}`;
      for (const m of this.marks.values()) this.applyMark(m);
    }
    const at = displayIndex(w.surah, w.ayah, w.word);
    this.spans.forEach((s, i) => {
      s.classList.toggle("said", i < at);
      s.classList.toggle("current", i === at);
    });
    this.ayah.dataset.word = String(w.word);
  }

  private checker(e: CheckerEvent) {
    if (!isShown(e)) return;
    if (e.state === "confirmed") {
      this.marks.set(e.id, e);
      this.applyMark(e);
      this.addNote(e);
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
    const li = document.createElement("li");
    li.className = "note";
    li.dataset.note = e.id;
    li.dataset.class = e.errorClass;
    const head = document.createElement("span");
    head.className = "note-head";
    head.textContent = `${EXPLAIN[e.errorClass].label} · ${surah}:${ayah}`;
    const text = this.quran.get(ayahKey(surah, ayah))?.words[displayIndex(surah, ayah, word)];
    if (text) {
      const ar = document.createElement("span");
      ar.className = "note-word";
      ar.lang = "ar";
      ar.dir = "rtl";
      ar.textContent = text;
      head.append(" ", ar);
    }
    const body = document.createElement("span");
    body.className = "note-text";
    body.textContent = EXPLAIN[e.errorClass].text;
    li.append(head, body);
    $("notes").append(li);
    $("notes-wrap").hidden = false;
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
        o.textContent = `${s.surah}. ${s.nameEn}`;
        o.dataset.ayahs = String(s.ayahs);
        return o;
      }),
    );
    select.dataset.ready = "1";
    syncRange();
  } catch {
    select.firstElementChild!.textContent = "Couldn't load the surah list";
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

$("btn-start").addEventListener("click", () => void start());
$("btn-again").addEventListener("click", () => void start());
$("btn-retry-load").addEventListener("click", () => void start());
$("btn-retry-mic").addEventListener("click", () => void start());
$("btn-stop").addEventListener("click", () => void stop());
$("btn-cancel").addEventListener("click", () => loadAbort?.abort());
window.addEventListener("pagehide", () => void take?.mic?.stop());

show("idle");
