import {
  ayahKey,
  displayIndex,
  type QuranText,
} from "../../src/lib/quran-words";
import {
  getAyahsForPage,
  getJuzForPage,
  getPageForVerse,
  toArabicDigits,
  TOTAL_PAGES,
} from "../../src/lib/mushaf";
import { EXPLAIN, isShown } from "../../src/lib/explain";
import { prepareMic } from "../../src/lib/mic";
import { loadQuran, loadTilawa } from "../../src/lib/tilawa";
import type {
  CheckerEvent,
  EngineEvent,
  MemorizationErrorClass,
  PositionEvent,
  TilawaSession,
} from "@tilawa/core";
import type { Microphone } from "@tilawa/core/web";

type ShownCheckerEvent = CheckerEvent & { errorClass: MemorizationErrorClass };
type PositionSource = { kind: "recognized"; confidence: number } | { kind: "manual" };

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;

// ---------------- State ----------------
let quran: QuranText | null = null;
let currentPage = 1;
let tilawaLoaded = false;
let starting = false;
let activeSession: TilawaSession | null = null;
let activeMic: Microphone | null = null;
let isReciting = false;
let currentHighlightedWordEl: HTMLElement | null = null;
let manualWordEl: HTMLElement | null = null;

const activeMistakes = new Map<string, ShownCheckerEvent>();

const levelBar = $("audio-level").firstElementChild as HTMLElement;
function meter(db: number) {
  const pct = Math.max(0, Math.min(100, ((db + 60) / 50) * 100));
  levelBar.style.setProperty("--level", `${pct.toFixed(0)}%`);
}

// ---------------- Init ----------------
async function init() {
  const params = new URLSearchParams(location.search);
  const p = parseInt(params.get("page") ?? "1", 10);
  if (!isNaN(p) && p >= 1 && p <= TOTAL_PAGES) {
    currentPage = p;
  }

  setupEventListeners();
  updatePageControls();

  try {
    quran = await loadQuran();
    renderPage(currentPage);
  } catch (err) {
    console.error("Quran text load error:", err);
    $("mushaf-page-text").textContent = "Error loading Quran text. Please refresh.";
  }
}

// ---------------- Navigation & Rendering ----------------
function renderPage(pageNum: number) {
  if (!quran) return;
  currentPage = Math.max(1, Math.min(TOTAL_PAGES, pageNum));
  currentHighlightedWordEl = null;
  manualWordEl = null;

  // Update URL without reload
  const url = new URL(window.location.href);
  url.searchParams.set("page", String(currentPage));
  window.history.replaceState({}, "", url.toString());

  updatePageControls();

  const ayahs = getAyahsForPage(currentPage, quran);
  const juzInfo = getJuzForPage(currentPage);
  const firstAyah = ayahs[0];

  // Headers
  const surahTitle = firstAyah ? firstAyah.surahName : "سورة";
  $("header-juz").textContent = juzInfo.nameAr;
  $("header-surah").textContent = surahTitle;
  $("meta-juz-name").textContent = `الجُزْءُ ${toArabicDigits(juzInfo.number)}`;
  $("meta-surah-name").textContent = surahTitle;
  $("meta-page-num").textContent = toArabicDigits(currentPage);
  $("meta-page-footer").textContent = toArabicDigits(currentPage);

  // Content
  const container = $("mushaf-page-text");
  container.replaceChildren();

  ayahs.forEach((ayah) => {
    // New Surah Banner
    if (ayah.ayah === 1) {
      const banner = document.createElement("div");
      banner.className = "surah-header-banner";
      const titleSpan = document.createElement("div");
      titleSpan.className = "surah-title-arabic";
      titleSpan.textContent = ayah.surahName;
      banner.appendChild(titleSpan);
      container.appendChild(banner);

      // Bismillah banner (not for Fatiha 1 or Tawbah 9)
      if (ayah.surah !== 1 && ayah.surah !== 9) {
        const bismillah = document.createElement("div");
        bismillah.className = "bismillah-banner";
        bismillah.textContent = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ";
        container.appendChild(bismillah);
      }
    }

    // Verse words
    ayah.words.forEach((wordText, wordIdx) => {
      const wordSpan = document.createElement("span");
      wordSpan.className = "m-word";
      wordSpan.id = `w-${ayah.surah}-${ayah.ayah}-${wordIdx}`;
      wordSpan.dataset.surah = String(ayah.surah);
      wordSpan.dataset.ayah = String(ayah.ayah);
      wordSpan.dataset.word = String(wordIdx);
      wordSpan.textContent = wordText;

      // Manual jump / inspection on click
      wordSpan.addEventListener("click", () => {
        if (manualWordEl && manualWordEl !== wordSpan) {
          manualWordEl.classList.remove("manual");
        }
        wordSpan.classList.add("manual");
        manualWordEl = wordSpan;
        setMonitorPosition(ayah.surah, ayah.ayah, wordIdx + 1, wordText, { kind: "manual" });
      });

      container.appendChild(wordSpan);
      container.appendChild(document.createTextNode(" "));
    });

    // Verse End Symbol ۝
    const endSymbol = document.createElement("span");
    endSymbol.className = "ayah-end-symbol";
    endSymbol.textContent = ` ۝${toArabicDigits(ayah.ayah)} `;
    container.appendChild(endSymbol);
  });

  // Re-apply mistake markers for words on this page
  syncMistakeMarkersOnCurrentPage();
}

function updatePageControls() {
  $<HTMLInputElement>("page-input").value = String(currentPage);
  $<HTMLButtonElement>("btn-prev-page").disabled = currentPage <= 1;
  $<HTMLButtonElement>("btn-next-page").disabled = currentPage >= TOTAL_PAGES;
}

function resetSessionView() {
  activeMistakes.clear();
  renderMistakes();

  currentHighlightedWordEl = null;
  manualWordEl = null;
  for (const el of $("mushaf-page-text").querySelectorAll<HTMLElement>(".m-word")) {
    el.classList.remove("current", "said", "mistake", "manual");
    el.removeAttribute("title");
  }

  $("mon-surah").textContent = "-";
  $("mon-ayah").textContent = "-";
  $("mon-word").textContent = "-";
  $("mon-word-text").textContent = "";
  $("mon-confidence").textContent = "Waiting for recitation...";
}

// ---------------- Engine & Recitation ----------------
function toggleRecitation() {
  if (isReciting) {
    void stopRecitation();
  } else {
    void startRecitation();
  }
}

async function startRecitation() {
  if (starting || isReciting) return;
  starting = true;
  // Before any await: keeps the tap's user gesture for iOS audio (listen() takes ownership).
  let prepared = prepareMic();

  const btn = $<HTMLButtonElement>("btn-start");
  const btnLabel = $("btn-start-label");

  btn.disabled = true;
  setStatus("loading", "Initializing engine...");
  resetSessionView();

  try {
    if (!tilawaLoaded) {
      showLoadingBanner(true, "Preparing speech recognition engine...");
    }
    const [tilawa, loadedQuran] = await Promise.all([
      loadTilawa({
        onProgress: ({ pct, cached, stage }) => {
          $<HTMLDivElement>("loading-progress").style.width = `${pct}%`;
          $("loading-pct").textContent = `${pct}%`;
          $("loading-msg").textContent = cached
            ? "Loading from this device..."
            : stage === "warmup"
              ? "Warming up engine..."
              : `Downloading model (${pct}%)...`;
        },
      }),
      loadQuran(),
    ]);
    tilawaLoaded = true;
    showLoadingBanner(false);

    if (!quran) {
      quran = loadedQuran;
      renderPage(currentPage);
    }

    const session = tilawa.startFreeRecitation();
    session.subscribe(handleEngineEvent);

    let mic: Microphone;
    try {
      const micOpts = prepared ? { onLevel: meter, prepared } : { onLevel: meter };
      prepared = null;
      mic = await tilawa.listen(session, micOpts);
    } catch (err) {
      session.dispose();
      throw err;
    }

    activeSession = session;
    activeMic = mic;
    isReciting = true;

    btn.classList.add("active");
    btnLabel.textContent = "Stop Reciting";
    btn.disabled = false;
    $("audio-level").hidden = false;
    setStatus("listening", "Listening... Start reciting anywhere");
  } catch (err: unknown) {
    prepared?.release();
    showLoadingBanner(false);
    btn.disabled = false;
    isReciting = false;
    const msg = err instanceof Error ? err.message : String(err);
    console.error("Recitation start failed:", err);
    setStatus("idle", `Failed to start: ${msg}`);
  } finally {
    starting = false;
  }
}

async function stopRecitation() {
  const mic = activeMic;
  const session = activeSession;
  if (!mic && !session) return;

  const btn = $<HTMLButtonElement>("btn-start");
  btn.disabled = true;

  activeMic = null;
  activeSession = null;

  try {
    await mic?.stop(); // ends the session: its final events arrive through subscribe
  } catch (err) {
    console.error("Recitation stop failed:", err);
  }
  session?.dispose();

  isReciting = false;
  meter(-100);
  btn.classList.remove("active");
  $("btn-start-label").textContent = "Start Reciting";
  btn.disabled = false;
  $("audio-level").hidden = true;
  setStatus("idle", "Recitation stopped");
}

function handleEngineEvent(e: EngineEvent) {
  if (e.type === "position") {
    handlePosition(e);
  } else if (e.type === "checker") {
    handleChecker(e);
  }
}

function handlePosition(e: PositionEvent) {
  const top = e.observed[0];

  if (e.state === "acquiring") {
    setStatus("listening", "Finding your place...");
    return;
  }
  if (e.state === "ambiguous") {
    setStatus("listening", "Multiple passages match...");
    return;
  }
  if (e.state === "lost") {
    setStatus("listening", "Lost place. Keep reciting...");
    return;
  }

  if (e.state !== "locked" || !top) return;

  const { surah, ayah, word } = top.word;
  const wordDisplayIdx = displayIndex(surah, ayah, word);
  const targetPage = getPageForVerse(surah, ayah);

  const autoflip = $<HTMLInputElement>("toggle-autoflip").checked;
  if (autoflip && targetPage !== currentPage) {
    renderPage(targetPage);
  }

  if (manualWordEl) {
    manualWordEl.classList.remove("manual");
    manualWordEl = null;
  }

  // Highlight word on page
  const wordEl = document.getElementById(`w-${surah}-${ayah}-${wordDisplayIdx}`);
  if (wordEl) {
    if (currentHighlightedWordEl && currentHighlightedWordEl !== wordEl) {
      currentHighlightedWordEl.classList.remove("current");
      currentHighlightedWordEl.classList.add("said");
    }
    wordEl.classList.add("current");
    currentHighlightedWordEl = wordEl;

    // Smooth scroll into view if out of page bounds
    wordEl.scrollIntoView({ behavior: "smooth", block: "nearest" });

    setMonitorPosition(surah, ayah, word, wordEl.textContent ?? "", {
      kind: "recognized",
      confidence: top.confidence,
    });
  }

  setStatus("locked", `Following: Surah ${surah}:${ayah} (Page ${targetPage})`);
}

function handleChecker(e: CheckerEvent) {
  if (!isShown(e)) return;

  if (e.state === "confirmed") {
    activeMistakes.set(e.id, e);
    renderMistakes();
    syncMistakeMarkersOnCurrentPage();
  } else if (e.state === "resolved" && activeMistakes.delete(e.id)) {
    // A self-correction: clear the mark and take the note back.
    renderMistakes();
    syncMistakeMarkersOnCurrentPage();
  }
}

function setMonitorPosition(
  surah: number,
  ayah: number,
  word: number,
  text: string,
  source: PositionSource,
) {
  $("mon-surah").textContent = String(surah);
  $("mon-ayah").textContent = String(ayah);
  $("mon-word").textContent = String(word);
  $("mon-word-text").textContent = text;
  $("mon-confidence").textContent =
    source.kind === "manual"
      ? "Manual jump"
      : `Confidence: ${(source.confidence * 100).toFixed(0)}%`;
}

function renderMistakes() {
  const listEl = $("monitor-mistakes");
  const mistakes = [...activeMistakes.values()].reverse();
  $("mistake-count").textContent = String(mistakes.length);

  if (!mistakes.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "No mistakes recorded yet. Recite smoothly.";
    listEl.replaceChildren(empty);
    return;
  }

  listEl.replaceChildren(
    ...mistakes.map((m) => {
      const li = document.createElement("li");
      li.className = "mistake-item";
      li.dataset.note = m.id;

      const header = document.createElement("div");
      header.className = "mistake-item-header";

      const typeEl = document.createElement("span");
      typeEl.className = "mistake-type";
      typeEl.textContent = EXPLAIN[m.errorClass].label;

      const locEl = document.createElement("span");
      locEl.className = "mistake-loc";
      locEl.textContent = `${m.location.surah}:${m.location.ayah}`;

      header.append(typeEl, locEl);
      li.append(header);

      const wordText = quran?.get(ayahKey(m.location.surah, m.location.ayah))?.words[
        displayIndex(m.location.surah, m.location.ayah, m.location.word)
      ];
      if (wordText) {
        const ar = document.createElement("div");
        ar.className = "mistake-word";
        ar.dir = "rtl";
        ar.lang = "ar";
        ar.textContent = wordText;
        li.append(ar);
      }

      const detail = document.createElement("div");
      detail.className = "mistake-detail";
      detail.textContent = `${EXPLAIN[m.errorClass].text} (word ${m.location.word}, ${(m.timeMs / 1000).toFixed(1)}s)`;

      li.append(detail);
      return li;
    }),
  );
}

function syncMistakeMarkersOnCurrentPage() {
  for (const el of $("mushaf-page-text").querySelectorAll<HTMLElement>(".m-word.mistake")) {
    el.classList.remove("mistake");
    el.removeAttribute("title");
  }
  for (const m of activeMistakes.values()) {
    const at = displayIndex(m.location.surah, m.location.ayah, m.location.word);
    const el = document.getElementById(`w-${m.location.surah}-${m.location.ayah}-${at}`);
    if (el) {
      el.classList.add("mistake");
      el.title = EXPLAIN[m.errorClass].text;
    }
  }
}

function setStatus(state: "idle" | "loading" | "listening" | "locked", text: string) {
  const pill = $("status-pill");
  pill.className = `status-pill ${state}`;
  $("status-text").textContent = text;
}

function showLoadingBanner(show: boolean, msg = "") {
  const banner = $("loading-banner");
  banner.hidden = !show;
  if (show && msg) {
    $("loading-msg").textContent = msg;
  }
}

// ---------------- Event Listeners ----------------
function setupEventListeners() {
  $("btn-start").addEventListener("click", toggleRecitation);

  $("btn-prev-page").addEventListener("click", () => {
    if (currentPage > 1) renderPage(currentPage - 1);
  });

  $("btn-next-page").addEventListener("click", () => {
    if (currentPage < TOTAL_PAGES) renderPage(currentPage + 1);
  });

  const pageInput = $<HTMLInputElement>("page-input");
  pageInput.addEventListener("change", () => {
    const val = parseInt(pageInput.value, 10);
    if (!isNaN(val)) renderPage(val);
  });

  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

    if (e.code === "Space") {
      e.preventDefault();
      toggleRecitation();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      if (currentPage < TOTAL_PAGES) renderPage(currentPage + 1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      if (currentPage > 1) renderPage(currentPage - 1);
    }
  });

  window.addEventListener("pagehide", () => void activeMic?.stop());
}

void init();
