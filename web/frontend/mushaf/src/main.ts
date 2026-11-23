import {
  displayIndex,
  indexQuran,
  type QuranText,
  type QuranVerse,
} from "../../src/lib/quran-words.js";
import {
  getAyahsForPage,
  getJuzForPage,
  getPageForVerse,
  toArabicDigits,
  TOTAL_PAGES,
} from "../../src/lib/mushaf.js";
import type {
  CheckerEvent,
  EngineEvent,
  PositionEvent,
  TilawaSession,
  WordRef,
} from "@tilawa/core";
import type {
  Tilawa,
  Microphone,
} from "@tilawa/core/web";

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const ASSET_BASE = new URL(
  import.meta.env.VITE_TILAWA_ASSET_BASE ?? "/v1/",
  location.href
).href;

// ---------------- State ----------------
let quran: QuranText | null = null;
let currentPage = 1;
let tilawa: Tilawa | null = null;
let activeSession: TilawaSession | null = null;
let activeMic: Microphone | null = null;
let isReciting = false;
let currentHighlightedWordEl: HTMLElement | null = null;

const mistakeList: Array<{
  id: string;
  errorClass: string;
  location: WordRef;
  timeMs: number;
}> = [];

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
    const res = await fetch("/quran.json");
    if (!res.ok) throw new Error("Failed to load Quran text");
    const verses: QuranVerse[] = await res.json();
    quran = indexQuran(verses);
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

      // Click word to view info
      wordSpan.addEventListener("click", () => {
        setMonitorPosition(ayah.surah, ayah.ayah, wordIdx + 1, wordText, 1.0);
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
  applyMistakeMarkersOnCurrentPage();
}

function updatePageControls() {
  $<HTMLInputElement>("page-input").value = String(currentPage);
  $<HTMLButtonElement>("btn-prev-page").disabled = currentPage <= 1;
  $<HTMLButtonElement>("btn-next-page").disabled = currentPage >= TOTAL_PAGES;
}

// ---------------- Engine & Recitation ----------------
async function toggleRecitation() {
  if (isReciting) {
    await stopRecitation();
  } else {
    await startRecitation();
  }
}

async function startRecitation() {
  const btn = $<HTMLButtonElement>("btn-start");
  const btnLabel = $("btn-start-label");

  btn.disabled = true;
  setStatus("loading", "Initializing engine...");

  try {
    if (!tilawa) {
      showLoadingBanner(true, "Downloading & warming up model...");
      const [web] = await Promise.all([
        import("@tilawa/core/web"),
        import("onnxruntime-web/all"),
      ]);

      tilawa = await web.createTilawa({
        assetBase: ASSET_BASE,
        tier: "auto",
        onProgress: (p) => {
          const pct = Math.floor(p.fraction * 100);
          $<HTMLDivElement>("loading-progress").style.width = `${pct}%`;
          $("loading-pct").textContent = `${pct}%`;
          if (p.stage === "model") {
            $("loading-msg").textContent = `Loading acoustic model (${pct}%)...`;
          } else if (p.stage === "warmup") {
            $("loading-msg").textContent = "Warming up engine...";
          }
        },
      });
      showLoadingBanner(false);
    }

    activeSession = tilawa.startFreeRecitation();
    activeSession.subscribe(handleEngineEvent);

    activeMic = await tilawa.listen(activeSession);
    isReciting = true;

    btn.classList.add("active");
    btnLabel.textContent = "Stop Reciting";
    btn.disabled = false;
    $("audio-level").hidden = false;
    setStatus("listening", "Listening... Start reciting anywhere");
  } catch (err: unknown) {
    showLoadingBanner(false);
    btn.disabled = false;
    isReciting = false;
    const msg = err instanceof Error ? err.message : String(err);
    console.error("Recitation start failed:", err);
    setStatus("idle", `Failed to start: ${msg}`);
  }
}

async function stopRecitation() {
  const btn = $<HTMLButtonElement>("btn-start");
  btn.disabled = true;

  if (activeMic) {
    try {
      await activeMic.stop();
    } catch {
      // ignore
    }
    activeMic = null;
  }

  if (activeSession) {
    try {
      await activeSession.end();
    } catch {
      // ignore
    }
    activeSession = null;
  }

  isReciting = false;
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

    setMonitorPosition(surah, ayah, word, wordEl.textContent ?? "", top.confidence);
  }

  setStatus("locked", `Following: Surah ${surah}:${ayah} (Page ${targetPage})`);
}

function handleChecker(e: CheckerEvent) {
  if (e.state !== "confirmed") return;

  mistakeList.push({
    id: e.id,
    errorClass: e.errorClass,
    location: e.location,
    timeMs: e.timeMs,
  });

  renderMistakes();
  applyMistakeMarkersOnCurrentPage();
}

function setMonitorPosition(
  surah: number,
  ayah: number,
  word: number,
  text: string,
  confidence: number
) {
  $("mon-surah").textContent = String(surah);
  $("mon-ayah").textContent = String(ayah);
  $("mon-word").textContent = String(word);
  $("mon-word-text").textContent = text;
  $("mon-confidence").textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
}

function renderMistakes() {
  const listEl = $("monitor-mistakes");
  $("mistake-count").textContent = String(mistakeList.length);

  if (!mistakeList.length) {
    listEl.innerHTML = `<li class="empty-state">No mistakes recorded yet. Recite smoothly.</li>`;
    return;
  }

  listEl.innerHTML = mistakeList
    .slice()
    .reverse()
    .map((m) => {
      const typeLabel = m.errorClass.replace(/_/g, " ");
      return `
        <li class="mistake-item">
          <div class="mistake-item-header">
            <span class="mistake-type">${typeLabel}</span>
            <span class="mistake-loc">${m.location.surah}:${m.location.ayah}</span>
          </div>
          <div class="mistake-detail">Around word ${m.location.word} (${(m.timeMs / 1000).toFixed(1)}s)</div>
        </li>
      `;
    })
    .join("");
}

function applyMistakeMarkersOnCurrentPage() {
  mistakeList.forEach((m) => {
    const at = displayIndex(m.location.surah, m.location.ayah, m.location.word);
    const el = document.getElementById(`w-${m.location.surah}-${m.location.ayah}-${at}`);
    if (el) {
      el.classList.add("mistake");
      el.title = `Mistake flagged: ${m.errorClass}`;
    }
  });
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
      void toggleRecitation();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      if (currentPage < TOTAL_PAGES) renderPage(currentPage + 1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      if (currentPage > 1) renderPage(currentPage - 1);
    }
  });
}

void init();
