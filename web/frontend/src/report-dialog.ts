import { encodeWav, concatChunks } from "./lib/wav-encoder";
import type { QuranVerse } from "./lib/types";

interface ReportDialogOptions {
  audioChunks: Float32Array[];
  modelPrediction: { surah: number; ayah: number; confidence: number } | null;
  quranData: QuranVerse[];
  debugBundle?: unknown;
}

const $dialog = document.getElementById("report-dialog") as HTMLDialogElement;
const $surah = document.getElementById("report-surah") as HTMLSelectElement;
const $ayah = document.getElementById("report-ayah") as HTMLSelectElement;
const $audio = document.getElementById("report-audio") as HTMLAudioElement;
const $prediction = document.getElementById("report-prediction")!;
const $notes = document.getElementById("report-notes") as HTMLTextAreaElement;
const $submit = document.getElementById("btn-submit-report")!;
const $cancel = document.getElementById("btn-cancel-report")!;
const $status = document.getElementById("report-status")!;

let currentAudioBlob: Blob | null = null;
let currentQuranData: QuranVerse[] = [];
let currentDebugBundle: unknown = null;

// Build surah list (called once after quran.json loads)
export function initSurahDropdown(quranData: QuranVerse[]): void {
  currentQuranData = quranData;
  const surahs = new Map<number, { name: string; nameEn: string }>();
  for (const v of quranData) {
    if (!surahs.has(v.surah)) {
      surahs.set(v.surah, { name: v.surah_name, nameEn: v.surah_name_en });
    }
  }
  $surah.innerHTML = "";
  for (const [num, info] of surahs) {
    const opt = document.createElement("option");
    opt.value = String(num);
    opt.textContent = `${num}. ${info.nameEn} — ${info.name}`;
    $surah.appendChild(opt);
  }
  $surah.addEventListener("change", () => updateAyahDropdown(parseInt($surah.value)));
}

function updateAyahDropdown(surahNum: number): void {
  const verses = currentQuranData.filter(v => v.surah === surahNum);
  $ayah.innerHTML = "";
  for (const v of verses) {
    const opt = document.createElement("option");
    opt.value = String(v.ayah);
    opt.textContent = `Ayah ${v.ayah}`;
    $ayah.appendChild(opt);
  }
}

export function openReportDialog(opts: ReportDialogOptions): void {
  currentDebugBundle = opts.debugBundle ?? null;

  // Encode audio
  const combined = concatChunks(opts.audioChunks);
  currentAudioBlob = encodeWav(combined);
  const url = URL.createObjectURL(currentAudioBlob);
  $audio.src = url;

  // Pre-fill with model prediction
  if (opts.modelPrediction) {
    $surah.value = String(opts.modelPrediction.surah);
    updateAyahDropdown(opts.modelPrediction.surah);
    $ayah.value = String(opts.modelPrediction.ayah);
    const pred = opts.modelPrediction;
    $prediction.textContent = `Surah ${pred.surah}, Ayah ${pred.ayah} (${Math.round(pred.confidence * 100)}% confidence)`;
  } else {
    $surah.value = "1";
    updateAyahDropdown(1);
    $prediction.textContent = "No prediction available";
  }

  $notes.value = "";
  $status.hidden = true;
  $submit.removeAttribute("disabled");
  $dialog.showModal();
}

// Submit handler
$submit.addEventListener("click", async () => {
  if (!currentAudioBlob) return;
  $submit.setAttribute("disabled", "true");
  $status.textContent = "Submitting...";
  $status.hidden = false;

  const metadata = {
    surah: parseInt($surah.value),
    ayah: parseInt($ayah.value),
    modelPrediction: $prediction.textContent,
    notes: $notes.value.trim(),
    debugBundle: currentDebugBundle,
  };

  const formData = new FormData();
  formData.append("audio", currentAudioBlob, "recording.wav");
  formData.append("metadata", JSON.stringify(metadata));

  try {
    const res = await fetch("/api/reports", { method: "POST", body: formData });
    if (res.ok) {
      $status.textContent = "Report submitted. Thank you!";
      setTimeout(() => $dialog.close(), 1500);
    } else {
      const err = await res.json();
      $status.textContent = `Error: ${err.error || "Unknown error"}`;
      $submit.removeAttribute("disabled");
    }
  } catch (e) {
    $status.textContent = "Network error. Please try again.";
    $submit.removeAttribute("disabled");
  }
});

// Cancel handler
$cancel.addEventListener("click", () => {
  $dialog.close();
  if ($audio.src) URL.revokeObjectURL($audio.src);
});
