/**
 * Listener-dependent checker constants, derived by `tools/listener_calibration.py`.
 *
 * Soft costs are the negative log confusion rate on gold-excluded decodes, floored at today's
 * cost. The confidence cut is the clean-word quantile that keeps the shipped gate's pass rate.
 * `support` and `natsPerEdit` scale by the ratio of median per-phone CTC nll on words the
 * alignment matched. The phrase-switch penalty stays at today's value while the 99th percentile
 * of correct parallel-phrase margins is still under it, and follows a tail that crosses it.
 * The blank-cost cut keeps the shipped offset from the maximum
 * blank-forcing cost of a clean pause (today that offset lands on 2× support, 20 nats).
 *
 * On the shipped student's own decodes every value lands on the constant below, within the noise
 * the estimator's idempotence test allows. `ACTIVE` is the wide student's derivation.
 */

export const CONFUSABLE_PAIRS = ["هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس"];
export const SHORT_VOWELS = ["َ", "ُ", "ِ"];
const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");

export interface CalibrationStats {
  wordConf: number[];
  phoneNll: number[];
  sub: Record<string, number>;
  refCount: Record<string, number>;
  phraseMargins: number[];
  blankCosts: number[];
}

export interface Calibration {
  unsure: number;
  support: number;
  natsPerEdit: number;
  phraseSwitch: number;
  soft: number;
  costs: Record<string, number>;
  blankCut: number;
}

/** Constants the shipped student was tuned against. The estimator must reproduce these. */
export const TODAY: Calibration = {
  unsure: 0.8,
  support: 10,
  natsPerEdit: 3,
  phraseSwitch: 4,
  soft: 3,
  costs: floors(),
  blankCut: 20,
};

export interface Anchor {
  passRate: number;
  medianNll: number;
  phraseTail: number | null;
  blankMax: number | null;
  pairRates: Record<string, number>;
  vowelRate: number;
}

function floors(): Record<string, number> {
  const costs: Record<string, number> = {};
  for (const pair of CONFUSABLE_PAIRS) {
    costs[pair] = 5;
    costs[pair[1]! + pair[0]!] = 5;
  }
  for (let i = 0; i < SHORT_VOWELS.length; i++) {
    for (let j = i + 1; j < SHORT_VOWELS.length; j++) {
      costs[SHORT_VOWELS[i]! + SHORT_VOWELS[j]!] = 5;
      costs[SHORT_VOWELS[j]! + SHORT_VOWELS[i]!] = 5;
    }
  }
  return costs;
}

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[n >> 1]! : 0.5 * (s[n / 2 - 1]! + s[n / 2]!);
}

function tail(xs: number[], q = 0.99): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.ceil(q * s.length) - 1));
  return s[i]!;
}

function roundHalfUp(x: number): number {
  return Math.floor(x + 0.5);
}

export function quantileCut(confs: number[], passRate: number): number {
  if (!confs.length) return TODAY.unsure;
  const xs = [...confs].sort((a, b) => a - b);
  const i = Math.min(xs.length - 1, Math.max(0, Math.round(xs.length * (1 - passRate))));
  return xs[i]!;
}

function pairRate(stats: CalibrationStats, a: string, b: string): number {
  const n = (stats.refCount[a] ?? 0) + (stats.refCount[b] ?? 0);
  if (!n) return 0;
  return ((stats.sub[`${a}>${b}`] ?? 0) + (stats.sub[`${b}>${a}`] ?? 0)) / n;
}

function vowelRate(stats: CalibrationStats): number {
  let denom = 0;
  for (const [p, n] of Object.entries(stats.refCount)) if (!CONSONANTS.has(p)) denom += n;
  if (!denom) return 0;
  let num = 0;
  for (const [key, n] of Object.entries(stats.sub)) {
    const [a, b] = key.split(">");
    if (a && b && !CONSONANTS.has(a) && !CONSONANTS.has(b)) num += n;
  }
  return num / denom;
}

function pairRates(stats: CalibrationStats): Record<string, number> {
  const rates: Record<string, number> = {};
  for (const pair of CONFUSABLE_PAIRS) rates[pair] = pairRate(stats, pair[0]!, pair[1]!);
  for (let i = 0; i < SHORT_VOWELS.length; i++) {
    for (let j = i + 1; j < SHORT_VOWELS.length; j++) {
      rates[SHORT_VOWELS[i]! + SHORT_VOWELS[j]!] = pairRate(stats, SHORT_VOWELS[i]!, SHORT_VOWELS[j]!);
    }
  }
  return rates;
}

/** Negative log confusion rate, in units where the shipped rate costs exactly `floor`. */
function softCost(rate: number, floor: number, shippedRate: number): number {
  if (rate <= 0 || shippedRate <= 0) return floor;
  // A soft pair never prices above one plain edit. The discount shrinks; it does not become a super-edit.
  return Math.min(10, Math.max(floor, roundHalfUp(floor * Math.log(rate) / Math.log(shippedRate))));
}

export function anchorFrom(stats: CalibrationStats): Anchor {
  const confs = stats.wordConf;
  return {
    passRate: confs.length ? confs.filter((c) => c >= TODAY.unsure - 1e-12).length / confs.length : 1,
    medianNll: median(stats.phoneNll),
    phraseTail: tail(stats.phraseMargins),
    blankMax: stats.blankCosts.length ? Math.max(...stats.blankCosts) : null,
    pairRates: pairRates(stats),
    vowelRate: vowelRate(stats),
  };
}

export function derive(stats: CalibrationStats, anchor: Anchor): Calibration {
  const scale = anchor.medianNll ? median(stats.phoneNll) / anchor.medianNll : 1;
  const margins = stats.phraseMargins;
  const modelTail = tail(margins);
  // Today's penalty already clears a correct-phrase tail that sits below it. The gap
  // between two such tails is not headroom to add: both tails are under the penalty, so
  // the penalty still clears them. Follow a tail only once it crosses today's penalty.
  const phraseSwitch = modelTail !== null && modelTail > TODAY.phraseSwitch ? modelTail : TODAY.phraseSwitch;
  const costs: Record<string, number> = {};
  const shippedRates = anchor.pairRates;
  for (const pair of CONFUSABLE_PAIRS) {
    const cost = softCost(pairRate(stats, pair[0]!, pair[1]!), 5, shippedRates[pair] ?? 0);
    costs[pair] = cost;
    costs[pair[1]! + pair[0]!] = cost;
  }
  for (let i = 0; i < SHORT_VOWELS.length; i++) {
    for (let j = i + 1; j < SHORT_VOWELS.length; j++) {
      const a = SHORT_VOWELS[i]!, b = SHORT_VOWELS[j]!;
      const cost = softCost(pairRate(stats, a, b), 5, shippedRates[a + b] ?? 0);
      costs[a + b] = cost;
      costs[b + a] = cost;
    }
  }
  const blankMax = stats.blankCosts.length ? Math.max(...stats.blankCosts) : null;
  const blankCut = blankMax !== null && anchor.blankMax !== null ? 2 * TODAY.support + (blankMax - anchor.blankMax) : TODAY.blankCut;
  return {
    unsure: quantileCut(stats.wordConf, anchor.passRate),
    support: TODAY.support * scale,
    natsPerEdit: TODAY.natsPerEdit * scale,
    phraseSwitch,
    soft: softCost(vowelRate(stats), TODAY.soft, anchor.vowelRate),
    costs,
    blankCut,
  };
}

/** Differences a re-run of the shipped student is allowed to show. Empty means the constants reproduced. */
export function calibrationNoise(got: Calibration): string[] {
  const bad: string[] = [];
  if (Math.abs(got.unsure - TODAY.unsure) > 0.05) bad.push(`unsure ${got.unsure}`);
  if (Math.abs(got.support - TODAY.support) > 0.5) bad.push(`support ${got.support}`);
  if (Math.abs(got.natsPerEdit - TODAY.natsPerEdit) > 0.25) bad.push(`natsPerEdit ${got.natsPerEdit}`);
  if (Math.abs(got.phraseSwitch - TODAY.phraseSwitch) > 0.5) bad.push(`phraseSwitch ${got.phraseSwitch}`);
  if (got.soft !== TODAY.soft) bad.push(`soft ${got.soft}`);
  for (const [pair, cost] of Object.entries(TODAY.costs)) if (got.costs[pair] !== cost) bad.push(`cost ${pair} ${got.costs[pair]}`);
  if (Math.abs(got.blankCut - TODAY.blankCut) > 0.5) bad.push(`blankCut ${got.blankCut}`);
  return bad;
}

/**
 * Wide-student constants from `derive(wide, anchorFrom(shipped))` on the gold-excluded decodes
 * (60 readers, shipped augmentation). The idempotence test re-derives both and checks the shipped
 * student lands on `TODAY` and this object matches the wide student.
 */
const WIDE_COSTS = floors();
WIDE_COSTS["ثس"] = 6;
WIDE_COSTS["سث"] = 6;
export const ACTIVE: Calibration = {
  unsure: 0.7976594037479825,
  support: 9.600874953296223,
  natsPerEdit: 2.880262485988867,
  phraseSwitch: 4,
  soft: 3,
  costs: WIDE_COSTS,
  blankCut: 18.45123529434204,
};

export const SUPPORT = ACTIVE.support;
export const UNSURE = ACTIVE.unsure;
export const PHRASE_SWITCH = ACTIVE.phraseSwitch;
export const NATS_PER_EDIT = ACTIVE.natsPerEdit;
export const SOFT = ACTIVE.soft;
export const BLANK_CUT = ACTIVE.blankCut;

/** Per-pair soft cost, or null when the pair is not one of the soft pairs. */
export function pairCost(ref: string, obs: string): number | null {
  const cost = ACTIVE.costs[ref + obs];
  return cost === undefined ? null : cost;
}
