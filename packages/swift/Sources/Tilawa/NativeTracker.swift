// SPDX-License-Identifier: MIT
import Foundation

struct HeardPhoneme: Codable, Sendable {
    let code: UInt16
    let start: Int
    var end: Int
    var confidence: Double
}

public struct ReviewItem: Codable, Identifiable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable {
        case wrongWord = "wrong_word", omittedWord = "omitted_word", addedWord = "added_word", skippedPassage = "skipped_passage"

    }
    public let location: WordLocation
    public let kind: Kind
    public var id: String { "\(location.surah):\(location.ayah):\(location.word):\(kind.rawValue)" }
}

struct TrackingUpdate {
    var state = "acquiring"
    var position: WordLocation?
    var confidence = 0.0
    var reviews: [ReviewItem] = []
}

/// Custom Swift tracker. Uses Tilawa's phoneme costs and model, with conservative n-gram
/// candidate acquisition and bounded alignment. This is not a port of the SDK's full checker.
final class NativeTracker {
    private let index: ReferenceIndex
    private let passage: Range<Int>?
    private var heard: [HeardPhoneme] = []
    private var lastWord: Int?
    private var tentativeWord: Int?
    private var stable = 0
    private var reviewCandidates: [String: (ReviewItem, Int)] = [:]
    private var confirmed: [String: ReviewItem] = [:]
    private(set) var update = TrackingUpdate()

    init(index: ReferenceIndex, passage: Range<Int>? = nil) { self.index = index; self.passage = passage }

    func push(_ phonemes: [HeardPhoneme], final: Bool = false) -> TrackingUpdate {
        guard !phonemes.isEmpty || final else { return update }
        heard += phonemes
        if heard.count > 160 { heard.removeFirst(heard.count - 160) }
        guard heard.count >= 16 else { return update }
        let observed = heard.map(\.code)
        let meanConfidence = heard.reduce(0) { $0 + $1.confidence } / Double(heard.count)
        guard meanConfidence >= 0.45 else { update.state = "uncertain"; return update }
        let candidates = search(observed)
        guard let best = candidates.first else { update.state = "lost"; return update }
        let word = index.word(at: max(best.start, best.end - 1))
        let fit = max(0, 1 - best.cost / (Double(observed.count) * 10))
        let runner = candidates.dropFirst().first { abs(index.word(at: max($0.start, $0.end - 1)) - word) > 2 }
        let margin = runner.map { max(0, $0.cost - best.cost) / max(10, Double(observed.count) * 1.5) } ?? 1
        let continuation = lastWord.map { abs($0 - word) < 24 && index.words[$0].location.surah == index.words[word].location.surah } ?? false
        let confidence = fit * min(1, continuation ? max(margin, 0.85) : margin)
        guard fit >= 0.70, confidence >= 0.60 else {
            update.state = fit >= 0.70 ? "ambiguous" : "lost"
            update.confidence = confidence
            return update
        }
        if let tentativeWord, abs(tentativeWord - word) <= 16 { stable += 1 } else { stable = 1 }
        tentativeWord = word
        guard stable >= 2 || final else { update.state = "acquiring"; return update }
        lastWord = word
        update.state = "locked"; update.position = index.words[word].location; update.confidence = confidence
        // Require two independently extended alignments before presenting a review prompt.
        // Weak fits do not create new memorization accusations.
        if fit >= 0.80 {
            let items = review(best, currentWord: word)
            let active = Set(items.map(\.id))
            for key in Array(reviewCandidates.keys) where !active.contains(key) { reviewCandidates.removeValue(forKey: key) }
            // The first word may have been cut by the rolling audio window, and the
            // current word is unfinished. Neither provides evidence of a correction.
            // Resolve only words that the checker could assess in full this time.
            let firstAssessable = index.word(at: best.start)
            for key in Array(confirmed.keys) where !active.contains(key) {
                if let old = confirmed[key], let at = index.locations[old.location], at > firstAssessable && at < word {
                    confirmed.removeValue(forKey: key)
                }
            }
            for item in items {
                let count = (reviewCandidates[item.id]?.1 ?? 0) + (phonemes.isEmpty ? 0 : 1)
                reviewCandidates[item.id] = (item, count)
                if count >= 2 { confirmed[item.id] = item }
            }
        }
        update.reviews = confirmed.values.sorted {
            (index.locations[$0.location] ?? 0) < (index.locations[$1.location] ?? 0)
        }
        return update
    }

    private func search(_ obs: [UInt16]) -> [PhonemeAlignment] {
        let allowed = passage ?? 0..<index.phonemes.count
        var ranges: [Range<Int>] = []
        if let lastWord {
            let center = index.words[lastWord]
            let lo = max(allowed.lowerBound, center.start - 280)
            let hi = min(allowed.upperBound, center.end + 360)
            if lo < hi { ranges.append(lo..<hi) }
        }
        var votes: [Int: Int] = [:]
        let grams = (0...obs.count - 4).compactMap { i -> (Int, [Int])? in
            guard let positions = index.grams[ReferenceIndex.gram(obs, i)], positions.count < 3000 else { return nil }
            return (i, positions)
        }.sorted { $0.1.count < $1.1.count }.prefix(20)
        for (offset, postings) in grams {
            for position in postings where position >= allowed.lowerBound && position < allowed.upperBound {
                let origin = position - offset
                votes[Int(floor(Double(origin) / 8)), default: 0] += 1
            }
        }
        for (bin, _) in votes.sorted(by: {
            if $0.value != $1.value { return $0.value > $1.value }; return $0.key < $1.key
        }).prefix(20) {
            let lo = max(allowed.lowerBound, bin * 8 - 20)
            let hi = min(allowed.upperBound, bin * 8 + obs.count + 32)
            if lo < hi { ranges.append(lo..<hi) }
        }
        if ranges.isEmpty, let passage, passage.count < 3000 { ranges.append(passage) }
        var found: [Int: PhonemeAlignment] = [:]
        for range in Set(ranges) {
            guard let result = PhonemeAlignment.find(reference: index.phonemes, range: range, observed: obs, starts: index.starts), result.end > result.start else { continue }
            let word = index.word(at: result.end - 1)
            if let old = found[word], old.cost <= result.cost { continue }
            found[word] = result
        }
        return found.values.sorted {
            if $0.cost != $1.cost { return $0.cost < $1.cost }
            if let lastWord {
                let a = abs(index.word(at: $0.end - 1) - lastWord), b = abs(index.word(at: $1.end - 1) - lastWord)
                if a != b { return a < b }
            }
            if $0.errorRate != $1.errorRate { return $0.errorRate < $1.errorRate }
            return $0.start < $1.start
        }
    }

    private func review(_ alignment: PhonemeAlignment, currentWord: Int) -> [ReviewItem] {
        struct Tally {
            var matches = 0; var substitutions = 0; var deletions = 0
            var matchingConsonants = 0; var lexicalSubstitutions = 0
        }
        var tallies: [Int: Tally] = [:]
        var insertions: [Int: [UInt16]] = [:]
        var anchor = index.word(at: alignment.start)
        for op in alignment.operations {
            if let reference = op.reference {
                anchor = index.word(at: reference)
                switch op.kind {
                case .match:
                    tallies[anchor, default: Tally()].matches += 1
                    if isConsonant(index.phonemes[reference]) { tallies[anchor, default: Tally()].matchingConsonants += 1 }
                case .sub:
                    tallies[anchor, default: Tally()].substitutions += 1
                    if let observed = op.observed, isConsonant(index.phonemes[reference]),
                       PhonemeCosts.substitution(index.phonemes[reference], heard[observed].code) >= PhonemeCosts.unit {
                        tallies[anchor, default: Tally()].lexicalSubstitutions += 1
                    }
                case .del: tallies[anchor, default: Tally()].deletions += 1
                case .ins: break
                }
            } else if let observed = op.observed { insertions[anchor, default: []].append(heard[observed].code) }
        }
        let first = index.word(at: alignment.start)
        var items: [ReviewItem] = []
        // Starting part-way through a word and the current unfinished word are never graded.
        for (word, tally) in tallies where word > first && word < currentWord {
            let total = index.words[word].end - index.words[word].start
            let consonants = index.phonemes[index.words[word].start..<index.words[word].end].filter(isConsonant).count
            // Common vowels must not hide a different lexical word. Require at least
            // two full-cost consonant substitutions; confusable sounds alone do not qualify.
            let lexicalMismatch = consonants >= 2 && Double(tally.matchingConsonants) / Double(consonants) < 0.50
                && tally.lexicalSubstitutions >= 2
            // A skipped word can share a few boundary phonemes with its neighbors.
            // Treat it as omitted when most of its sound was deleted, rather than
            // requiring a perfect all-deletion path through an ambiguous boundary.
            let mostlyDeleted = total >= 4 && tally.deletions >= max(3, Int(ceil(Double(total) * 0.60))) && tally.matches <= 2
            if (tally.matches == 0 && tally.deletions >= total) || mostlyDeleted {
                items.append(ReviewItem(location: index.words[word].location, kind: .omittedWord))
            } else if lexicalMismatch || (total >= 4 && Double(tally.matches) / Double(total) < 0.40 && tally.substitutions >= 3) {
                items.append(ReviewItem(location: index.words[word].location, kind: .wrongWord))
            }
        }
        for (word, phonemes) in insertions where word > first && word < currentWord {
            // Held vowels and ghunna are pronunciation variation, not extra words.
            let consonants = phonemes.filter { !PhonemeCosts.vowels.contains($0) && !PhonemeCosts.held.contains($0) }
            if consonants.count >= 4 {
                items.append(ReviewItem(location: index.words[word].location, kind: .addedWord))
            }
        }
        let omitted = items.filter { $0.kind == .omittedWord }.sorted { (index.locations[$0.location] ?? 0) < (index.locations[$1.location] ?? 0) }
        var run: [ReviewItem] = []
        func flushRun() {
            if run.count >= 3 {
                let ids = Set(run.map(\.id)); items.removeAll { ids.contains($0.id) }
                items.append(ReviewItem(location: run[0].location, kind: .skippedPassage))
            }
            run = []
        }
        for item in omitted {
            if let last = run.last, (index.locations[item.location] ?? 0) != (index.locations[last.location] ?? 0) + 1 { flushRun() }
            run.append(item)
        }
        flushRun()
        return items
    }

    private func isConsonant(_ code: UInt16) -> Bool {
        !PhonemeCosts.vowels.contains(code) && !PhonemeCosts.held.contains(code)
    }
}
