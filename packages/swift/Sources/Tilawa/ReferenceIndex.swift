// SPDX-License-Identifier: MIT
import Foundation

struct ReferenceAsset: Decodable {
    let ayahs: [Ayah]
    struct Ayah: Decodable {
        let s: Int, a: Int
        let w: [String], p: String, o: [Int], iso: [String]
    }
}

struct ReferenceWord {
    let location: WordLocation
    let start: Int
    let end: Int
    let text: String
    let isolated: [UInt16]
}

final class ReferenceIndex {
    let phonemes: [UInt16]
    let words: [ReferenceWord]
    let starts: Set<Int>
    let grams: [UInt64: [Int]]
    let locations: [WordLocation: Int]

    init(data: Data) throws {
        let asset = try JSONDecoder().decode(ReferenceAsset.self, from: data)
        var all: [UInt16] = [], list: [ReferenceWord] = []
        guard !asset.ayahs.isEmpty else { throw TilawaError.invalidReference }
        var seen = Set<WordLocation>()
        for ayah in asset.ayahs {
            let base = all.count, p = Array(ayah.p.utf16)
            guard (1...114).contains(ayah.s), ayah.a > 0, !ayah.w.isEmpty,
                  ayah.w.count == ayah.o.count, ayah.w.count == ayah.iso.count,
                  ayah.o.first == 0, !p.isEmpty else { throw TilawaError.invalidReference }
            for i in ayah.o.indices {
                let end = i + 1 < ayah.o.count ? ayah.o[i + 1] : p.count
                guard ayah.o[i] >= 0, end > ayah.o[i], end <= p.count,
                      seen.insert(WordLocation(surah: ayah.s, ayah: ayah.a, word: i + 1)).inserted else {
                    throw TilawaError.invalidReference
                }
            }
            for i in ayah.w.indices {
                list.append(ReferenceWord(location: WordLocation(surah: ayah.s, ayah: ayah.a, word: i + 1),
                    start: base + ayah.o[i], end: base + (i + 1 < ayah.o.count ? ayah.o[i + 1] : p.count),
                    text: ayah.w[i], isolated: Array(ayah.iso[i].utf16)))
            }
            all += p
        }
        phonemes = all; words = list; starts = Set(list.map(\.start))
        locations = Dictionary(uniqueKeysWithValues: list.enumerated().map { ($0.element.location, $0.offset) })
        var postings: [UInt64: [Int]] = [:]
        if all.count >= 4 {
            for i in 0...all.count - 4 { postings[Self.gram(all, i), default: []].append(i) }
        }
        grams = postings
    }

    static func gram(_ values: [UInt16], _ i: Int) -> UInt64 {
        UInt64(values[i]) << 48 | UInt64(values[i + 1]) << 32 | UInt64(values[i + 2]) << 16 | UInt64(values[i + 3])
    }
    func word(at offset: Int) -> Int {
        var lo = 0, hi = words.count - 1
        while lo < hi {
            let middle = (lo + hi + 1) / 2
            if words[middle].start <= offset { lo = middle } else { hi = middle - 1 }
        }
        return lo
    }
    func passage(surah: Int, from: Int, to: Int) -> Range<Int>? {
        guard let first = locations[WordLocation(surah: surah, ayah: from, word: 1)] else { return nil }
        let last = words.lastIndex { $0.location.surah == surah && $0.location.ayah == to }
        guard let last, first <= last else { return nil }
        return words[first].start..<words[last].end
    }
}
