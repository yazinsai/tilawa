// SPDX-License-Identifier: MIT
import Foundation

enum PhonemeCosts {
    static let unit = 10.0
    static let edit = 1.0 / 1_048_576
    static let vowels = Set("َُِ".utf16)
    static let held = Set("اۦۥں۾".utf16)
    static let doubled = Set("نم".utf16)
    static let pairs: Set<UInt32> = {
        var result = Set<UInt32>()
        for pair in ["هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس"] {
            let p = Array(pair.utf16)
            result.insert(UInt32(p[0]) << 16 | UInt32(p[1]))
            result.insert(UInt32(p[1]) << 16 | UInt32(p[0]))
        }
        return result
    }()
    static func substitution(_ ref: UInt16, _ obs: UInt16) -> Double {
        if ref == obs { return 0 }
        if pairs.contains(UInt32(ref) << 16 | UInt32(obs)) || (vowels.contains(ref) && vowels.contains(obs)) { return 5 }
        return unit
    }
    static func run(_ ref: [UInt16], at: Int) -> Int {
        var a = at, b = at
        while a > 0 && ref[a - 1] == ref[at] { a -= 1 }
        while b + 1 < ref.count && ref[b + 1] == ref[at] { b += 1 }
        return b - a + 1
    }
    static func lengthening(_ p: UInt16, count: Int) -> Bool {
        (held.contains(p) && count >= 2) || (doubled.contains(p) && count >= 3)
    }
    static func indel(_ p: UInt16, lengthened: Bool) -> Double {
        lengthened ? 2 : vowels.contains(p) ? 5 : unit
    }
    static func insertion(_ ref: [UInt16], at: Int, p: UInt16, end: Int) -> Double {
        let before = at > 0 && ref[at - 1] == p ? run(ref, at: at - 1) : 0
        let after = at < end && ref[at] == p ? run(ref, at: at) : 0
        return indel(p, lengthened: lengthening(p, count: max(before, after) + 1))
    }
    static func deletion(_ ref: [UInt16], at: Int) -> Double {
        indel(ref[at], lengthened: lengthening(ref[at], count: run(ref, at: at)))
    }
}

struct AlignmentOperation: Codable {
    enum Kind: String, Codable { case match, sub, del, ins }
    let kind: Kind
    let reference: Int?
    let observed: Int?
}

struct PhonemeAlignment {
    let start: Int
    let end: Int
    let cost: Double
    let errorRate: Double
    let operations: [AlignmentOperation]

    /// Port of Tilawa's weighted semi-global substring alignment, using UTF-16 phoneme units.
    static func find(reference ref: [UInt16], range: Range<Int>, observed obs: [UInt16], starts: Set<Int>, strictStart: Bool = false) -> PhonemeAlignment? {
        guard !range.isEmpty, !obs.isEmpty else { return nil }
        let lo = range.lowerBound, hi = range.upperBound, n = range.count, m = obs.count, width = m + 1
        var d = [Double](repeating: 0, count: (n + 1) * width)
        func startCost(_ i: Int) -> Double { starts.contains(lo + i) ? 0 : strictStart ? .infinity : 10 }
        func insertion(_ i: Int, _ j: Int) -> Double { PhonemeCosts.insertion(ref, at: lo + i, p: obs[j - 1], end: hi) + PhonemeCosts.edit }
        func deletion(_ i: Int) -> Double { PhonemeCosts.deletion(ref, at: lo + i - 1) + PhonemeCosts.edit }
        func substitution(_ i: Int, _ j: Int) -> Double {
            let a = ref[lo + i - 1], b = obs[j - 1]
            return a == b ? 0 : PhonemeCosts.substitution(a, b) + PhonemeCosts.edit
        }
        for i in 0...n {
            let row = i * width
            d[row] = startCost(i)
            for j in 1...m {
                var best = d[row + j - 1] + insertion(i, j)
                if i > 0 { best = min(best, d[row - width + j - 1] + substitution(i, j), d[row - width + j] + deletion(i)) }
                d[row + j] = best
            }
        }
        var endI = 0
        for i in 1...n where d[i * width + m] < d[endI * width + m] { endI = i }
        guard d[endI * width + m].isFinite else { return nil }
        var operations: [AlignmentOperation] = []
        var i = endI, j = m
        while j > 0 {
            let here = d[i * width + j]
            if i > 0 && here == d[(i - 1) * width + j - 1] + substitution(i, j) {
                operations.append(AlignmentOperation(kind: ref[lo + i - 1] == obs[j - 1] ? .match : .sub, reference: lo + i - 1, observed: j - 1))
                i -= 1; j -= 1
            } else if i > 0 && here == d[(i - 1) * width + j] + deletion(i) {
                operations.append(AlignmentOperation(kind: .del, reference: lo + i - 1, observed: nil)); i -= 1
            } else {
                operations.append(AlignmentOperation(kind: .ins, reference: nil, observed: j - 1)); j -= 1
            }
        }
        operations.reverse()
        let length = endI - i, edits = operations.filter { $0.kind != .match }.count
        return PhonemeAlignment(start: lo + i, end: lo + endI, cost: floor(d[endI * width + m]),
            errorRate: length > 0 ? Double(edits) / Double(length) : .infinity, operations: operations)
    }
}
