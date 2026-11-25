// SPDX-License-Identifier: MIT
import Foundation

struct ModelManifest: Decodable {
    let tokens: String
    let window: Int
    let shift: Int
    let frameMs: Int
    let fbank: Features
    struct Features: Decodable { let mels: Int; let padValue: Float }
}

/// Streaming inference and greedy CTC decoding, ported from Tilawa's listener.ts.
/// Call only on the engine's serial processing queue.
final class NativeListener {
    private let backend: any InferenceBackend
    private let manifest: ModelManifest
    private let tokens: [UInt16]
    private var fbank = Fbank()
    private var features: [[Float]] = []
    private var consumed = 0
    private(set) var featureFrames = 0
    private(set) var outputFrames = 0
    private var previous = 0
    private var carried = false

    init(manifest data: Data, backend: any InferenceBackend) throws {
        manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
        guard manifest.window > 0, manifest.window <= 4096, manifest.shift > 0,
              manifest.shift <= manifest.window, manifest.fbank.mels == 80,
              manifest.frameMs == 40, !manifest.tokens.isEmpty else {
            throw TilawaError.invalidManifest
        }
        tokens = Array(manifest.tokens.utf16)
        self.backend = backend
    }

    func reset() throws {
        try backend.reset(); fbank = Fbank(); features = []; consumed = 0
        featureFrames = 0; outputFrames = 0; previous = 0; carried = false
    }

    func push(_ samples: [Float]) throws -> [HeardPhoneme] {
        let frames = fbank.push(samples)
        featureFrames += frames.count
        features += frames
        var result: [HeardPhoneme] = []
        while features.count - consumed >= manifest.window {
            result += try step(keep: Int.max)
            consumed += manifest.shift
        }
        if consumed > 4 * manifest.window { features.removeFirst(consumed); consumed = 0 }
        return result
    }

    func end() throws -> [HeardPhoneme] {
        let total = Int(floor((floor(Double(featureFrames - 7) / 2) + 1) / 2))
        var result: [HeardPhoneme] = []
        while outputFrames < total {
            while features.count - consumed < manifest.window {
                features.append([Float](repeating: manifest.fbank.padValue, count: manifest.fbank.mels))
            }
            result += try step(keep: total - outputFrames)
            consumed += manifest.shift
        }
        return result
    }

    private func step(keep: Int) throws -> [HeardPhoneme] {
        let window = features[consumed..<consumed + manifest.window].flatMap { $0 }
        let values = try backend.run(features: window)
        let vocabulary = tokens.count + 1
        guard !values.isEmpty, values.count % vocabulary == 0, values.allSatisfy({ $0.isFinite }) else { throw TilawaError.invalidModelOutput }
        let count = min(keep, values.count / vocabulary)
        var open: HeardPhoneme?
        var result: [HeardPhoneme] = []
        for t in 0..<count {
            let offset = t * vocabulary
            var id = 0
            for k in 1..<vocabulary where values[offset + k] > values[offset + id] { id = k }
            let probability = (exp(Double(values[offset + id])) * 10000).rounded() / 10000
            let frame = outputFrames + t
            if id != 0 && id == previous && open != nil {
                open!.end = frame
                open!.confidence = max(open!.confidence, probability)
            } else if !(id != 0 && id == previous && carried) {
                if let value = open { result.append(value) }
                open = id == 0 ? nil : HeardPhoneme(code: tokens[id - 1], start: frame, end: frame, confidence: probability)
                carried = false
            }
            if id != previous { carried = false }
            previous = id
        }
        outputFrames += count
        if let value = open { result.append(value); carried = true }
        return result
    }
}
