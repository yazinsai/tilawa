// SPDX-License-Identifier: MIT
import Foundation

/// A model adapter owned by one engine. Implementations must serialize access to any
/// mutable runtime state; the engine itself calls this interface only from its actor.
public protocol InferenceBackend: Sendable {
    /// A flattened [1, manifest.window, 80] Float32 feature tensor; returns flattened log_probs.
    /// The backend retains and advances the recurrent ONNX tensors between calls.
    func run(features: [Float]) throws -> [Float]
    func reset() throws
}

public struct WordLocation: Codable, Hashable, Sendable {
    public let surah: Int
    public let ayah: Int
    public let word: Int
    public init(surah: Int, ayah: Int, word: Int) { self.surah = surah; self.ayah = ayah; self.word = word }
}

public struct MemorizationPassage: Equatable, Sendable {
    public let surah: Int
    public let fromAyah: Int
    public let toAyah: Int
    public init(surah: Int, fromAyah: Int, toAyah: Int) { self.surah = surah; self.fromAyah = fromAyah; self.toAyah = toAyah }
}

public enum TrackingState: String, Codable, Sendable { case acquiring, locked, ambiguous, lost, uncertain }
public enum EngineState: String, Sendable { case idle, active, ended, failed }

public struct RecitationUpdate: Sendable {
    public let tracking: TrackingState
    /// The last confidently locked position; uncertain updates do not move it.
    public let position: WordLocation?
    public let confidence: Double
    public let reviewPrompts: [ReviewItem]
    public let audioSeconds: Double
    public let ayahsFollowed: Int
}

public enum TilawaError: Error, Equatable, LocalizedError {
    case invalidManifest, invalidReference, invalidModelOutput, invalidPassage
    case sessionAlreadyActive, sessionNotActive, invalidAudio, sampleRateChanged
    public var errorDescription: String? {
        switch self {
        case .invalidManifest: "Unsupported or invalid model manifest."
        case .invalidReference: "Invalid Quran reference asset."
        case .invalidModelOutput: "The inference backend returned invalid log probabilities."
        case .invalidPassage: "The passage is not present in the reference."
        case .sessionAlreadyActive: "End the current take before starting another."
        case .sessionNotActive: "Start a take before supplying audio."
        case .invalidAudio: "Expected finite mono samples in [-1, 1] at a supported sample rate."
        case .sampleRateChanged: "Keep the sample rate constant within a take."
        }
    }
}

/// Experimental native Swift pipeline using Tilawa's model/reference assets.
///
/// One engine owns one backend and processes one take at a time. This Swift tracker
/// and checker do not reproduce the original TypeScript checker's full behavior;
/// published SDK accuracy figures do not apply. Feed mono PCM, not compressed audio.
public actor TilawaEngine {
    public private(set) var state: EngineState = .idle
    private let listener: NativeListener
    private let reference: ReferenceIndex
    private var tracker: NativeTracker?
    private var resampler: NativeResampler?
    private var sampleRate: Double?
    private var sampleCount = 0
    private var followed: Set<String> = []
    private var lastPosition: WordLocation?
    private var last = RecitationUpdate(tracking: .acquiring, position: nil, confidence: 0,
                                        reviewPrompts: [], audioSeconds: 0, ayahsFollowed: 0)

    public init(reference: Data, manifest: Data, backend: any InferenceBackend) throws {
        self.reference = try ReferenceIndex(data: reference)
        self.listener = try NativeListener(manifest: manifest, backend: backend)
    }

    public func startFreeRecitation() throws { try begin(passage: nil) }

    public func startMemorization(_ passage: MemorizationPassage) throws {
        guard passage.fromAyah > 0, passage.toAyah >= passage.fromAyah,
              let range = reference.passage(surah: passage.surah, from: passage.fromAyah, to: passage.toAyah) else {
            throw TilawaError.invalidPassage
        }
        try begin(passage: range)
    }

    private func begin(passage: Range<Int>?) throws {
        guard state != .active else { throw TilawaError.sessionAlreadyActive }
        try listener.reset()
        tracker = NativeTracker(index: reference, passage: passage)
        sampleRate = nil; resampler = nil; sampleCount = 0; followed = []; lastPosition = nil
        last = RecitationUpdate(tracking: .acquiring, position: nil, confidence: 0,
                                reviewPrompts: [], audioSeconds: 0, ayahsFollowed: 0)
        state = .active
    }

    @discardableResult
    public func pushAudio(_ samples: [Float], sampleRate rate: Double = 16000) throws -> RecitationUpdate {
        guard state == .active else { throw TilawaError.sessionNotActive }
        guard rate.isFinite, rate >= 8000, rate <= 192000,
              samples.allSatisfy({ $0.isFinite && abs($0) <= 1 }) else { throw TilawaError.invalidAudio }
        if let sampleRate, sampleRate != rate { throw TilawaError.sampleRateChanged }
        if sampleRate == nil { sampleRate = rate; resampler = NativeResampler(from: rate) }
        do {
            let pcm = resampler?.push(samples) ?? samples
            sampleCount += pcm.count
            return receive(try listener.push(pcm))
        } catch { state = .failed; throw error }
    }

    /// Drains buffered audio and finalizes the take. Repeated calls return the same result.
    @discardableResult
    public func end() throws -> RecitationUpdate {
        if state == .ended { return last }
        guard state == .active else { throw TilawaError.sessionNotActive }
        do {
            if let tail = resampler?.flush(), !tail.isEmpty {
                sampleCount += tail.count
                _ = receive(try listener.push(tail))
            }
            let result = receive(try listener.end(), final: true)
            state = .ended
            return result
        } catch { state = .failed; throw error }
    }

    private func receive(_ evidence: [HeardPhoneme], final: Bool = false) -> RecitationUpdate {
        let update = tracker?.push(evidence, final: final) ?? TrackingUpdate()
        if update.state == "locked", let position = update.position {
            lastPosition = position; followed.insert("\(position.surah):\(position.ayah)")
        }
        last = RecitationUpdate(tracking: TrackingState(rawValue: update.state) ?? .uncertain,
            position: lastPosition, confidence: update.confidence, reviewPrompts: update.reviews,
            audioSeconds: Double(sampleCount) / 16000, ayahsFollowed: followed.count)
        return last
    }
}
