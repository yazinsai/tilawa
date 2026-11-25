import Foundation
import XCTest
@testable import Tilawa

final class TilawaTests: XCTestCase {
    private func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures")))
    }

    func testFeatureExtractionMatchesReferenceAcrossChunkBoundaries() throws {
        struct Fixture: Decodable { let samples: [Float]; let frames: [[Float]] }
        let input = try JSONDecoder().decode(Fixture.self, from: fixture("fbank"))
        let bank = Fbank()
        let output = bank.push(Array(input.samples[..<317])) + bank.push(Array(input.samples[317..<2057])) + bank.push(Array(input.samples[2057...]))
        XCTAssertEqual(output.count, input.frames.count)
        for (actual, expected) in zip(output.flatMap { $0 }, input.frames.flatMap { $0 }) {
            XCTAssertEqual(actual, expected, accuracy: 0.00002)
        }
    }

    func testResamplerMatchesReferenceAcrossChunkBoundaries() throws {
        struct Fixture: Decodable { let samples: [Float]; let output: [Float] }
        let input = try JSONDecoder().decode(Fixture.self, from: fixture("resampler"))
        let resampler = NativeResampler(from: 44100)
        let output = resampler.push(Array(input.samples[..<317])) + resampler.push(Array(input.samples[317..<2057])) + resampler.push(Array(input.samples[2057...])) + resampler.flush()
        XCTAssertEqual(output.count, input.output.count)
        for (actual, expected) in zip(output, input.output) { XCTAssertEqual(actual, expected, accuracy: 0.000001) }
    }

    func testWeightedAlignmentMatchesReference() throws {
        struct Fixture: Decodable { let reference: String; let observed: String; let start: Int; let end: Int; let cost: Double; let errorRate: Double }
        for input in try JSONDecoder().decode([Fixture].self, from: fixture("alignment")) {
            let ref = Array(input.reference.utf16)
            let output = try XCTUnwrap(PhonemeAlignment.find(reference: ref, range: 0..<ref.count, observed: Array(input.observed.utf16), starts: [0]))
            XCTAssertEqual(output.start, input.start); XCTAssertEqual(output.end, input.end)
            XCTAssertEqual(output.cost, input.cost); XCTAssertEqual(output.errorRate, input.errorRate)
        }
    }

    func testTrackerKeepsConfirmedReplacementWhenWindowCutsTheWord() throws {
        let ref = try ReferenceIndex(data: fixture("reference"))
        let first = try XCTUnwrap(ref.locations[WordLocation(surah: 36, ayah: 1, word: 1)])
        let target = ref.words[try XCTUnwrap(ref.locations[WordLocation(surah: 36, ayah: 3, word: 2)])]
        let end = ref.words[first + 26].end
        let observed = Array(ref.phonemes[ref.words[first].start..<target.start]) + Array("لَعَزِۦۦزِ".utf16) + Array(ref.phonemes[target.end..<end])
        let update = track(observed, reference: ref)
        XCTAssertTrue(update.reviews.contains { $0.location == target.location && $0.kind == .wrongWord })
    }

    func testTrackerDistinguishesCleanOmittedAddedAndSkippedWords() throws {
        let ref = try ReferenceIndex(data: fixture("reference"))
        let first = try XCTUnwrap(ref.locations[WordLocation(surah: 36, ayah: 1, word: 1)])
        let begin = ref.words[first].start, end = ref.words[first + 26].end
        let target = ref.words[first + 12]
        XCTAssertTrue(track(Array(ref.phonemes[begin..<end]), reference: ref).reviews.isEmpty)
        let omitted = Array(ref.phonemes[begin..<target.start]) + Array(ref.phonemes[target.end..<end])
        XCTAssertTrue(track(omitted, reference: ref).reviews.contains { $0.kind == .omittedWord && $0.location == target.location })
        let added = Array(ref.phonemes[begin..<target.end]) + Array("زززززز".utf16) + Array(ref.phonemes[target.end..<end])
        XCTAssertTrue(track(added, reference: ref).reviews.contains { $0.kind == .addedWord })
        let skipped = Array(ref.phonemes[begin..<target.start]) + Array(ref.phonemes[ref.words[first + 14].end..<end])
        XCTAssertTrue(track(skipped, reference: ref).reviews.contains { $0.kind == .skippedPassage })
    }

    func testReferenceRejectsMalformedWordBoundaries() throws {
        let invalid = Data(#"{"ayahs":[{"s":1,"a":1,"w":["x"],"p":"اب","o":[9],"iso":["اب"]}]}"#.utf8)
        XCTAssertThrowsError(try ReferenceIndex(data: invalid)) { XCTAssertEqual($0 as? TilawaError, .invalidReference) }
        XCTAssertThrowsError(try ReferenceIndex(data: Data(#"{"ayahs":[]}"#.utf8)))
    }

    func testSessionLifecycleAndSilence() async throws {
        let engine = try makeEngine()
        do { _ = try await engine.pushAudio([0]); XCTFail("push before start succeeded") }
        catch { XCTAssertEqual(error as? TilawaError, .sessionNotActive) }
        try await engine.startFreeRecitation()
        do { try await engine.startFreeRecitation(); XCTFail("concurrent take succeeded") }
        catch { XCTAssertEqual(error as? TilawaError, .sessionAlreadyActive) }
        let update = try await engine.pushAudio([Float](repeating: 0, count: 8000))
        XCTAssertNil(update.position); XCTAssertTrue(update.reviewPrompts.isEmpty)
        let ended = try await engine.end(), repeated = try await engine.end()
        XCTAssertEqual(ended.audioSeconds, 0.5); XCTAssertEqual(repeated.audioSeconds, ended.audioSeconds)
        do { _ = try await engine.pushAudio([0]); XCTFail("push after end succeeded") }
        catch { XCTAssertEqual(error as? TilawaError, .sessionNotActive) }
        try await engine.startFreeRecitation()
        let reset = try await engine.end()
        XCTAssertEqual(reset.audioSeconds, 0); XCTAssertNil(reset.position); XCTAssertTrue(reset.reviewPrompts.isEmpty)
    }

    func testInvalidAudioAndRateChangesDoNotPoisonTheTake() async throws {
        let engine = try makeEngine()
        try await engine.startFreeRecitation()
        do { _ = try await engine.pushAudio([.nan]); XCTFail("NaN was accepted") }
        catch { XCTAssertEqual(error as? TilawaError, .invalidAudio) }
        _ = try await engine.pushAudio([0], sampleRate: 16000)
        do { _ = try await engine.pushAudio([0], sampleRate: 44100); XCTFail("sample rate changed") }
        catch { XCTAssertEqual(error as? TilawaError, .sampleRateChanged) }
        let state = await engine.state
        XCTAssertEqual(state, .active)
        _ = try await engine.end()
    }

    func testInvalidBackendOutputFailsTakeAndCanReset() async throws {
        let backend = BlankBackend()
        let engine = try makeEngine(backend: backend)
        try await engine.startFreeRecitation()
        backend.invalidOutput = true
        do { _ = try await engine.pushAudio([Float](repeating: 0, count: 8000)); XCTFail("invalid output accepted") }
        catch { XCTAssertEqual(error as? TilawaError, .invalidModelOutput) }
        let failed = await engine.state
        XCTAssertEqual(failed, .failed)
        backend.invalidOutput = false
        try await engine.startFreeRecitation()
        _ = try await engine.pushAudio([Float](repeating: 0, count: 8000))
        let result = try await engine.end()
        XCTAssertTrue(result.reviewPrompts.isEmpty)
    }

    private func track(_ codes: [UInt16], reference: ReferenceIndex) -> TrackingUpdate {
        let tracker = NativeTracker(index: reference, passage: reference.passage(surah: 36, from: 1, to: 12))
        for offset in stride(from: 0, to: codes.count, by: 12) {
            _ = tracker.push(codes[offset..<min(codes.count, offset + 12)].enumerated().map {
                HeardPhoneme(code: $0.element, start: offset + $0.offset, end: offset + $0.offset, confidence: 1)
            })
        }
        return tracker.push([], final: true)
    }

    private func makeEngine(backend: BlankBackend = BlankBackend()) throws -> TilawaEngine {
        let manifest = Data(#"{"tokens":"ابتث","window":45,"shift":32,"frameMs":40,"fbank":{"mels":80,"padValue":-23.0259}}"#.utf8)
        return try TilawaEngine(reference: fixture("reference"), manifest: manifest, backend: backend)
    }
}

private final class BlankBackend: InferenceBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var invalid = false
    var invalidOutput: Bool {
        get { lock.lock(); defer { lock.unlock() }; return invalid }
        set { lock.lock(); defer { lock.unlock() }; invalid = newValue }
    }
    func reset() throws {}
    func run(features: [Float]) throws -> [Float] {
        lock.lock(); defer { lock.unlock() }
        if invalid { return [] }
        return (0..<40).map { $0 % 5 == 0 ? 0 : -10 }
    }
}
