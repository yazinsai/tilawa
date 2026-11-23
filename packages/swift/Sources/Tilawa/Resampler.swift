// SPDX-License-Identifier: MIT
import Foundation

/// Tilawa's streaming, low-pass windowed-sinc resampler (MIT).
/// Converts host-supplied mono PCM to the model's 16 kHz input rate.
final class NativeResampler {
    let from: Double
    let to: Double
    private let ratio: Double
    private let cutoff: Double
    private var history: [Float] = []
    private var position = 0.0
    init(from: Double, to: Double = 16000) {
        self.from = from; self.to = to; ratio = from / to; cutoff = min(1, to / from) * 0.95
    }
    func push(_ input: [Float]) -> [Float] {
        if from == to { return input }
        let buffer = history + input, taps = Int(ceil(16 / cutoff))
        var output: [Float] = []
        while position + Double(taps) < Double(buffer.count) {
            let center = Int(floor(position))
            var sample = 0.0, norm = 0.0
            for k in center - taps + 1...center + taps {
                let x = (Double(k) - position) * cutoff
                let window = 0.5 + 0.5 * cos(.pi * (Double(k) - position) / Double(taps + 1))
                let h = (x == 0 ? 1 : sin(.pi * x) / (.pi * x)) * window
                norm += h
                if k >= 0 { sample += Double(buffer[k]) * h }
            }
            output.append(Float(sample / norm)); position += ratio
        }
        let keep = max(0, Int(floor(position)) - taps)
        history = Array(buffer.dropFirst(keep)); position -= Double(keep)
        return output
    }
    func flush() -> [Float] {
        from == to ? [] : push([Float](repeating: 0, count: Int(ceil(16 / cutoff)) + 1))
    }
}
