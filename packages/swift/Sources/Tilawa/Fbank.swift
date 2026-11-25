// SPDX-License-Identifier: MIT
import Foundation

/// Direct Swift port of Tilawa's Kaldi-compatible streaming 80-bin log-mel features (MIT).
/// Double precision FFT arithmetic and Float32 frame boundaries match the original implementation.
final class Fbank {
    static let window = 400, hop = 160, fftSize = 512, mels = 80
    private var pending: [Float] = []
    private let povey: [Double]
    private let bank: [(start: Int, weights: [Double])]
    private let cosine: [Double]
    private let sine: [Double]
    private let reversed: [Int]

    init() {
        let n = Self.fftSize
        povey = (0..<Self.window).map { pow(0.5 - 0.5 * cos(2 * .pi * Double($0) / Double(Self.window - 1)), 0.85) }
        func mel(_ hz: Double) -> Double { 1127 * log(1 + hz / 700) }
        let low = mel(20), step = (mel(7600) - low) / Double(Self.mels + 1)
        bank = (0..<Self.mels).map { m in
            let left = low + Double(m) * step, center = left + step, right = center + step
            var bins: [Int] = [], weights: [Double] = []
            for k in 0..<n / 2 {
                let f = mel(Double(k) * 16000 / Double(n))
                let v = min((f - left) / (center - left), (right - f) / (right - center))
                if v > 0 { bins.append(k); weights.append(v) }
            }
            return (bins.first ?? 0, weights)
        }
        cosine = (0..<n / 2).map { cos(-2 * .pi * Double($0) / Double(n)) }
        sine = (0..<n / 2).map { sin(-2 * .pi * Double($0) / Double(n)) }
        reversed = (0..<n).map { i in
            var r = 0, x = i, b = 1
            while b < n { r = (r << 1) | (x & 1); x >>= 1; b <<= 1 }
            return r
        }
    }

    func push(_ samples: [Float]) -> [[Float]] {
        let buffer = pending + samples
        var frames: [[Float]] = [], offset = 0
        while offset + Self.window <= buffer.count {
            var real = [Double](repeating: 0, count: Self.fftSize)
            var imaginary = real
            let mean = buffer[offset..<offset + Self.window].reduce(0.0) { $0 + Double($1) } / Double(Self.window)
            for i in 0..<Self.window {
                let x = Double(buffer[offset + i]) - mean
                let previous = i > 0 ? Double(buffer[offset + i - 1]) - mean : x
                real[i] = (x - 0.97 * previous) * povey[i]
            }
            fft(&real, &imaginary)
            frames.append(bank.map { filter in
                var energy = 0.0
                for (j, weight) in filter.weights.enumerated() {
                    let k = filter.start + j
                    energy += weight * (real[k] * real[k] + imaginary[k] * imaginary[k])
                }
                return Float(log(max(energy, 1.1920928955078125e-7)))
            })
            offset += Self.hop
        }
        pending = Array(buffer.dropFirst(offset))
        return frames
    }

    private func fft(_ re: inout [Double], _ im: inout [Double]) {
        for i in 0..<Self.fftSize where i < reversed[i] {
            re.swapAt(i, reversed[i]); im.swapAt(i, reversed[i])
        }
        var length = 2
        while length <= Self.fftSize {
            let half = length / 2, stride = Self.fftSize / length
            for i in Swift.stride(from: 0, to: Self.fftSize, by: length) {
                for k in 0..<half {
                    let a = i + k, b = a + half
                    let wr = cosine[k * stride], wi = sine[k * stride]
                    let xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr
                    re[b] = re[a] - xr; im[b] = im[a] - xi
                    re[a] += xr; im[a] += xi
                }
            }
            length <<= 1
        }
    }
}
