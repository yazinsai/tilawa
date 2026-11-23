// SPDX-License-Identifier: MIT
import Foundation
import Tilawa

/// Add this file and TilawaONNXModel.{h,mm} to your iOS target; expose the header
/// through its bridging header and link Microsoft's onnxruntime.xcframework.
/// Each engine must have its own backend instance.
public final class ONNXRuntimeBackend: InferenceBackend, @unchecked Sendable {
    // Objective-C++ runtime state is protected for every access, including reset.
    private let lock = NSLock()
    private let model: TilawaONNXModel

    public init(modelURL: URL, manifest: Data) throws {
        model = try TilawaONNXModel(path: modelURL.path, manifest: manifest)
    }
    public func reset() throws {
        lock.lock(); defer { lock.unlock() }
        model.reset()
    }
    public func run(features: [Float]) throws -> [Float] {
        lock.lock(); defer { lock.unlock() }
        let data = features.withUnsafeBytes { Data($0) }
        let output = try model.runFeatures(data)
        return output.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
