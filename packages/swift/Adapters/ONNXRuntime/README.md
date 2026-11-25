# Optional iOS ONNX Runtime adapter

This source-only adapter connects the Swift engine to the same native ONNX model used by Tilawa. It is not linked into the dependency-free Swift package automatically.

1. Add the repository's `Tilawa` Swift package product to your iOS target.
2. Add `TilawaONNXModel.h`, `TilawaONNXModel.mm`, and `ONNXRuntimeBackend.swift` to that target. Add `#import "TilawaONNXModel.h"` to its bridging header. Enable Objective-C ARC and C++17.
3. Obtain Microsoft's `onnxruntime.xcframework` from an official distribution and link it, together with CoreML, Accelerate, and libc++. The development integration used ONNX Runtime **1.24.3** from the iOS XCFramework embedded in the official `Microsoft.ML.OnnxRuntime` NuGet package. Include its MIT license when redistributing the runtime.
4. Supply a matching model file and model-manifest JSON. Verify asset hashes against the model registry before loading them.

```swift
let backend = try ONNXRuntimeBackend(modelURL: modelURL, manifest: manifestData)
let engine = try TilawaEngine(reference: referenceData, manifest: manifestData, backend: backend)
```

The Objective-C++ bridge owns recurrent model tensors and advances them on each run; it uses the CPU provider and one intra-op thread. The Swift wrapper locks all access, including reset. Create a separate backend instance for each engine. `InferenceBackend.run(features:)` expects a flattened `[1, window, 80]` tensor and returns flattened `log_probs`; the adapter retains the remaining output tensors as the next recurrent state.

No network requests or audio capture are performed by the adapter. The host app owns permissions, microphone conversion, session lifetime, and cancellation. This adapter is experimental; the simulator integration does not establish performance or accuracy on physical devices.
