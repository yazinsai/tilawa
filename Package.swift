// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Tilawa",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [.library(name: "Tilawa", targets: ["Tilawa"])],
    targets: [
        .target(name: "Tilawa", path: "packages/swift/Sources/Tilawa"),
        .testTarget(name: "TilawaTests", dependencies: ["Tilawa"],
                    path: "packages/swift/Tests/TilawaTests", resources: [.copy("Fixtures")]),
    ]
)
