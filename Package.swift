// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXTester",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "MLXTester",
            path: "Sources"
        ),
    ]
)
