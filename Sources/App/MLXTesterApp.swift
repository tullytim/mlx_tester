import SwiftUI

@main
struct MLXTesterApp: App {
    @StateObject private var launcher = ProcessLauncher()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(launcher)
                .frame(minWidth: 700, minHeight: 600)
        }
        .defaultSize(width: 800, height: 650)
    }
}
