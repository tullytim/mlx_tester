import Foundation
import Combine

/// Represents an available model preset
struct ModelPreset: Identifiable, Hashable {
    let id: String
    let name: String
    let repo: String
    let sizeLabel: String

    static let presets: [ModelPreset] = [
        // Small (< 2 GB)
        ModelPreset(id: "llama-1b", name: "Llama 3.2 1B", repo: "mlx-community/Llama-3.2-1B-Instruct-4bit", sizeLabel: "~0.7 GB"),
        ModelPreset(id: "smollm2-1.7b", name: "SmolLM2 1.7B", repo: "mlx-community/SmolLM2-1.7B-Instruct-4bit", sizeLabel: "~1.0 GB"),
        ModelPreset(id: "qwen-1.5b", name: "Qwen 2.5 1.5B", repo: "mlx-community/Qwen2.5-1.5B-Instruct-4bit", sizeLabel: "~1.0 GB"),
        ModelPreset(id: "gemma-2b", name: "Gemma 2 2B", repo: "mlx-community/gemma-2-2b-it-4bit", sizeLabel: "~1.4 GB"),
        ModelPreset(id: "llama-3b", name: "Llama 3.2 3B", repo: "mlx-community/Llama-3.2-3B-Instruct-4bit", sizeLabel: "~1.8 GB"),
        ModelPreset(id: "qwen-3b", name: "Qwen 2.5 3B", repo: "mlx-community/Qwen2.5-3B-Instruct-4bit", sizeLabel: "~1.8 GB"),

        // Medium (2–5 GB)
        ModelPreset(id: "phi-3.5", name: "Phi 3.5 Mini", repo: "mlx-community/Phi-3.5-mini-instruct-4bit", sizeLabel: "~2.2 GB"),
        ModelPreset(id: "phi-4-mini", name: "Phi 4 Mini", repo: "mlx-community/phi-4-mini-instruct-4bit", sizeLabel: "~2.4 GB"),
        ModelPreset(id: "gemma3-4b", name: "Gemma 3 4B", repo: "mlx-community/gemma-3-4b-it-4bit", sizeLabel: "~2.5 GB"),
        ModelPreset(id: "mistral-7b", name: "Mistral 7B", repo: "mlx-community/Mistral-7B-Instruct-v0.3-4bit", sizeLabel: "~4.1 GB"),
        ModelPreset(id: "qwen-7b", name: "Qwen 2.5 7B", repo: "mlx-community/Qwen2.5-7B-Instruct-4bit", sizeLabel: "~4.4 GB"),
        ModelPreset(id: "deepseek-r1-7b", name: "DeepSeek-R1 Qwen 7B", repo: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", sizeLabel: "~4.4 GB"),
        ModelPreset(id: "llama-8b", name: "Llama 3.1 8B", repo: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", sizeLabel: "~4.5 GB"),

        // Large (8+ GB — needs more RAM)
        ModelPreset(id: "gemma3-12b", name: "Gemma 3 12B", repo: "mlx-community/gemma-3-12b-it-4bit", sizeLabel: "~7.1 GB"),
        ModelPreset(id: "mistral-small-24b", name: "Mistral Small 24B", repo: "mlx-community/Mistral-Small-24B-Instruct-2501-4bit", sizeLabel: "~13 GB"),
        ModelPreset(id: "llama-70b", name: "Llama 3.3 70B", repo: "mlx-community/Llama-3.3-70B-Instruct-4bit", sizeLabel: "~38 GB"),
    ]
}

/// A single chat message
struct ChatMessage: Identifiable {
    let id = UUID()
    let role: MessageRole
    var text: String
    let timestamp: Date

    enum MessageRole {
        case user
        case assistant
        case system
    }
}

/// Model loading state
enum ModelState: Equatable {
    case idle
    case loading(String)
    case ready(String)
    case error(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var modelName: String? {
        switch self {
        case .ready(let name), .loading(let name): return name
        default: return nil
        }
    }
}

/// Manages the interactive MLX process
@MainActor
final class ProcessLauncher: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var modelState: ModelState = .idle
    @Published var isGenerating = false

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var currentResponseIndex: Int?
    private let pythonPath = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"

    private var scriptsDir: URL {
        let candidates = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Scripts"),
            URL(fileURLWithPath: "/Users/tim/mlx_tester/Scripts"),
        ]
        return candidates.first {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("mlx_interactive.py").path)
        } ?? candidates.last!
    }

    func loadModel(_ repo: String) {
        // Kill existing process if any
        stopProcess()

        modelState = .loading(repo)
        messages.append(ChatMessage(role: .system, text: "Loading \(repo)...", timestamp: Date()))

        let proc = Process()
        let stdin = Pipe()
        let stdout = Pipe()

        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.arguments = [
            scriptsDir.appendingPathComponent("mlx_interactive.py").path,
            "--model", repo,
        ]
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = stdout

        self.process = proc
        self.stdinPipe = stdin
        self.stdoutPipe = stdout

        // Read output
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in
                self?.handleOutput(str)
            }
        }

        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if case .loading = self.modelState {
                    self.modelState = .error("Process exited with code \(p.terminationStatus)")
                } else if self.modelState.isReady {
                    self.modelState = .idle
                }
                self.isGenerating = false
                self.process = nil
            }
        }

        do {
            try proc.run()
        } catch {
            modelState = .error("Failed to launch: \(error.localizedDescription)")
        }
    }

    func sendPrompt(_ prompt: String) {
        guard modelState.isReady, !isGenerating else { return }
        guard let stdinPipe else { return }

        messages.append(ChatMessage(role: .user, text: prompt, timestamp: Date()))
        messages.append(ChatMessage(role: .assistant, text: "", timestamp: Date()))
        currentResponseIndex = messages.count - 1
        isGenerating = true

        let data = (prompt + "\n").data(using: .utf8)!
        stdinPipe.fileHandleForWriting.write(data)
    }

    func stopProcess() {
        if let process, process.isRunning {
            process.terminate()
        }
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        isGenerating = false
        currentResponseIndex = nil
    }

    func clearChat() {
        messages.removeAll()
    }

    // MARK: - Output parsing

    /// Buffer for partial lines
    private var lineBuffer = ""

    private func handleOutput(_ raw: String) {
        lineBuffer += raw
        // Process complete lines
        while let newlineRange = lineBuffer.range(of: "\n") {
            let line = String(lineBuffer[lineBuffer.startIndex..<newlineRange.lowerBound])
            lineBuffer = String(lineBuffer[newlineRange.upperBound...])
            processLine(line)
        }
        // If we're generating and there's partial content without newline, it's streaming tokens
        if isGenerating, !lineBuffer.isEmpty, let idx = currentResponseIndex {
            // Only append if it doesn't look like a control line
            if !lineBuffer.hasPrefix("__") {
                messages[idx].text += lineBuffer
                lineBuffer = ""
            }
        }
    }

    private func processLine(_ line: String) {
        if line.hasPrefix("__READY__") {
            if case .loading(let model) = modelState {
                modelState = .ready(model)
                // Update the loading system message
                if let idx = messages.lastIndex(where: { $0.role == .system && $0.text.contains("Loading") }) {
                    messages[idx].text = "Model loaded and ready."
                }
            }
        } else if line.hasPrefix("__LOADED__") {
            // Model finished loading
        } else if line.hasPrefix("__STATUS__") {
            let status = String(line.dropFirst("__STATUS__".count)).trimmingCharacters(in: .whitespaces)
            if let idx = messages.lastIndex(where: { $0.role == .system }) {
                messages[idx].text = status
            }
        } else if line.hasPrefix("__DONE__") {
            // Parse stats
            let parts = line.split(separator: " ")
            var tps = ""
            var tokens = ""
            for part in parts {
                if part.hasPrefix("tps=") { tps = String(part.dropFirst(4)) }
                if part.hasPrefix("tokens=") { tokens = String(part.dropFirst(7)) }
            }
            if let idx = currentResponseIndex {
                messages[idx].text += "\n\n*\(tokens) tokens, \(tps) tok/s*"
            }
            isGenerating = false
            currentResponseIndex = nil
        } else if line.hasPrefix("__ERROR__") {
            let err = String(line.dropFirst("__ERROR__".count)).trimmingCharacters(in: .whitespaces)
            if isGenerating, let idx = currentResponseIndex {
                messages[idx].text += "\n\n**Error:** \(err)"
            } else {
                modelState = .error(err)
                messages.append(ChatMessage(role: .system, text: "Error: \(err)", timestamp: Date()))
            }
            isGenerating = false
            currentResponseIndex = nil
        } else if isGenerating, let idx = currentResponseIndex {
            // Regular output line — append to response
            if !messages[idx].text.isEmpty {
                messages[idx].text += "\n"
            }
            messages[idx].text += line
        }
    }
}
