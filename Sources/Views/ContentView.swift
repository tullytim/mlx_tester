import SwiftUI

struct ContentView: View {
    @EnvironmentObject var launcher: ProcessLauncher

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HeaderBar()
            Divider()

            HSplitView {
                // Left: Model picker
                ModelPanel()
                    .frame(minWidth: 220, maxWidth: 280)

                // Right: Chat
                ChatPanel()
                    .frame(minWidth: 400)
            }
        }
        .frame(minWidth: 700, minHeight: 500)
    }
}

// MARK: - Header

struct HeaderBar: View {
    @EnvironmentObject var launcher: ProcessLauncher

    var body: some View {
        HStack {
            Image(systemName: "brain.head.profile")
                .font(.title2)
                .foregroundStyle(.purple)
            Text("MLX Tester")
                .font(.title2.bold())
            Spacer()
            statusBadge
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    var statusBadge: some View {
        switch launcher.modelState {
        case .idle:
            Label("No model", systemImage: "circle")
                .font(.callout)
                .foregroundStyle(.secondary)
        case .loading:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Loading…")
                    .font(.callout)
            }
            .foregroundStyle(.orange)
        case .ready(let name):
            let short = name.split(separator: "/").last.map(String.init) ?? name
            Label(short, systemImage: "checkmark.circle.fill")
                .font(.callout)
                .foregroundStyle(.green)
        case .error:
            Label("Error", systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
        }
    }
}

// MARK: - Model Panel

struct ModelPanel: View {
    @EnvironmentObject var launcher: ProcessLauncher
    @State private var customRepo = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Models")
                .font(.headline)

            ForEach(ModelPreset.presets) { preset in
                let isActive = launcher.modelState.modelName == preset.repo
                let isLoading = isActive && !launcher.modelState.isReady
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(preset.name)
                            .font(.callout.bold())
                        Text(preset.sizeLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if isActive {
                        if isLoading {
                            ProgressView().controlSize(.small)
                        } else if launcher.modelState.isReady {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        }
                    }
                }
                .padding(.vertical, 6)
                .padding(.horizontal, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isActive ? Color.purple.opacity(0.12) : Color.clear)
                )
                .contentShape(Rectangle())
                .onTapGesture {
                    guard !isLoading else { return }
                    launcher.loadModel(preset.repo)
                }
            }

            Divider()

            Text("Custom Model")
                .font(.subheadline.bold())

            HStack {
                TextField("mlx-community/...", text: $customRepo)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption.monospaced())

                Button("Load") {
                    guard !customRepo.isEmpty else { return }
                    launcher.loadModel(customRepo)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .tint(.purple)
                .disabled(customRepo.isEmpty)
            }

            Spacer()

            if launcher.modelState.isReady || launcher.modelState != .idle {
                Button(role: .destructive) {
                    launcher.stopProcess()
                } label: {
                    Label("Unload Model", systemImage: "xmark.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding()
    }
}

// MARK: - Chat Panel

struct ChatPanel: View {
    @EnvironmentObject var launcher: ProcessLauncher
    @State private var prompt = ""
    @FocusState private var promptFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(launcher.messages) { msg in
                            MessageBubble(message: msg)
                                .id(msg.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: launcher.messages.count) { _, _ in
                    if let last = launcher.messages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                // Also scroll when the assistant message text updates (streaming)
                .onChange(of: launcher.messages.last?.text) { _, _ in
                    if let last = launcher.messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }

            Divider()

            // Input bar
            HStack(alignment: .bottom, spacing: 10) {
                TextField("Send a message…", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                    .focused($promptFocused)
                    .onSubmit {
                        send()
                    }

                Button {
                    send()
                } label: {
                    Image(systemName: launcher.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(canSend ? .purple : .gray)
                }
                .buttonStyle(.plain)
                .disabled(!canSend && !launcher.isGenerating)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(.horizontal)
            .padding(.vertical, 12)

            // Clear chat
            if !launcher.messages.isEmpty {
                HStack {
                    Spacer()
                    Button("Clear Chat") {
                        launcher.clearChat()
                    }
                    .font(.caption)
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .padding(.trailing)
                    .padding(.bottom, 4)
                }
            }
        }
        .onAppear { promptFocused = true }
    }

    private var canSend: Bool {
        launcher.modelState.isReady && !launcher.isGenerating && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func send() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, launcher.modelState.isReady, !launcher.isGenerating else { return }
        launcher.sendPrompt(text)
        prompt = ""
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                if message.role == .system {
                    HStack(spacing: 4) {
                        Image(systemName: "info.circle")
                            .font(.caption)
                        Text(message.text)
                            .font(.caption)
                    }
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 4)
                } else {
                    Text(message.text.isEmpty ? " " : message.text)
                        .font(.body)
                        .textSelection(.enabled)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(message.role == .user
                                      ? Color.purple.opacity(0.2)
                                      : Color(nsColor: .controlBackgroundColor))
                        )
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }
}
