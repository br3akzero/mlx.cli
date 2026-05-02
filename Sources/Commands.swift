import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run a single prompt against a model"
    )

    @Argument(help: "Model ID (e.g. mlx-community/Qwen3-4B-4bit)")
    var model: String

    @Argument(help: "Prompt to send")
    var prompt: String

    @Option(name: .shortAndLong, help: "Max tokens to generate")
    var maxTokens: Int = 512

    @Option(name: .shortAndLong, help: "Temperature (0.0 - 2.0)")
    var temperature: Float = 0.6

    @Option(name: .long, help: "Top-p sampling")
    var topP: Float = 1.0

    @Option(name: .long, help: "Seed for reproducibility")
    var seed: Int?

    func run() async throws {
        if let seed { MLXRandom.seed(UInt64(seed)) }

        let modelId = Helpers.resolveModelId(model)
        print("Loading \(modelId)...")
        let container = try await Helpers.loadContainer(modelId: modelId)

        let input = UserInput(prompt: prompt)
        let lmInput = try await container.prepare(input: input)
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )

        print()
        let stream = try await container.generate(input: lmInput, parameters: params)
        for await event in stream {
            switch event {
            case .chunk(let text):
                print(text, terminator: "")
            case .toolCall(let call):
                print("\n[Tool: \(call.function.name)]")
            case .info(let info):
                print("\n\n--- \(info.stopReason) | \(String(format: "%.1f", info.tokensPerSecond)) tok/s ---")
            }
        }
        print()
    }
}

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Start an interactive chat session"
    )

    @Argument(help: "Model ID (e.g. mlx-community/Qwen3-4B-4bit)")
    var model: String

    @Option(name: .shortAndLong, help: "Max tokens per response")
    var maxTokens: Int = 1024

    @Option(name: .shortAndLong, help: "Temperature (0.0 - 2.0)")
    var temperature: Float = 0.6

    @Option(name: .long, help: "System prompt")
    var system: String?

    @Option(name: .long, help: "Seed for reproducibility")
    var seed: Int?

    func run() async throws {
        if let seed { MLXRandom.seed(UInt64(seed)) }

        let modelId = Helpers.resolveModelId(model)
        print("Loading \(modelId)...")
        let container = try await Helpers.loadContainer(modelId: modelId)

        var messages: [[String: String]] = []
        if let system {
            messages.append(["role": "system", "content": system])
        }

        print("Chat ready. Type 'quit' to exit, 'clear' to reset history.\n")

        while true {
            print("You> ", terminator: "")
            guard let line = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !line.isEmpty else { continue }

            if line == "quit" { break }
            if line == "clear" {
                messages.removeAll(keepingCapacity: true)
                if let system { messages.append(["role": "system", "content": system]) }
                print("History cleared.\n")
                continue
            }

            messages.append(["role": "user", "content": line])

            let input = UserInput(prompt: .messages(messages))
            let lmInput = try await container.prepare(input: input)
            let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature)

            print("Assistant> ", terminator: "")
            var response = ""
            let stream = try await container.generate(input: lmInput, parameters: params)
            for await event in stream {
                switch event {
                case .chunk(let text):
                    print(text, terminator: "")
                    response += text
                case .info(let info):
                    print(" [\(String(format: "%.1f", info.tokensPerSecond)) tok/s]")
                default: break
                }
            }
            messages.append(["role": "assistant", "content": response])
            print()
        }
    }
}

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download a model from HuggingFace"
    )

    @Argument(help: "Model ID (e.g. mlx-community/Qwen3-4B-4bit)")
    var model: String

    func run() async throws {
        let modelId = Helpers.resolveModelId(model)
        print("Downloading \(modelId)...")
        _ = try await Helpers.loadContainer(modelId: modelId)
        print("Done. Model cached locally.")
    }
}

struct List: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List locally cached models"
    )

    func run() async throws {
        let cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let fileManager = FileManager.default

        guard let contents = try? fileManager.contentsOfDirectory(
            at: cacheDir, includingPropertiesForKeys: nil
        ) else {
            print("No models cached.")
            return
        }

        var found = false
        for url in contents {
            let name = url.lastPathComponent
            if name.hasPrefix("models--") {
                let modelId = name
                    .replacingOccurrences(of: "models--", with: "")
                    .replacingOccurrences(of: "--", with: "/")
                print(modelId)
                found = true
            }
        }

        if !found {
            print("No models cached.")
        }
    }
}
