import ArgumentParser
import Foundation
import Hummingbird
import MLX
import MLXLLM
import MLXLMCommon

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Start OpenAI-compatible API server"
    )

    @Argument(help: "Model ID (e.g. mlx-community/Qwen3-4B-4bit)")
    var model: String

    @Option(name: .shortAndLong, help: "Host to bind")
    var host: String = "0.0.0.0"

    @Option(name: .shortAndLong, help: "Port to listen on")
    var port: Int = 8080

    @Option(name: .shortAndLong, help: "System prompt")
    var system: String?

    @Option(name: .long, help: "Default max tokens")
    var defaultMaxTokens: Int = 2048

    @Option(name: .long, help: "Default temperature")
    var defaultTemperature: Float = 0.6

    @Option(name: .long, help: "Seed for reproducibility")
    var seed: Int?

    func run() async throws {
        if let seed { MLXRandom.seed(UInt64(seed)) }

        let modelId = Helpers.resolveModelId(model)
        print("Loading \(modelId)...")
        let container = try await Helpers.loadContainer(modelId: modelId)
        print("Model loaded. Starting server on \(host):\(port)")

        let state = ServerState(container: container, modelId: modelId)

        let router = Router()

        router.get("v1/models") { _, _ in
            return ModelsResponse(data: [ModelObject(id: state.modelId)])
        }

        router.post("v1/chat/completions") { request, context -> Response in
            let startTime = ContinuousClock.now
            let req = try await request.decode(as: ChatRequest.self, context: context)

            let maxTokens = req.maxTokens ?? self.defaultMaxTokens
            let temperature = req.temperature ?? self.defaultTemperature
            let topP = req.topP ?? 1.0
            let stream = req.stream ?? false

            let lastUserMsg = req.messages.last(where: { $0.role == "user" })?.textContent ?? ""
            let preview = String(lastUserMsg.prefix(80))
            let toolCount = req.tools?.count ?? 0
            print("[request] messages=\(req.messages.count) tools=\(toolCount) stream=\(stream) max_tokens=\(maxTokens) temp=\(String(format: "%.2f", temperature)) prompt=\"\(preview)\"")

            var messages = req.messages
            if let system = self.system, !messages.contains(where: { $0.role == "system" }) {
                messages.insert(ChatMessage(role: "system", content: .text(system)), at: 0)
            }

            let toolSpecs: [ToolSpec]? = req.tools?.map { tool -> ToolSpec in
                guard let dict = tool as? [String: any Sendable] else { return [:] }
                return dict
            }

            let input = UserInput(
                prompt: .messages(messages.map { ["role": $0.role, "content": $0.textContent] }),
                tools: toolSpecs
            )
            let lmInput = try await state.container.prepare(input: input)
            nonisolated(unsafe) let captureInput = lmInput
            let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: topP)

            let id = "chatcmpl-\(UUID().uuidString.lowercased())"
            let created = Int(Date().timeIntervalSince1970)

            if stream {
                let responseBody = ResponseBody { writer in
                    var tokenCount = 0
                    var inThink = false
                    let genStream = try await state.container.generate(input: captureInput, parameters: params)
                    for await event in genStream {
                        switch event {
                        case .chunk(let text):
                            let filtered = Self.stripThinkTags(text, inThink: &inThink)
                            if !filtered.isEmpty {
                                tokenCount += 1
                                let chunk = StreamChunk(
                                    id: id, created: created, model: modelId,
                                    delta: StreamDelta(content: filtered)
                                )
                                let data = try JSONEncoder().encode(chunk)
                                let sse = "data: \(String(data: data, encoding: .utf8)!)\n\n"
                                var buffer = ByteBuffer(string: sse)
                                try await writer.write(buffer)
                            }
                        case .toolCall(let call):
                            tokenCount += 1
                            let toolChunk = StreamChunk(
                                id: id, created: created, model: modelId,
                                toolCalls: [ToolCallChunk(
                                    index: 0,
                                    function: ToolCallFunction(
                                        name: call.function.name,
                                        arguments: call.function.arguments
                                    )
                                )]
                            )
                            let data = try JSONEncoder().encode(toolChunk)
                            let sse = "data: \(String(data: data, encoding: .utf8)!)\n\n"
                            var buffer = ByteBuffer(string: sse)
                            try await writer.write(buffer)
                        case .info(let info):
                            let elapsed = ContinuousClock.now - startTime
                            let elapsedSec = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000_000
                            print("[response] stream=true tokens=\(tokenCount) tok/s=\(String(format: "%.1f", info.tokensPerSecond)) elapsed=\(String(format: "%.2f", elapsedSec))s stop=\(info.stopReason)")
                        }
                    }
                    var doneBuf = ByteBuffer(string: "data: [DONE]\n\n")
                    try await writer.write(doneBuf)
                    try await writer.finish(nil)
                }
                return Response(
                    status: .ok,
                    headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
                    body: responseBody
                )
            } else {
                var fullResponse = ""
                var reasoning = ""
                var tokenCount = 0
                var tokensPerSecond: Double = 0
                var stopReason = "stop"
                var toolCalls: [ResponseToolCall] = []
                var inThink = false

                let genStream = try await state.container.generate(input: lmInput, parameters: params)
                for await event in genStream {
                    switch event {
                    case .chunk(let text):
                        let filtered = Self.extractThinkContent(text, inThink: &inThink, reasoning: &reasoning)
                        fullResponse += filtered
                        tokenCount += 1
                    case .toolCall(let call):
                        let argsData = try JSONSerialization.data(withJSONObject: call.function.arguments.mapValues { $0.anyValue }, options: [.sortedKeys])
                        let argsStr = String(data: argsData, encoding: .utf8) ?? "{}"
                        toolCalls.append(ResponseToolCall(
                            id: "call_\(UUID().uuidString.lowercased().prefix(24))",
                            type: "function",
                            function: ResponseToolCallFunction(name: call.function.name, arguments: argsStr)
                        ))
                    case .info(let info):
                        tokensPerSecond = info.tokensPerSecond
                        stopReason = String(describing: info.stopReason)
                    }
                }
                let elapsed = ContinuousClock.now - startTime
                let elapsedSec = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000_000
                print("[response] stream=false tokens=\(tokenCount) tok/s=\(String(format: "%.1f", tokensPerSecond)) elapsed=\(String(format: "%.2f", elapsedSec))s stop=\(stopReason) tool_calls=\(toolCalls.count) chars=\(fullResponse.count)")

                let finishReason = toolCalls.isEmpty ? stopReason : "tool_calls"
                let result = ChatCompletion(
                    id: id, created: created, model: modelId,
                    content: fullResponse,
                    reasoningContent: reasoning.isEmpty ? nil : reasoning,
                    toolCalls: toolCalls.isEmpty ? nil : toolCalls,
                    finishReason: finishReason
                )
                let encoder = JSONEncoder()
                encoder.keyEncodingStrategy = .convertToSnakeCase
                let data = try encoder.encode(result)
                let buffer = ByteBuffer(data: data)
                return Response(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: buffer)
                )
            }
        }

        router.get("health") { _, _ -> String in
            return "ok"
        }

        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )
        try await app.runService()
    }

    static func stripThinkTags(_ text: String, inThink: inout Bool) -> String {
        var result = ""
        var remaining = text[...]

        while !remaining.isEmpty {
            if inThink {
                if let range = remaining.range(of: "</think") {
                    inThink = false
                    remaining = remaining[range.upperBound...]
                    while remaining.hasPrefix(">") || remaining.first?.isWhitespace == true {
                        remaining = remaining[remaining.index(after: remaining.startIndex)...]
                    }
                } else {
                    remaining = ""
                }
            } else {
                if let range = remaining.range(of: "<think") {
                    result += String(remaining[..<range.lowerBound])
                    inThink = true
                    remaining = remaining[range.upperBound...]
                    while remaining.hasPrefix(">") || remaining.first?.isWhitespace == true {
                        remaining = remaining[remaining.index(after: remaining.startIndex)...]
                    }
                } else {
                    result += String(remaining)
                    remaining = ""
                }
            }
        }
        return result
    }

    static func extractThinkContent(_ text: String, inThink: inout Bool, reasoning: inout String) -> String {
        var result = ""
        var remaining = text[...]

        while !remaining.isEmpty {
            if inThink {
                if let range = remaining.range(of: "</think") {
                    reasoning += String(remaining[..<range.lowerBound])
                    inThink = false
                    remaining = remaining[range.upperBound...]
                    while remaining.hasPrefix(">") || remaining.first?.isWhitespace == true {
                        remaining = remaining[remaining.index(after: remaining.startIndex)...]
                    }
                } else {
                    reasoning += String(remaining)
                    remaining = ""
                }
            } else {
                if let range = remaining.range(of: "<think") {
                    result += String(remaining[..<range.lowerBound])
                    inThink = true
                    remaining = remaining[range.upperBound...]
                    while remaining.hasPrefix(">") || remaining.first?.isWhitespace == true {
                        remaining = remaining[remaining.index(after: remaining.startIndex)...]
                    }
                } else {
                    result += String(remaining)
                    remaining = ""
                }
            }
        }
        return result
    }
}

struct ChatRequest: Codable {
    let messages: [ChatMessage]
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    let stream: Bool?
    let tools: [AnyJSON]?

    enum CodingKeys: String, CodingKey {
        case messages, tools
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case stream
    }
}

struct ChatMessage: Codable {
    let role: String
    let content: MessageContent

    var textContent: String {
        switch content {
        case .text(let str): return str
        case .parts(let parts):
            return parts.compactMap { part -> String? in
                if case .text(let t) = part { return t }
                return nil
            }.joined()
        }
    }
}

enum MessageContent: Codable {
    case text(String)
    case parts([ContentPart])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else if let parts = try? container.decode([ContentPart].self) {
            self = .parts(parts)
        } else {
            self = .text("")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let str): try container.encode(str)
        case .parts(let parts): try container.encode(parts)
        }
    }
}

enum ContentPart: Codable {
    case text(String)

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text":
            let text = try container.decodeIfPresent(String.self, forKey: .text) ?? ""
            self = .text(text)
        default:
            self = .text("")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        }
    }

    enum CodingKeys: String, CodingKey {
        case type, text
    }
}

struct AnyJSON: Codable, @unchecked Sendable {
    let value: Any

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let dict = try? container.decode([String: AnyJSON].self) {
            self.value = dict.mapValues { $0.value }
        } else if let arr = try? container.decode([AnyJSON].self) {
            self.value = arr.map { $0.value }
        } else if let str = try? container.decode(String.self) {
            self.value = str
        } else if let int = try? container.decode(Int.self) {
            self.value = int
        } else if let double = try? container.decode(Double.self) {
            self.value = double
        } else if let bool = try? container.decode(Bool.self) {
            self.value = bool
        } else {
            self.value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let str as String: try container.encode(str)
        case let int as Int: try container.encode(int)
        case let double as Double: try container.encode(double)
        case let bool as Bool: try container.encode(bool)
        default: try container.encodeNil()
        }
    }

    init(wrapping: Any) {
        self.value = wrapping
    }
}

struct ChatCompletion: Codable {
    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]

    init(id: String, created: Int, model: String, content: String, reasoningContent: String? = nil, toolCalls: [ResponseToolCall]? = nil, finishReason: String = "stop") {
        self.id = id
        self.created = created
        self.model = model
        self.choices = [Choice(
            message: ChoiceMessage(content: content, reasoningContent: reasoningContent, toolCalls: toolCalls),
            finishReason: finishReason
        )]
    }

    struct Choice: Codable {
        let index: Int = 0
        let message: ChoiceMessage
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct ChoiceMessage: Codable {
        let role: String = "assistant"
        let content: String
        let reasoningContent: String?
        let toolCalls: [ResponseToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
        }
    }
}

struct ResponseToolCall: Codable {
    let id: String
    let type: String
    let function: ResponseToolCallFunction
}

struct ResponseToolCallFunction: Codable {
    let name: String
    let arguments: String
}

struct StreamChunk: Codable {
    let id: String
    let object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [StreamChoice]

    init(id: String, created: Int, model: String, delta: StreamDelta? = nil, toolCalls: [ToolCallChunk]? = nil) {
        self.id = id
        self.created = created
        self.model = model
        self.choices = [StreamChoice(delta: delta, toolCalls: toolCalls)]
    }
}

struct StreamChoice: Codable {
    let index: Int = 0
    let delta: StreamDelta?
    let toolCalls: [ToolCallChunk]?
    let finishReason: String? = nil

    enum CodingKeys: String, CodingKey {
        case index, delta
        case toolCalls = "tool_calls"
        case finishReason = "finish_reason"
    }
}

struct StreamDelta: Codable {
    let role: String? = "assistant"
    let content: String?
}

struct ToolCallChunk: Codable {
    let index: Int
    let function: ToolCallFunction
}

struct ToolCallFunction: Codable {
    let name: String
    let arguments: [String: JSONValue]

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        let argsData = try JSONSerialization.data(withJSONObject: arguments.mapValues { $0.anyValue }, options: [.sortedKeys])
        let argsStr = String(data: argsData, encoding: .utf8) ?? "{}"
        try container.encode(argsStr, forKey: .arguments)
    }

    enum CodingKeys: String, CodingKey {
        case name, arguments
    }
}

struct ModelsResponse: Codable, ResponseEncodable {
    let object: String = "list"
    let data: [ModelObject]
}

struct ModelObject: Codable {
    let id: String
    let object: String = "model"
    let ownedBy: String = "mlx-community"

    enum CodingKeys: String, CodingKey {
        case id, object
        case ownedBy = "owned_by"
    }
}

final class ServerState: Sendable {
    let container: ModelContainer
    let modelId: String

    init(container: ModelContainer, modelId: String) {
        self.container = container
        self.modelId = modelId
    }
}
