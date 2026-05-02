import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

enum Helpers {
    static func resolveModelId(_ model: String) -> String {
        model.contains("/") ? model : "mlx-community/\(model)"
    }

    static func loadContainer(modelId: String) async throws -> ModelContainer {
        let config = ModelConfiguration(id: modelId)
        return try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: config,
            progressHandler: { progress in
                let pct = Int(progress.fractionCompleted * 100)
                let completed = ByteCountFormatter.string(
                    fromByteCount: Int64(progress.completedUnitCount), countStyle: .file
                )
                let total = ByteCountFormatter.string(
                    fromByteCount: Int64(progress.totalUnitCount), countStyle: .file
                )
                fputs("\rDownloading: \(pct)% (\(completed) / \(total))", stderr)
                if pct >= 100 {
                    fputs("\rDownload complete. Loading model...\n", stderr)
                }
            }
        )
    }
}
