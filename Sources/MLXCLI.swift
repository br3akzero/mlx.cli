import ArgumentParser

@main
struct MLXCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx",
        abstract: "MLX Swift CLI - Run LLMs on Apple Silicon",
        subcommands: [Run.self, Download.self, List.self, Chat.self],
        defaultSubcommand: Run.self
    )
}
