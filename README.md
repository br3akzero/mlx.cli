# mlx.cli

Standalone Swift CLI for running LLMs on Apple Silicon via MLX. Downloads models from HuggingFace Hub, runs inference natively via Metal, and serves an OpenAI-compatible API.

## Requirements

- macOS 15+
- Apple Silicon Mac
- Xcode (with command line tools)

## Build

SwiftPM cannot compile Metal shaders. Use xcodebuild:

```bash
xcodebuild build -scheme mlx-cli -configuration Release \
  -destination 'platform=OS X' \
  -skipPackagePluginValidation -skipMacroValidation
```

Binary output:
```
~/Library/Developer/Xcode/DerivedData/MLX-*/Build/Products/Release/mlx-cli
```

## Commands

### `mlx run` -- Single prompt inference

```bash
mlx-cli run mlx-community/Qwen3-4B-4bit "What is 2+2?" --max-tokens 128
mlx-cli run Qwen3.5-35B-A3B-4bit "Explain quantum computing" --temperature 0.8
```

Options: `--max-tokens`, `--temperature`, `--top-p`, `--seed`

### `mlx chat` -- Interactive chat session

```bash
mlx-cli chat mlx-community/Qwen3-4B-4bit
mlx-cli chat Qwen3-8B-4bit --system "You are a helpful coding assistant."
```

Type `quit` to exit, `clear` to reset history.

### `mlx download` -- Cache a model locally

```bash
mlx-cli download mlx-community/Qwen3-4B-4bit
```

### `mlx list` -- List cached models

```bash
mlx-cli list
```

### `mlx serve` -- OpenAI-compatible API server

```bash
mlx-cli serve mlx-community/Qwen3.5-35B-A3B-4bit --port 8080
mlx-cli serve Qwen3-4B-4bit --host 127.0.0.1 --port 9090 --system "You are concise."
```

Options: `--host`, `--port`, `--system`, `--default-max-tokens`, `--default-temperature`, `--seed`

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List loaded model |
| POST | `/v1/chat/completions` | Chat completion (OpenAI format) |

#### Chat completions

Supports streaming (`"stream": true`), tool/function calling, and multimodal content arrays. Think/reasoning tags (`<think`) are stripped from output. Reasoning content available via `reasoning_content` field in non-streaming mode.

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "stream": false
  }'
```

#### Per-request logging

```
[request] messages=3 tools=2 stream=true max_tokens=2048 temp=0.60 prompt="Explain Rust ownership"
[response] stream=true tokens=142 tok/s=97.2 elapsed=1.47s stop=stop
```

## Model shorthand

If no namespace is provided, `mlx-community/` is prepended:

```bash
mlx-cli run Qwen3-4B-4bit "hi"
# equivalent to mlx-cli run mlx-community/Qwen3-4B-4bit "hi"
```

## Install globally

```bash
sudo cp ~/Library/Developer/Xcode/DerivedData/MLX-*/Build/Products/Release/mlx-cli /usr/local/bin/mlx
```

## Configuration with opencode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "mlx-community/Qwen3.5-35B-A3B-4bit": {
          "name": "Qwen3.5 35B (local MLX)"
        }
      }
    }
  }
}
```

## License

MIT
