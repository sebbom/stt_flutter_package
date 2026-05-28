# stt_flutter — agent instructions

## Quick start

```bash
flutter pub get
flutter test                                # unit tests
```

## Architecture

- **STT engine**: `SttFlutter` (main isolate) manages `OnnxRuntime` + `OrtSession`. All `session.run()` calls are async via MethodChannel.
- **Threading**: Audio preprocessing (WAV parsing, resampling, mel/Fbank extraction) runs in ephemeral `Isolate.run()` background isolates. ONNX inference runs on the main isolate (async, non-blocking).
- **Model registry**: `ModelRegistry` singleton. Default models registered on first access via `_ensureDefaults()`. Add custom models with `ModelRegistry.register(...)`.
- **Model downloader**: `ModelDownloader.download()` — HTTP streaming from HuggingFace / GitHub, with progress callbacks. Extracts `.tar.bz2` for Sherpa models.
- **Three engines** (each in `lib/src/engines/`):
  - `whisper/` — mel spectrogram → encoder → autoregressive decoder → BPE decode
  - `sherpa/` — Fbank features → encoder → transducer greedy search → tokens.txt
  - `voxtral/` — mel spectrogram → audio_encoder → embed_tokens → LLM decoder → Tekken decode

## Dependencies

- `flutter_onnxruntime: ^1.7.1` — ONNX Runtime inference
- `http: ^1.2.0` — model downloads
- `path_provider: ^2.1.0` — model storage path
- `archive: ^4.0.0` — `.tar.bz2` extraction

## Testing

- `flutter test` — all tests
- Unit tests in `test/` — no network, fast, pure logic validation
- Integration tests (future) — download smallest model, transcribe WAV fixtures
- WAV fixtures in `test/fixtures/` — 4 files (DE/EN/FR/ES) with known text

## Conventions

- SDK `>=3.0.0`, null safety enabled
- Pure functions for `Isolate.run()` — must be top-level or static, no closures capturing native resources
- Each engine receives `OnnxRuntime` in constructor and creates its own `OrtSession` instances
- Dispose `OrtValue` tensors and `OrtSession` in `dispose()` — native memory leak otherwise

## Common pitfalls

- Forget `.close()` on `OrtSession` in `dispose()` → native memory leak
- Pass non-sendable objects to `Isolate.run()` → runtime error
- Expect real-time streaming from batch API → use `transcribeFile` or `transcribeBuffer`
