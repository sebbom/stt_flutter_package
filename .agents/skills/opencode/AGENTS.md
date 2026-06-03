# stt_flutter — agent instructions

## Quick start

```bash
flutter pub get
flutter test                                # unit tests (20 pass)
dart analyze lib/                           # zero errors
```

## Architecture

- **STT engine**: `SttEngine` singleton → `SttFlutter` → `WhisperInferenceEngine` / `SherpaInferenceEngine`. All inference via `sherpa_onnx` native FFI (no `flutter_onnxruntime`).
- **Threading**: Audio preprocessing (WAV parsing, resampling) runs in ephemeral `Isolate.run()` background isolates. `sherpa_onnx` native calls are non-blocking on the main isolate.
- **Model registry**: `ModelRegistry` singleton. Default models registered on first access via `_ensureDefaults()`. Add custom models with `ModelRegistry.register(...)`.
- **Two engines** (each in `lib/src/engines/`):
  - `whisper/` — `sherpa_onnx.OfflineRecognizer` (Whisper, SenseVoice, etc.)
  - `sherpa/` — `sherpa_onnx.OnlineRecognizer` (Zipformer transducer, NeMo Parakeet)
- **Voxtral removed** — `SttModelType.voxtral` throws `UnsupportedError`

## Dependencies

- `sherpa_onnx: ^1.13.2` — Native ONNX Runtime inference (replaces flutter_onnxruntime)
- `http: ^1.2.0` — model downloads
- `path_provider: ^2.1.0` — model storage path
- `archive: ^4.0.0` — `.tar.bz2` extraction

## Testing

- `flutter test` — all tests (20 pass)
- Unit tests in `test/` — no network, fast, pure logic validation
- WAV fixtures in `test/fixtures/`

## Conventions

- SDK `>=3.0.0`, null safety enabled
- `sherpa_onnx.initBindings()` must be called before creating any recognizer (done in `SttEngine.loadModel()`)
- Engines create `OnlineRecognizer` / `OfflineRecognizer` in `load()`, dispose in `dispose()`
- `AudioBuffer.samples` is `Float32List` normalized to `[-1, 1]`
- Pure functions for `Isolate.run()` — must be top-level or static, no closures capturing native resources

## Common pitfalls

- Forget `recognizer.free()` / `stream.free()` → native memory leak
- `sherpa_onnx` and `flutter_onnxruntime` cannot coexist (native lib symbol conflict)
- Pass non-sendable objects to `Isolate.run()` → runtime error
