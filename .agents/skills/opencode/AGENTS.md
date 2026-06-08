# stt_flutter — agent instructions

## Quick start

```bash
flutter pub get
flutter test                                # 87 library + 14 example = 101 tests
flutter analyze lib/ example/lib/           # zero errors
```

## Architecture

- **STT engine**: `SttEngine` singleton → `SttFlutter` → one of seven `InferenceEngine`s. All inference via `sherpa_onnx` native FFI (no `flutter_onnxruntime`).
- **Threading**: Audio preprocessing (WAV parsing, resampling) runs in one-shot `Isolate.run()` background isolates. `sherpa_onnx` native calls are non-blocking on the main isolate.
- **Model registry**: `ModelRegistry` singleton. Default models registered on first access via `_ensureDefaults()`. Add custom models with `ModelRegistry.register(...)`.
- **Seven engines** (each in `lib/src/engines/`, all share `offline_engine_base.dart`):
  - `whisper/` — `sherpa_onnx.OfflineRecognizer` (Whisper, multilingual or English-only)
  - `sherpa/` — `sherpa_onnx.OfflineRecognizer` (Zipformer transducer, English-only)
  - `nemo/` — `sherpa_onnx.OfflineRecognizer` (NeMo Parakeet TDT, multilingual)
  - `canary/` — `sherpa_onnx.OfflineRecognizer` (Canary, en/es/de/fr)
  - `sensevoice/` — `sherpa_onnx.OfflineRecognizer` (SenseVoice, zh/en/ja/ko/yue + emotion/events)
  - `omnilingual/` — `sherpa_onnx.OfflineRecognizer` (Omnilingual ASR, 1600 languages)
  - `qwen3asr/` — `sherpa_onnx.OfflineRecognizer` (Qwen3-ASR 0.6B, multilingual)
- **Language is never hardcoded**. Three modes:
  1. `null` everywhere → auto-detect (engine decides).
  2. `loadModel(defaultLanguage: 'de')` → default for that engine.
  3. `transcribeFile(path, language: 'fr')` → forced per-call, always wins.
- **Language detection fallback**: `lib/src/language/language_detector.dart` wraps `sherpa_onnx.SpokenLanguageIdentification` (Whisper-tiny SLI). Used to populate `SttResult.lang` when the engine doesn't return one (Parakeet, Zipformer).
- **Audio chunking**: `lib/src/audio/audio_chunker.dart`. All non-Whisper engines chunk at 30 s / 2 s overlap. Whisper keeps 30 s / 5 s for back-compat. `dedupJoinedText` strips fuzzy-matched prefix from chunk N+1.
- **Denoiser**: sherpa-onnx `OfflineSpeechDenoiser` (GTCRN / DPDFNet). Applied as first step in preprocessing pipeline. Denoiser ONNX models bundled in example app via Flutter assets; library accepts a plain file path.
- **Hotwords**: Zipformer uses `hotwordsFile` + `hotwordsScore` on `OfflineRecognizerConfig`. Qwen3 uses `OfflineQwen3AsrModelConfig.hotwords` (comma-separated string). Wired through `SttEngine.loadModel(hotwords:)` and `SttEngine.setHotwords`.
- **Preprocessing pipeline**: denoiser → high-pass (80 Hz RC IIR) → gain → normalize (peak or RMS). Exposed via `PreprocessConfig` and `AudioProcessor.applyPreprocessAsync`.

## Dependencies

- `sherpa_onnx: ^1.13.2` — Native ONNX Runtime inference (replaces flutter_onnxruntime)
- `http: ^1.2.0` — model downloads
- `path_provider: ^2.1.0` — model storage path
- `archive: ^4.0.0` — `.tar.bz2` extraction

## Testing

- `flutter test` — 101 tests (87 library + 14 example)
- Unit tests in `test/` — no network, fast, pure logic validation
- WAV fixtures in `test/fixtures/hello_en.wav`
- Use `SttFlutter.withEngine(...)` (`@visibleForTesting`) to test plumbing without loading a real model.

## Conventions

- SDK `>=3.0.0`, null safety enabled
- `sherpa_onnx.initBindings()` must be called before creating any recognizer (done in `SttEngine.loadModel()`)
- Engines extend `OfflineEngineBase` and declare `modelType` / `supportsExplicitLanguage` in their concrete class
- `AudioBuffer.samples` is `Float32List` normalized to `[-1, 1]`
- `Isolate.run` is used for one-shot compute (resample). No long-lived background isolate.

## Common pitfalls

- Forget `recognizer.free()` / `stream.free()` → native memory leak
- `sherpa_onnx` and `flutter_onnxruntime` cannot coexist (native lib symbol conflict)
- Pass non-sendable objects to `Isolate.run()` → runtime error
- Forcing a language not in `ModelDescriptor.languages` will log a warning (intentional) and continue; the engine decides whether to honor it
- For monolingual models (zipformer-en) `supportsExplicitLanguage == false`; passing any non-matching language is a warning, not an error
- Qwen3 C++ `Validate()` requires `tokenizer` to be a directory — `ModelFile.filename` must use `tokenizer/` subdir prefix
- Qwen3 has no auto-detect mode — when no language is supplied, defaults to `'en'` with a warning log
- SenseVoice emotion tags (`<|HAPPY|>`, etc.) are extracted separately from text; parser uses `<|` / `>` pairs
- Denoiser assets in example app are extracted to temp dir on first use via `DenoiserBundle`; library only sees plain file paths
