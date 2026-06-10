# stt_flutter — Agent Guide

## Project overview

Fully local on-device speech-to-text for Flutter via `sherpa_onnx` native FFI.
Supports 7 model families: Whisper, Sherpa Zipformer, NeMo Parakeet, Canary,
SenseVoice, Omnilingual, Qwen3-ASR.

## Key commands

```bash
# Analyze
flutter analyze

# Run all tests (97 total)
flutter test

# Run single test
flutter test test/audio_processor_test.dart

# Publish dry-run
dart pub publish --dry-run
```

## Architecture

```
lib/
├── stt_flutter.dart                    # Public exports
├── src/
│   ├── stt_flutter_impl.dart           # SttFlutter facade
│   ├── stt/stt_engine.dart             # SttEngine singleton wrapper
│   ├── stt_config.dart                 # SttConfig
│   ├── stt_result.dart                 # SttResult model
│   ├── stt_logger.dart / stt_exception.dart
│   ├── cancellation_token.dart
│   ├── compute_worker.dart             # Isolate.run for resample
│   ├── model_registry.dart             # ModelRegistry, ModelDescriptor, ModelFile
│   ├── model_downloader.dart           # HTTP download + SHA256 + tar.bz2
│   ├── audio/
│   │   ├── audio_buffer.dart           # AudioBuffer (Float32List + sampleRate)
│   │   ├── audio_processor.dart        # Resample, WAV parse, PreprocessConfig, denoiser
│   │   ├── audio_chunker.dart          # ChunkingConfig, chunkBuffer, dedupJoinedText
│   │   ├── audio_capture.dart          # Streaming capture → Float32
│   │   └── vad.dart                    # EnergyVadEngine, SherpaOnnxVadEngine
│   ├── language/
│   │   └── language_detector.dart      # sherpa_onnx SLI wrapper
│   ├── engines/
│   │   ├── inference_engine.dart       # Abstract: load, transcribe(options), dispose
│   │   ├── engine_factory.dart         # ModelDescriptor → InferenceEngine
│   │   ├── offline_engine_base.dart    # Shared: file-lookup, thread-count, warn
│   │   ├── whisper/whisper_engine.dart
│   │   ├── sherpa/sherpa_engine.dart
│   │   ├── canary/canary_engine.dart
│   │   ├── nemo/nemo_engine.dart
│   │   ├── sensevoice/sensevoice_engine.dart
│   │   ├── omnilingual/omnilingual_engine.dart
│   │   └── qwen3asr/qwen3asr_engine.dart
│   └── default_models/
│       ├── whisper_models.dart         # 10 Whisper variants
│       ├── sherpa_models.dart          # Zipformer + Parakeet
│       ├── canary_models.dart
│       ├── sensevoice_models.dart
│       ├── omnilingual_models.dart
│       ├── qwen_models.dart
│       └── register_defaults.dart
test/                                  # 12 test files, 97 tests
example/
├── assets/denoisers/{gtcrn,dpdfnet}/
├── lib/main.dart + screens/
└── test/
```

## Key API patterns

### Transcribe with options
```dart
// Default (greedy)
await stt.transcribeFile('audio.wav');

// beamSearch flag (Nemo logs warning and falls back to greedy)
await stt.transcribeFile('audio.wav', beamSearch: true);

// With preprocessing
await stt.transcribeFile('audio.wav', preprocess: PreprocessConfig(
  denoiserType: DenoiserType.gtcrn,
  denoiserModelDir: '/path/to/model',
  highPass: true,
  gain: 1.3,
  normalize: NormalizeMode.peak,
));
```

### Options plumbing
Each engine receives `Map<String, dynamic>? options` on `transcribe()`.
Currently only `{'beamSearch': true}` is defined — only `NemoInferenceEngine`
reads it (and warns that beam search is unsupported for Nemo transducer).

## Conventions

- **No comments in production code** — let the code speak. Comments OK in test files.
- **SDK**: Dart `^3.4.4`, Flutter `>=3.7.0`. CI runs Flutter 3.44.x.
- **All inference**: main isolate via native FFI (`sherpa_onnx` is non-blocking).
- **Resampling**: `AudioProcessor.resampleSync()` before passing to engine.
- **Language**: never hardcoded — flows as nullable string through the API.
  Three modes: auto (null), default (from loadModel), forced per-call.
- **Chunking**: 30s windows, 2s overlap for transducer models, 5s for Whisper.
  `dedupJoinedText` strips overlap duplicates.
- **Denoisers**: GTCRN (535 KB) and DPDFNet (10 MB) via sherpa_onnx
  `OfflineSpeechDenoiser`. Applied first in the preprocess pipeline.
- **Tests**: `flutter_test` with `group`/`test`. Mock engines extend
  `OfflineEngineBase`. Mock HTTP via custom `http.Client` closure.

## Model registry

```dart
ModelRegistry.register(ModelDescriptor(
  id: 'my-model', type: SttModelType.whisper, languages: ['en'],
  files: [ModelFile(url: '...', filename: 'encoder.onnx')],
  sizeMb: 150,
));
```

## Debugging

- `SttLogger.i` / `SttLogger.d` / `SttLogger.w` / `SttLogger.e` for structured logs
- `SttResult` carries `inferenceTimeMs`, `lang`, `confidence`, `durationMs`
- CI badge: ![CI](https://github.com/sebbom/stt_flutter_package/actions/workflows/ci.yaml/badge.svg)
