# `stt_flutter` — Architecture & Implementation Plan

Fully local, on-device speech-to-text for Flutter using ONNX models via `sherpa_onnx`.

> **Language is never hardcoded.** Every engine accepts a `language` parameter
> and reports what it actually produced in `SttResult.lang`. There are three
> modes: **auto-detect** (no language anywhere), **default from `loadModel`**,
> and **forced per-call** (always wins).

---

## Threading Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Main Isolate (UI)                           │
│                                                                   │
│  SttEngine (singleton)                                            │
│    └─ SttFlutter                                                  │
│         ├─ WhisperInferenceEngine  (OfflineRecognizer, whisper)    │
│         ├─ SherpaInferenceEngine   (OfflineRecognizer, zipformer2) │
│         ├─ NemoInferenceEngine     (OfflineRecognizer, nemo_trans) │
│         └─ CanaryInferenceEngine   (OfflineRecognizer, canary)     │
│                                                                   │
│  All inference: native sherpa_onnx FFI — non-blocking on main    │
│                                                                   │
│  Audio preprocessing                                              │
│    ├─ AudioProcessor.loadWav()       (async I/O, main isolate)    │
│    ├─ Isolate.run(resampleSync)      (ephemeral bg isolate)       │
│    └─ AudioBuffer → engine.transcribe(audio, language: ...)       │
│         └─ sherpa_onnx native FFI — non-blocking                  │
│                                                                   │
│  Audio capture (streaming)                                        │
│    ├─ AudioCaptureService            (record package, Float32)    │
│    ├─ VadEngine                      (energy or Silero VAD)      │
│    └─ TranscriptionService           (per-chunk processing)       │
│                                                                   │
│  Optional language detection fallback                             │
│    └─ LanguageDetector               (sherpa_onnx SLI Whisper)   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Why not a long-lived background isolate:**
`sherpa_onnx` uses native FFI calls that do **not** block the Dart event loop.
Audio preprocessing (resampling) is offloaded to one-shot `Isolate.run()` calls.
The native `OfflineRecognizer` handles all ONNX Runtime
management internally — no manual session or tensor management needed.

**Native sherpa_onnx benefits:**
- No `flutter_onnxruntime` dependency (eliminates native library conflicts)
- `initBindings()` loads the shared library once globally
- `OfflineRecognizer` manages encoder/decoder/joiner internally
- Built-in support for Zipformer transducer, NeMo Parakeet, Whisper, Paraformer, CTC, Canary, and more
- Built-in `SpokenLanguageIdentification` for the language-detection fallback

---

## Model Registry — Extensible by Design

Users register any ONNX model in one line. The package ships with seeded models.

```dart
// --- Core types ---
enum SttModelType { whisper, sherpa, nemo, canary }  // 4 supported types

class ModelDescriptor {
  final String id;                // "whisper-tiny", "sherpa-zipformer-en"
  final String name;              // Human-readable label
  final SttModelType type;
  final List<String> languages;   // ["en", "de", "fr", "es"]
  final List<ModelFile> files;    // Ordered list of files to download
  final int sizeMb;               // Approximate total download size
}

class ModelFile {
  final String url;               // Download URL
  final String filename;          // Local filename
  final String? sha256;           // Optional integrity hash
}

// --- Singleton registry ---
class ModelRegistry {
  static void register(ModelDescriptor model);
  static List<ModelDescriptor> available({SttModelType? type});
  static ModelDescriptor get(String id);
}
```

### Adding a custom model later:

```dart
ModelRegistry.register(ModelDescriptor(
  id: 'my-custom-whisper',
  name: 'My Custom Whisper',
  type: SttModelType.whisper,
  languages: ['ja', 'ko', 'zh'],
  files: [
    ModelFile(url: '...', filename: 'encoder.onnx'),
    ModelFile(url: '...', filename: 'decoder.onnx'),
  ],
  sizeMb: 220,
));
```

Adding a **new engine type** = implement `InferenceEngine` + add to the factory
map in `engine_factory.dart`.

---

## Seeded Models

| ID | Type | Languages | Source | Size |
|----|------|-----------|--------|------|
| `whisper-tiny` | Whisper | 99+ langs | HF ONNX | ~150 MB |
| `whisper-tiny.en` | Whisper | en | HF ONNX | ~150 MB |
| `whisper-base` | Whisper | 99+ langs | HF ONNX | ~240 MB |
| `whisper-base.en` | Whisper | en | HF ONNX | ~240 MB |
| `whisper-small` | Whisper | 99+ langs | HF ONNX | ~460 MB |
| `whisper-small.en` | Whisper | en | HF ONNX | ~460 MB |
| `whisper-medium` | Whisper | 99+ langs | HF ONNX | ~960 MB |
| `whisper-medium.en` | Whisper | en | HF ONNX | ~960 MB |
| `whisper-large-v3` | Whisper | 99+ langs | HF ONNX | ~950 MB |
| `whisper-large-v3-turbo` | Whisper | 99+ langs | HF ONNX | ~550 MB |
| `sherpa-zipformer-en` | Sherpa | en | k2-fsa GH | ~300 MB |
| `parakeet-tdt-0.6b-multilingual` | NeMo Parakeet | 25 langs | HF ONNX | ~400 MB |
| `canary-180m-en-es-de-fr` | Canary | en, es, de, fr | HF ONNX | ~200 MB |

Sherpa models are downloaded as `.tar.bz2` and extracted via `package:archive`.
Whisper, Parakeet, Nemo, and Canary models use sherpa-onnx's individual-file ONNX format.

---

## Model Download System

```dart
class ModelDownloader {
  /// Default storage: {appDocDir}/stt_models/{modelId}/
  static Future<String> defaultStoragePath(ModelDescriptor model);

  /// Download all files with progress callbacks.
  /// Caller may inject an [http.Client] for testing; otherwise one is
  /// created internally and closed after use.
  static Future<void> download(
    ModelDescriptor model, {
    String? storagePath,
    http.Client? client,
    void Function(int received, int total)? onProgress,
    void Function(String file, int received, int total)? onFileProgress,
  });

  /// Check if all files exist
  static Future<bool> isDownloaded(ModelDescriptor model, {String? storagePath});
}
```

- `ModelDownloader` — downloads from HuggingFace / GitHub via `http` package, with SHA256 verification.

---

## Engine Interface

```dart
abstract class InferenceEngine {
  /// Load model files. [modelFiles] maps logical names to absolute paths.
  Future<void> load(Map<String, String> modelFiles);

  /// Transcribe audio and return text.
  /// [language] is an optional ISO-639-1 code (e.g. 'de'). Pass `null`
  /// to let the engine auto-detect (or use the model's default).
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
  });

  /// Release all native resources (recognizer, streams).
  Future<void> dispose();

  /// Whether this engine accepts an explicit [language] override at
  /// transcribe time. False for monolingual models (e.g. zipformer-en).
  bool get supportsExplicitLanguage;

  /// Set of language codes this model declares. May be empty.
  Set<String> get supportedLanguages;
}

/// Internal audio representation (16kHz mono Float32).
class AudioBuffer {
  final Float32List samples;
  final int sampleRate; // always 16000 after preprocessing
}

class SttResult {
  final String text;
  final double inferenceTimeMs;
  final String? lang;         // detected or forced language
  final double? confidence;
  final double? durationMs;
}
```

All four engine implementations share `lib/src/engines/offline_engine_base.dart`
which provides the model-file lookup, the `recognizer` lifecycle, and a shared
`warnIfLanguageUnsupported` helper.

---

## Language Modes

The package supports three explicit language modes. The language is **never
hardcoded at the call site** — it always flows from one of these three sources:

1. **Auto-detect** — caller passes `null` everywhere. Engine decides:
   - Whisper: auto-detects (no `language` set on the stream).
   - Parakeet / Zipformer / Canary: uses the model's default tokenizer
     (no `language` is set on the stream).
2. **Default from `loadModel`** — caller passes `defaultLanguage: 'de'` to
   `SttEngine.loadModel`. The engine stores it. Every subsequent
   `transcribeFile`/`transcribeBuffer` with no per-call override uses it.
3. **Forced per-call** — caller passes `language: 'fr'` to
   `transcribeFile`/`transcribeBuffer`. This **always wins** over the
   default. The engine writes the language to the stream via
   `setOption('language', code)`. `SttResult.lang` reflects what the
   engine actually returned so the caller can verify.

Engine behaviour per language mode:

| Engine | `supportsExplicitLanguage` | What forced-per-call does | What auto-detect does |
|---|---|---|---|
| Whisper | ✅ | `stream.setOption('language', code)` + long-form chunking | `language: ''` (Whisper auto-detects) |
| Sherpa (zipformer) | ❌ | Logs warning, uses model's native language | Same — model is monolingual |
| Nemo (Parakeet) | ✅ | `stream.setOption('language', code)` | `result.lang` is empty (Parakeet doesn't tag tokens with language) — fallback to `LanguageDetector` |
| Canary | ✅ | `stream.setOption('srcLang'/'tgtLang', code)` | Uses `model.languages.first` set at `load()` time |

When a forced language is **not in** the model's `supportedLanguages`, the
engine logs a warning (`SttLogger.w`) and continues. This is a soft signal
that the caller is misusing the model without breaking the transcription.

---

## Whisper Engine

**Files needed:** `encoder.onnx`, `decoder.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'whisper'`.

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `whisper_engine.dart` | `WhisperInferenceEngine` | Long-form chunking + `setOption('language', ...)` + token-level dedup |
| `offline_engine_base.dart` | `OfflineEngineBase` | Shared file-lookup + thread-count + warn helper |

### Long-form chunking

Whisper has a hard 30 s limit per stream. The engine:
1. Splits the input into 30 s windows with a 5 s overlap.
2. Calls `stream.setOption('language', code)` on each fresh `OfflineStream`.
3. `acceptWaveform` + `decode` + `getResult` per chunk.
4. Concatenates decoded text with a token-level dedup at chunk boundaries
   (strips the overlap tail when the next chunk's head matches the previous
   chunk's tail — case-insensitive, with last-4-character fallback for noisy
   tokens).

`enableSegmentTimestamps: true` and `tailPaddings: -1` are set at `load()`
time, mirroring sherpa-onnx's recommended config for chunked decoding.

---

## Sherpa Transducer Engine

**Files needed:** `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'zipformer2'`.

The Sherpa Zipformer model is English-only by design. The engine accepts the
`language` parameter for API symmetry but always logs a warning if a value
other than the model's `languages` is provided. `SttResult.lang` is populated
from `ModelDescriptor.languages.first` (typically `en`).

---

## NeMo Parakeet Engine

**Files needed:** `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'nemo_transducer'`.

Parakeet TDT 0.6b is multilingual. When the caller forces a language, the
engine calls `stream.setOption('language', code)` per stream. `SttResult.lang`
is filled in from the recognizer's result, or falls back to the LanguageDetector
wrapper if the recognizer returns empty.

---

## Canary Engine

**Files needed:** `encoder.int8.onnx`, `decoder.int8.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'canary'`. Supports
source/target language via `OfflineCanaryModelConfig.srcLang` / `tgtLang`
(dynamically set at transcribe time via `stream.setOption('srcLang', ...)` and
`stream.setOption('tgtLang', ...)`).

`OfflineCanaryModelConfig` is initialised at `load()` time with
`srcLang = tgtLang = model.languages.first` so a fresh recognizer always has a
sensible default. Per-stream `setOption` is only called when the effective
language differs from this default — avoiding an unnecessary FFI round-trip.

---

## Public API

```dart
/// Main entry point. Runs on the main isolate, delegates to native sherpa_onnx.
class SttFlutter {
  /// Initialize: loads model files, creates sherpa_onnx recognizer.
  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,        // defaults to {appDocDir}/stt_models/{model.id}
    String? language,        // default language (e.g. 'de'). null = auto-detect
  });

  /// Transcribe a WAV file at [path].
  Future<SttResult> transcribeFile(String path, {String? language, CancellationToken? token});

  /// Transcribe raw PCM [samples] (Float32, [-1.0, 1.0]) at [sampleRate] Hz.
  Future<SttResult> transcribeBuffer(
    Float32List samples,
    int sampleRate, {
    String? language,
    CancellationToken? token,
  });

  /// Release all resources.
  Future<void> dispose();
}

/// Singleton convenience wrapper around SttFlutter.
class SttEngine {
  static SttEngine get instance;

  /// Returns `null` on success, [SttException] on failure.
  Future<SttException?> loadModel(
    ModelDescriptor model, {
    String? modelDir,
    String? defaultLanguage,
  });

  Future<SttResult> transcribeFile(String path, {String? language});
  Future<SttResult> transcribeBuffer(Float32List samples, int sampleRate, {String? language});
  void cancel();
  Future<void> destroy();
  bool get isReady;
  ModelDescriptor? get currentModel;
  String? get currentDefaultLanguage;
}
```

---

## File Structure

```
lib/
├── stt_flutter.dart                       # Public exports
├── src/
│   ├── stt_flutter_impl.dart              # SttFlutter (main isolate facade)
│   ├── stt_config.dart                    # SttModelType, SttConfig
│   ├── stt_result.dart                    # SttResult (text, lang, confidence, durationMs)
│   ├── stt_logger.dart                    # Structured logging
│   ├── stt_exception.dart                 # Custom exception types
│   ├── cancellation_token.dart            # CancellationToken
│   ├── compute_worker.dart                # One-shot Isolate.run for resample
│   ├── model_registry.dart                # ModelRegistry, ModelDescriptor
│   ├── model_downloader.dart              # HTTP download + progress + tar.bz2 extract
│   ├── stt/
│   │   └── stt_engine.dart                # SttEngine (singleton, initBindings)
│   ├── audio/
│   │   ├── audio_buffer.dart              # AudioBuffer data class
│   │   ├── audio_processor.dart           # Resample, WAV parse (16/24/32-bit, float, multi-channel)
│   │   ├── audio_capture.dart             # Streaming audio capture (record package, Float32)
│   │   └── vad.dart                       # SherpaOnnxVadEngine wrapper
│   ├── language/
│   │   └── language_detector.dart         # sherpa_onnx SpokenLanguageIdentification wrapper
│   ├── engines/
│   │   ├── inference_engine.dart          # Abstract InferenceEngine
│   │   ├── engine_factory.dart            # ModelDescriptor → InferenceEngine
│   │   ├── offline_engine_base.dart       # Shared scaffolding for all 4 engines
│   │   ├── whisper/whisper_engine.dart    # OfflineRecognizer, modelType: 'whisper'
│   │   ├── sherpa/sherpa_engine.dart      # OfflineRecognizer, modelType: 'zipformer2'
│   │   ├── canary/canary_engine.dart      # OfflineRecognizer, modelType: 'canary'
│   │   └── nemo/nemo_engine.dart          # OfflineRecognizer, modelType: 'nemo_transducer'
│   └── default_models/
│       ├── whisper_models.dart            # All 10 Whisper variants (FP32 HF)
│       ├── sherpa_models.dart             # Zipformer EN (tar.bz2) + Parakeet TDT (HF)
│       └── canary_models.dart             # Canary 180M (HF)
test/
├── stt_flutter_test.dart
├── model_registry_test.dart
├── audio_processor_test.dart
├── audio_capture_test.dart
├── engine_factory_test.dart
├── language_handling_test.dart
├── model_downloader_test.dart
└── fixtures/
    └── hello_en.wav
example/
├── pubspec.yaml
├── lib/
│   ├── main.dart
│   └── screens/
│       ├── model_selection_screen.dart
│       └── transcription_screen.dart
```

---

## Dependencies

```yaml
dependencies:
  flutter:
    sdk: flutter
  sherpa_onnx: ^1.13.2          # Native ONNX inference (replaces flutter_onnxruntime)
  http: ^1.2.0                  # Model downloads
  path_provider: ^2.1.0         # Model storage path
  archive: ^4.0.0               # .tar.bz2 extraction (Sherpa models)
  file: ^7.0.0                  # File utilities
  record: ^7.0.0                # Audio capture
  device_info_plus: ^11.0.0     # Device capability detection
  crypto: ^3.0.0                # SHA256 verification

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^5.0.0         # Lint rules
```

---

## Implementation Order

| # | Step | Files | Verification |
|---|------|-------|--------------|
| 1 | Project scaffold | `pubspec.yaml`, `analysis_options.yaml`, `lib/stt_flutter.dart` | `flutter pub get` |
| 2 | Model descriptors | `model_registry.dart`, `whisper_models.dart`, `sherpa_models.dart` | `flutter test` (registry unit tests) |
| 3 | Model downloader | `model_downloader.dart` | Unit test with mock HTTP |
| 4 | Audio processing | `audio_buffer.dart`, `audio_processor.dart` | Unit test with known WAV files |
| 5 | Engine scaffolding | `offline_engine_base.dart`, `engine_factory.dart` | `flutter test` (engine_factory_test) |
| 6 | All four engines | `whisper/`, `sherpa/`, `canary/`, `nemo/` engines | `flutter test` (engine_factory_test) |
| 7 | SttFlutter plumbing | `stt_flutter_impl.dart` (language mode + LanguageDetector fallback) | `flutter test` (language_handling_test) |
| 8 | Singleton SttEngine | `stt/stt_engine.dart` (defaultLanguage + Object return) | `flutter test` (stt_flutter_test) |
| 9 | Streaming + VAD | `audio/audio_capture.dart`, `audio/vad.dart` | Real-time recording test |
| 10 | Example app | `main.dart`, `model_selection_screen.dart`, `transcription_screen.dart` | `flutter run` on device |

---

## Test Strategy

| Test | Type | Verifies |
|------|------|----------|
| `audio_processor_test.dart` | Unit | WAV parsing (8/16/24/32-bit, IEEE float, multi-channel), resample to 16kHz |
| `audio_capture_test.dart` | Unit | PCM16 → Float32 conversion, range, edge cases |
| `model_registry_test.dart` | Unit | Register, lookup, available, duplicates |
| `engine_factory_test.dart` | Unit | Each model type returns the correct engine with the right `supportsExplicitLanguage` / `supportedLanguages` |
| `language_handling_test.dart` | Unit | Per-call override wins over default; auto-detect path; `SttResult.lang` is preserved |
| `model_downloader_test.dart` | Unit | Mock HTTP: SHA256 success, SHA256 mismatch deletes file, 404 throws, user-supplied client is not auto-closed |
| `stt_flutter_test.dart` | Unit | Registry, model descriptor validation |
