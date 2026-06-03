# `stt_flutter` — Architecture & Implementation Plan

Fully local, on-device speech-to-text for Flutter using ONNX models via `sherpa_onnx`.

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
│    └─ AudioBuffer → engine.transcribe(audio)                      │
│         └─ sherpa_onnx native FFI — non-blocking                  │
│                                                                   │
│  Audio capture (streaming)                                        │
│    ├─ AudioCaptureService            (record package)             │
│    ├─ VadEngine                      (energy or Silero VAD)      │
│    └─ TranscriptionService           (per-chunk processing)       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Why not a long-lived background isolate:**
`sherpa_onnx` uses native FFI calls that do **not** block the Dart event loop.
Audio preprocessing (resampling) is offloaded to ephemeral `Isolate.run()` calls.
The native `OfflineRecognizer` handles all ONNX Runtime
management internally — no manual session or tensor management needed.

**Native sherpa_onnx benefits:**
- No `flutter_onnxruntime` dependency (eliminates native library conflicts)
- `initBindings()` loads the shared library once globally
- `OfflineRecognizer` manages encoder/decoder/joiner internally
- Built-in support for Zipformer transducer, NeMo Parakeet, Whisper, Paraformer, CTC, Canary, and more

---

## Model Registry — Extensible by Design

Users register any ONNX model in one line. The package ships with seeded models.

```dart
// --- Core types ---
enum SttModelType { whisper, sherpa, nemo, canary, voxtral }  // voxtral → UnsupportedError

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
  languages: ['ja'],
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
| `whisper-tiny` | Whisper | 99 langs | HF ONNX | ~220 MB |
| `whisper-tiny.en` | Whisper | en | HF ONNX | ~220 MB |
| `whisper-base` | Whisper | 99 langs | HF ONNX | ~370 MB |
| `whisper-base.en` | Whisper | en | HF ONNX | ~370 MB |
| `whisper-small` | Whisper | 99 langs | HF ONNX | ~1.1 GB |
| `whisper-small.en` | Whisper | en | HF ONNX | ~1.1 GB |
| `whisper-medium` | Whisper | 99 langs | HF ONNX | ~2.5 GB |
| `whisper-medium.en` | Whisper | en | HF ONNX | ~2.5 GB |
| `whisper-large-v3` | Whisper | 99 langs | HF ONNX | ~4.5 GB |
| `whisper-large-v3-turbo` | Whisper | 99 langs | HF ONNX | ~2.5 GB |
| `sherpa-zipformer-en` | Sherpa | en | k2-fsa GH | ~35 MB |
| `parakeet-tdt-0.6b-multilingual` | NeMo Parakeet | 25 langs | HF ONNX | ~400 MB |
| `canary-180m-flash` | Canary | en | HF ONNX | ~180 MB |

Sherpa models are downloaded as `.tar.bz2` and extracted via `package:archive`.
Whisper, Parakeet, Nemo, and Canary models use sherpa-onnx's individual-file ONNX format.

---

## Model Download System

Two downloaders coexist for different model sources:

```dart
class ModelDownloader {
  /// Default storage: {appDocDir}/stt_models/{modelId}/
  static Future<String> defaultStoragePath(ModelDescriptor model);

  /// Download all files with progress callbacks
  static Future<void> download(
    ModelDescriptor model, {
    String? storagePath,
    void Function(int received, int total)? onProgress,
    void Function(String file, int received, int total)? onFileProgress,
    CancelToken? cancelToken,
  });

  /// Check if all files exist
  static Future<bool> isDownloaded(ModelDescriptor model, {String? storagePath});
}
```

- `ModelDownloader` — downloads from HuggingFace / GitHub via `http` package, with SHA256 verification

---

## Engine Interface

```dart
abstract class InferenceEngine {
  /// Load model files. [modelFiles] maps logical names to absolute paths.
  Future<void> load(Map<String, String> modelFiles);

  /// Transcribe audio and return text.
  Future<SttResult> transcribe(AudioBuffer audio, {String? language});

  /// Release all native resources (recognizer, streams).
  Future<void> dispose();
}

/// Internal audio representation (16kHz mono Float32).
class AudioBuffer {
  final Float32List samples;
  final int sampleRate; // always 16000 after preprocessing
}

class SttResult {
  final String text;
  final double inferenceTimeMs;
  final String? lang;  // detected language (Whisper, Canary)
}
```

---

## Whisper Engine

**Files needed:** `encoder.onnx`, `decoder.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'whisper'`.

```
WAV file ──► resample to 16kHz mono ──► OfflineStream.acceptWaveform()
     ──► OfflineRecognizer.decode(stream)
     ──► OfflineRecognizer.getResult(stream).text
```

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `whisper_engine.dart` | `WhisperInferenceEngine` | Loads model → creates `OfflineRecognizer` → feeds stream → returns text |
| | | Language support via `OfflineWhisperModelConfig.language` |

---

## Sherpa Transducer Engine

**Files needed:** `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'zipformer2'`.

```
WAV file ──► resample to 16kHz mono ──► OfflineStream.acceptWaveform()
     ──► OfflineRecognizer.decode(stream)
     ──► OfflineRecognizer.getResult(stream).text
```

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `sherpa_engine.dart` | `SherpaInferenceEngine` | Loads model → creates `OfflineRecognizer` → feeds stream → returns text |

The `OfflineRecognizer` internally handles:
- Fbank feature extraction
- Encoder forward pass
- Transducer greedy search (joiner + decoder)
- Token lookup via `tokens.txt`

---

## NeMo Parakeet Engine

**Files needed:** `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'nemo_transducer'`.

```
WAV file ──► resample to 16kHz mono ──► OfflineStream.acceptWaveform()
     ──► OfflineRecognizer.decode(stream)
     ──► OfflineRecognizer.getResult(stream).text
```

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `nemo_engine.dart` | `NemoInferenceEngine` | Loads model → creates `OfflineRecognizer` → feeds stream → returns text |

---

## Canary Engine

**Files needed:** `encoder.onnx`, `decoder.onnx`, `tokens.txt`

Uses `sherpa_onnx.OfflineRecognizer` with `modelType: 'canary'`.
Supports source/target language via `OfflineCanaryModelConfig.srcLang` / `tgtLang`
(dynamically set at transcribe time via `stream.setOption()`).

```
WAV file ──► resample to 16kHz mono ──► OfflineStream.acceptWaveform()
     ──► OfflineRecognizer.decode(stream)
     ──► OfflineRecognizer.getResult(stream).text
     ──► OfflineRecognizerResult.lang   (detected language)
```

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `canary_engine.dart` | `CanaryInferenceEngine` | Loads model → creates `OfflineRecognizer` → feeds stream → returns text + lang |

---

## Public API

```dart
/// Main entry point. Runs on the main isolate, delegates to native sherpa_onnx.
class SttFlutter {
  /// Initialize: loads model files, creates sherpa_onnx recognizer.
  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,   // defaults to {appDocDir}/stt_models/{model.id}
    String? language,   // default language (e.g. 'de')
  });

  /// Transcribe a WAV file at [path].
  Future<SttResult> transcribeFile(String path);

  /// Transcribe raw PCM [samples] (Float32, [-1.0, 1.0]) at [sampleRate] Hz.
  Future<SttResult> transcribeBuffer(Float32List samples, int sampleRate);

  /// Release all resources.
  Future<void> dispose();
}

/// Singleton convenience wrapper around SttFlutter.
class SttEngine {
  static SttEngine get instance;
  void loadModel(ModelDescriptor model, {String? modelDir, String? language});
  Future<SttResult> transcribeFile(String path, {String? language});
  Future<SttResult> transcribeBuffer(Float32List samples, int sampleRate, {String? language});
  void cancel();
  Future<void> destroy();
  bool get isReady;
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
│   ├── stt_result.dart                    # SttResult (text, inferenceTimeMs, lang)
│   ├── stt_logger.dart                    # Structured logging
│   ├── stt_exception.dart                 # Custom exception types
│   ├── cancellation_token.dart            # CancellationToken
│   ├── compute_worker.dart                # ComputeWorker (bg isolate for resample)
│   ├── model_registry.dart                # ModelRegistry, ModelDescriptor
│   ├── model_downloader.dart              # HTTP download + progress + tar.bz2 extract
│   ├── stt/
│   │   └── stt_engine.dart                # SttEngine (singleton, initBindings)
│   ├── audio/
│   │   ├── audio_buffer.dart              # AudioBuffer data class
│   │   ├── audio_processor.dart           # Resample, normalize, WAV parse
│   │   ├── audio_capture.dart             # Streaming audio capture (record package)
│   │   └── vad.dart                       # SherpaOnnxVadEngine wrapper
│   ├── engines/
│   │   ├── inference_engine.dart          # Abstract InferenceEngine
│   │   ├── engine_factory.dart            # SttModelType → InferenceEngine
│   │   ├── whisper/
│   │   │   └── whisper_engine.dart        # OfflineRecognizer, modelType: 'whisper'
│   │   ├── sherpa/
│   │   │   └── sherpa_engine.dart         # OfflineRecognizer, modelType: 'zipformer2'
│   │   ├── canary/
│   │   │   └── canary_engine.dart         # OfflineRecognizer, modelType: 'canary'
│   │   └── nemo/
│   │       └── nemo_engine.dart           # OfflineRecognizer, modelType: 'nemo_transducer'
│   └── default_models/
│       ├── whisper_models.dart            # All 10 Whisper variants (FP32 HF)
│       ├── sherpa_models.dart             # Zipformer EN (tar.bz2) + Parakeet TDT (HF)
│       └── canary_models.dart             # Canary 180M Flash (HF)
test/
├── stt_flutter_test.dart
├── model_registry_test.dart
├── mel_spectrogram_test.dart
├── audio_processor_test.dart
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
|---|------|-------|-------------|
| 1 | Project scaffold | `pubspec.yaml`, `analysis_options.yaml`, `lib/stt_flutter.dart` | `flutter pub get` |
| 2 | Model descriptors | `model_registry.dart`, `whisper_models.dart`, `sherpa_models.dart` | `flutter test` (registry unit tests) |
| 3 | Model downloader | `model_downloader.dart` | Unit test with mock HTTP |
| 4 | Audio processing | `audio_buffer.dart`, `audio_processor.dart` | Unit test with known WAV files |
| 5 | Sherpa engine | `engines/sherpa/sherpa_engine.dart` | Integration test: download zipformer, transcribe EN |
| 6 | Whisper engine | `engines/whisper/whisper_engine.dart` | Integration test: download tiny model, transcribe |
| 7 | Engine factory wiring | `engine_factory.dart`, `stt_flutter_impl.dart`, `stt_flutter.dart` exports | All tests pass |
| 8 | Singleton SttEngine | `stt/stt_engine.dart` | Works end-to-end |
| 9 | Streaming + VAD | `audio/audio_capture.dart`, `audio/vad.dart` | Real-time recording test |
| 10 | Example app | `main.dart`, `model_selection_screen.dart`, `transcription_screen.dart` | `flutter run` on device |

---

## Test Strategy

| Test | Type | Verifies |
|------|------|----------|
| `audio_processor_test.dart` | Unit | WAV parsing, resample to 16kHz, PCM normalization |
| `model_registry_test.dart` | Unit | Register, lookup, available, duplicates |
| `stt_flutter_test.dart` | Unit | Registry, model descriptor validation |
