# `stt_flutter` — Architecture & Implementation Plan

Fully local, on-device speech-to-text for Flutter using ONNX models via `flutter_onnxruntime`.

---

## Threading Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Main Isolate (UI)                           │
│                                                                   │
│  SttFlutter                                                       │
│    ├─ manages OnnxRuntime + OrtSession(s)                        │
│    ├─ all session.run() calls are async (MethodChannel → native)  │
│    │                                                              │
│    ├─ transcribeFile(path)                                        │
│    │   ├─ AudioProcessor.loadWav()    (async I/O, main isolate)   │
│    │   ├─ Isolate.run(resampleSync)   (ephemeral bg isolate)      │
│    │   └─ engine.transcribe(audio)                                │
│    │       ├─ Isolate.run(mel/fbank)  (ephemeral bg isolate)      │
│    │       └─ session.run() × N      (async MethodChannel)        │
│    │                                                              │
│    └─ dispose() → sessions.close()                               │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  Engine Layer (main isolate)                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │   │
│  │  │ Whisper  │  │  Sherpa  │  │ Voxtral  │               │   │
│  │  └──────────┘  └──────────┘  └──────────┘               │   │
│  └───────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │  Isolate.spawn / Isolate.run
                         │  (ephemeral, one per preprocess call)
┌────────────────────────▼────────────────────────────────────────┐
│              Ephemeral Background Isolates                        │
│  (short-lived, terminated after each preprocessing task)          │
│    ┌────────────────┐   ┌──────────────────┐                     │
│    │ resampleSync() │   │ MelSpectrogram   │                     │
│    │                │   │ .compute()        │                     │
│    └────────────────┘   └──────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

**Why not a long-lived background isolate:**
`flutter_onnxruntime` v1.7.1 uses `MethodChannel` internally. In Flutter 3.44,
`BackgroundIsolateBinaryMessenger` is library-private (under `_`) and not
exported from any barrel file, making it inaccessible. However, all
`session.run()` calls are async via `MethodChannel` — they do **not** block
the Dart event loop. The only CPU-bound work is audio preprocessing (WAV
parsing, resampling, mel/Fbank extraction), which is offloaded to ephemeral
`Isolate.run()` calls. Model inference itself runs asynchronously on the main
isolate without blocking.

**Ephemeral isolates:** Audio preprocessing functions are pure functions
(stateless, no native resources) — perfect for `Isolate.run()` (Dart 2.19+).
Each call spawns a short-lived isolate, processes the data, returns the result,
and terminates. This avoids the `BackgroundIsolateBinaryMessenger`
compatibility issue while keeping the UI thread free during computation.

---

## Model Registry — Extensible by Design

Users register any ONNX model in one line. The package ships with seeded models.

```dart
// --- Core types ---
enum SttModelType { whisper, sherpa, voxtral }

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
| `voxtral-mini` | Voxtral | en,de,fr,es,pt,hi,nl,it | HF ONNX | ~2.7 GB |

Each `ModelDescriptor` encodes the exact file list and URLs. Sherpa models are
downloaded as `.tar.bz2` and extracted via `package:archive`.

---

## Model Download System

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

- Uses the `http` package (streaming downloads with progress)
- Sherpa: downloads `.tar.bz2` → extracts to model directory
- Whisper / Voxtral: downloads individual ONNX files from HuggingFace
- Progress reported via `SendPort` to main isolate for UI updates

---

## Engine Interface (runs on background isolate)

```dart
abstract class InferenceEngine {
  /// Load model files. [modelFiles] maps logical names to absolute paths.
  Future<void> load(Map<String, String> modelFiles);

  /// Transcribe audio and return text.
  Future<SttResult> transcribe(AudioBuffer audio, {String? language});

  /// Release all native resources (sessions, tensors).
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
}
```

---

## Whisper Engine

**Files needed:** `encoder.onnx`, `decoder.onnx`, `tokenizer.json`

```
WAV file ──► resample to 16kHz mono ──► log-mel spectrogram (80 bins)
     ──► encoder ONNX ──► audio embeddings
     ──► build decoder prompt [sot, lang, transcribe, notimestamps]
     ──► decoder autoregressive loop with KV cache ──► token IDs
     ──► BPE tokenizer.decode(text) ──► text
```

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `mel_spectrogram.dart` | `MelSpectrogram` | Hann STFT (400/160), 80 mel filterbanks, log10 normalization, output `[1,80,3000]` |
| `bpe_tokenizer.dart` | `BpeTokenizer` | Load `tiktoken`-format vocab, encode/decode BPE, handle special tokens |
| `whisper_decoder.dart` | `WhisperDecoder` | Build SOT prompt, run encoder once, autoregressive loop with KV cache, EOT detection |
| `whisper_engine.dart` | `WhisperInferenceEngine` | Orchestrates mel → enc → dec → tokenizer → text |

**KV cache management:** The decoder expects `past_self_key_0..N`,
`past_self_value_0..N`, `cross_key_0..N`, `cross_value_0..N` tensors. First call
has empty caches (zero-length). Each iteration updates them. All managed as
`OrtValue` objects in the background isolate.

**Language support:** SOT prompt includes language token (e.g. `<|de|>` for
German). The multilingual BPE tokenizer handles all 99 Whisper languages.

---

## Sherpa Transducer Engine

**Files needed:** `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`

```
WAV file ──► resample to 16kHz mono ──► Fbank features (80-dim, 25ms, 10ms)
     ──► encoder ONNX ──► acoustic embeddings
     ──► transducer greedy search:
         for each frame t:
           h = encoder[t]
           for each step:
             logits = joiner(h, decoder(prev_token))
             if argmax(logits) == blank: break
             else: emit token, set prev_token = token
     ──► lookup tokens.txt ──► text
```

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `fbank_extractor.dart` | `FbankExtractor` | 25ms Hamming window, 80 mel filterbanks, CMVN normalization |
| `transducer_decoder.dart` | `TransducerDecoder` | Greedy search: for each frame, iterate joiner(encoder[h], decoder[prev]) until blank |
| `sherpa_engine.dart` | `SherpaInferenceEngine` | Orchestrates fbank → enc → joiner-decoder → tokens.txt → text |

---

## Voxtral Engine

**Files needed:** `audio_encoder.onnx`, `decoder_model_merged.onnx`,
`embed_tokens.onnx`, `tokenizer.json`

```
WAV file ──► resample to 16kHz mono ──► log-mel spectrogram (128 bins)
     ──► audio_encoder ONNX ──► audio embeddings
     ──► embed_tokens ONNX ──► text embeddings for prompt template
     ──► combine embeddings ──► inputs_embeds
     ──► decoder_model_merged autoregressive loop with KV cache
     ──► Tekken tokenizer.decode(text) ──► text
```

| Sub-component | File | Responsibility |
|---------------|------|----------------|
| `tekken_tokenizer.dart` | `TekkenTokenizer` | Load `tokenizer.json` (HF format), encode/decode Tekken BPE |
| `voxtral_decoder.dart` | `VoxtralDecoder` | Build prompt with `[INST] lang:xx [TRANSCRIBE]`, combine audio + text embeddings, autoregressive LLM loop |
| `voxtral_engine.dart` | `VoxtralInferenceEngine` | Orchestrates mel → audio_enc → embed → decoder → tokenizer → text |

**Prompt template:** `<s>[INST]<audio_placeholder><text_instruction>[/INST]`
where `<audio_placeholder>` = projected audio embeddings and
`<text_instruction>` = e.g. `lang:de [TRANSCRIBE]`.

---

## Public API

```dart
/// Main entry point. Runs on the main isolate, delegates to background worker.
class SttFlutter {
  /// Initialize: spawns background isolate, loads ONNX sessions.
  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,   // defaults to {appDocDir}/stt_models/{model.id}
    String? language,   // default language (e.g. 'de')
  });

  /// Transcribe a WAV file at [path].
  Future<SttResult> transcribeFile(String path);

  /// Transcribe raw PCM [samples] (Float32, [-1.0, 1.0]) at [sampleRate] Hz.
  Future<SttResult> transcribeBuffer(Float32List samples, int sampleRate);

  /// Release all resources and kill the background isolate.
  Future<void> dispose();
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
│   ├── stt_result.dart                    # SttResult
│   ├── isolate_worker.dart                # InferenceWorker (bg isolate)
│   ├── model_registry.dart                # ModelRegistry, ModelDescriptor
│   ├── model_downloader.dart              # HTTP download + progress
│   ├── audio/
│   │   ├── audio_buffer.dart              # AudioBuffer data class
│   │   └── audio_processor.dart           # Resample, normalize, WAV parse
│   ├── engines/
│   │   ├── inference_engine.dart          # Abstract InferenceEngine
│   │   ├── engine_factory.dart            # SttModelType → InferenceEngine
│   │   ├── whisper/
│   │   │   ├── whisper_engine.dart
│   │   │   ├── mel_spectrogram.dart
│   │   │   ├── whisper_decoder.dart
│   │   │   └── bpe_tokenizer.dart
│   │   ├── sherpa/
│   │   │   ├── sherpa_engine.dart
│   │   │   ├── fbank_extractor.dart
│   │   │   └── transducer_decoder.dart
│   │   └── voxtral/
│   │       ├── voxtral_engine.dart
│   │       ├── voxtral_decoder.dart
│   │       └── tekken_tokenizer.dart
│   └── default_models/
│       ├── whisper_models.dart            # All 10 Whisper variants
│       ├── sherpa_models.dart             # Zipformer EN
│       └── voxtral_models.dart            # Voxtral Mini q4f16
test/
├── units/
│   ├── audio_processor_test.dart
│   ├── bpe_tokenizer_test.dart
│   ├── tekken_tokenizer_test.dart
│   ├── mel_spectrogram_test.dart
│   ├── fbank_extractor_test.dart
│   ├── transducer_decoder_test.dart
│   ├── whisper_decoder_test.dart
│   └── model_registry_test.dart
├── engines/
│   ├── whisper_engine_test.dart
│   └── sherpa_engine_test.dart
└── fixtures/
    ├── hello_en.wav
    ├── guten_tag_de.wav
    ├── bonjour_fr.wav
    └── hola_es.wav
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
  flutter_onnxruntime: ^1.7.1   # ONNX Runtime inference
  http: ^1.2.0                  # Model downloads
  path_provider: ^2.1.0         # Model storage path
  archive: ^4.0.0               # .tar.bz2 extraction (Sherpa models)
  file: ^7.0.0                  # File utilities

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
| 2 | Model descriptors | `model_registry.dart`, `whisper_models.dart`, `sherpa_models.dart`, `voxtral_models.dart` | `flutter test` (registry unit tests) |
| 3 | Model downloader | `model_downloader.dart` | Unit test with mock HTTP |
| 4 | Audio processing | `audio_buffer.dart`, `audio_processor.dart` | Unit test with known WAV files |
| 5 | Background isolate | `isolate_worker.dart`, `stt_flutter_impl.dart` | Integration test (spawn + hello) |
| 6 | Whisper mel + tokenizer | `mel_spectrogram.dart`, `bpe_tokenizer.dart` | Unit tests against known values |
| 7 | Whisper decoder + engine | `whisper_decoder.dart`, `whisper_engine.dart` | Integration test: download tiny model, transcribe 4 languages |
| 8 | Sherpa fbank + decoder | `fbank_extractor.dart`, `transducer_decoder.dart` | Unit tests |
| 9 | Sherpa engine | `sherpa_engine.dart` | Integration test: download zipformer, transcribe EN |
| 10 | Voxtral tokenizer + decoder | `tekken_tokenizer.dart`, `voxtral_decoder.dart` | Unit tests |
| 11 | Voxtral engine | `voxtral_engine.dart` | Integration test (optional, very large model) |
| 12 | Engine factory wiring | `engine_factory.dart`, `stt_flutter.dart` exports | All tests pass |
| 13 | Example app | `main.dart`, `model_selection_screen.dart`, `transcription_screen.dart` | `flutter run` on device |
| 14 | Test fixtures WAVs | 4 WAV files in `test/fixtures/` | Generated with `ffmpeg` TTS |

---

## Test Strategy

| Test | Type | Verifies |
|------|------|----------|
| `audio_processor_test.dart` | Unit | WAV parsing, resample to 16kHz, PCM normalization |
| `mel_spectrogram_test.dart` | Unit | Shape `[1,80,3000]`, values in expected range |
| `fbank_extractor_test.dart` | Unit | Shape matches, compares against reference impl |
| `bpe_tokenizer_test.dart` | Unit | Encode/decode roundtrip, special tokens |
| `tekken_tokenizer_test.dart` | Unit | Encode/decode roundtrip with Voxtral vocab |
| `transducer_decoder_test.dart` | Unit | Greedy search on dummy logits, blank handling |
| `whisper_decoder_test.dart` | Unit | Autoregressive loop on dummy logits, EOT detection |
| `model_registry_test.dart` | Unit | Register, lookup, available, duplicates |
| `whisper_engine_test.dart` | Integration | Download tiny model, transcribe DE/EN/FR/ES fixtures, verify text contains expected words |
| `sherpa_engine_test.dart` | Integration | Download zipformer, transcribe EN fixture |

Integration tests use `setUpAll` to download the smallest model variant once,
then run multiple transcriptions. Tests skip gracefully if network is
unavailable.
