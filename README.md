# stt_flutter

Fully local, on-device speech-to-text for Flutter using `sherpa_onnx`.

Runs inference on the **main isolate** (native `sherpa_onnx` FFI calls are non-blocking).
Supports four model families — all ONNX, all via `sherpa_onnx`.

---

## Features

- **Local only** — no network calls during transcription
- **Multi-language** — every language supported by the loaded model is available
  (99+ via Whisper, 25 via Parakeet, en/es/de/fr via Canary, en via Zipformer)
- **4 model families** — Whisper, Sherpa-ONNX (Zipformer transducer), NeMo Parakeet, Canary
- **Three language modes** — auto-detect, default from `loadModel`, forced per-call
- **Language detection** — detected language returned in `SttResult.lang` (Whisper, Canary, Parakeet, plus a Whisper-tiny SLI fallback)
- **Long-form audio** — Whisper chunks audio at 30 s with overlap and dedup
- **Runtime download** — models downloaded and cached on first use
- **Extensible registry** — add any ONNX model in one line of code
- **Native ONNX Runtime** — via `sherpa_onnx` (no `flutter_onnxruntime`)
- **Silero VAD support** — optional `SherpaOnnxVadEngine` wrapper for speech/noise gating

---

## Quick start

```dart
import 'package:stt_flutter/stt_flutter.dart';

// 1. Pick a model
final model = ModelRegistry.get('whisper-tiny');

// 2. Download it (once)
await ModelDownloader.download(model);

// 3. Initialize engine — pass an optional default language.
final stt = SttFlutter();
await stt.initialize(model: model, language: 'de');

// 4. Transcribe
final result = await stt.transcribeFile('/path/to/audio.wav');
print(result.text); // German text
print(result.lang); // "de"
await stt.dispose();
```

Or use the singleton `SttEngine`:

```dart
await SttEngine.instance.loadModel(model, defaultLanguage: 'de');
final result = await SttEngine.instance.transcribeFile('/path/to/audio.wav');
```

---

## Three language modes

The language is **never hardcoded** at the call site. Pick one of three modes:

```dart
// 1. Auto-detect — leave the language as null everywhere. The engine
//    decides (Whisper auto-detects, Parakeet uses the model's default
//    multilingual tokenizer, etc).
final r1 = await SttFlutter().transcribeFile(path);

// 2. Default — set once at loadModel. Used when no per-call override.
await SttEngine.instance.loadModel(model, defaultLanguage: 'de');
final r2 = await SttEngine.instance.transcribeFile(path);

// 3. Forced per-call — always wins over the default. Engine returns the
//    detected language in result.lang so you can verify what it produced.
final r3 = await SttEngine.instance.transcribeFile(path, language: 'fr');
print(r3.lang); // "fr" (or whatever the engine actually detected)
```

| Engine | `supportsExplicitLanguage` | Languages |
|---|---|---|
| Whisper | ✅ | from `ModelDescriptor.languages` (99+) |
| Sherpa (zipformer) | ❌ (English only) | `en` |
| Nemo (Parakeet) | ✅ | from `ModelDescriptor.languages` (25) |
| Canary | ✅ | from `ModelDescriptor.languages` (en, es, de, fr) |

When a forced language is **not in** the model's supported list, the engine
logs a warning and continues — useful for catching bugs without breaking the
transcription in production. For models that don't support explicit language
(Sherpa zipformer), the warning is the signal that the model is monolingual.

---

## Seeded models

| ID | Type | Languages | Size |
|----|------|-----------|------|
| `whisper-tiny` | Whisper | 13 langs declared, 99+ actually supported | ~150 MB |
| `whisper-tiny.en` | Whisper | en | ~150 MB |
| `whisper-base` | Whisper | 13 declared, 99+ supported | ~240 MB |
| `whisper-base.en` | Whisper | en | ~240 MB |
| `whisper-small` | Whisper | 13 declared, 99+ supported | ~460 MB |
| `whisper-small.en` | Whisper | en | ~460 MB |
| `whisper-medium` | Whisper | 13 declared, 99+ supported | ~960 MB |
| `whisper-medium.en` | Whisper | en | ~960 MB |
| `whisper-large-v3` | Whisper | 13 declared, 99+ supported | ~950 MB |
| `whisper-large-v3-turbo` | Whisper | 13 declared, 99+ supported | ~550 MB |
| `sherpa-zipformer-en` | Sherpa | en | ~300 MB |
| `parakeet-tdt-0.6b-multilingual` | NeMo Parakeet | 25 langs | ~400 MB |
| `canary-180m-en-es-de-fr` | Canary | en, es, de, fr | ~200 MB |

Add your own model:

```dart
ModelRegistry.register(ModelDescriptor(
  id: 'my-custom-model',
  type: SttModelType.whisper,
  languages: ['ja', 'ko', 'zh'],
  files: [
    ModelFile(url: '...', filename: 'encoder.onnx'),
    ModelFile(url: '...', filename: 'decoder.onnx'),
  ],
  sizeMb: 220,
));
```

---

## Architecture

```
Main isolate
  SttEngine (singleton)
    └─ SttFlutter
         ├─ WhisperInferenceEngine   (OfflineRecognizer, modelType: 'whisper')
         ├─ SherpaInferenceEngine    (OfflineRecognizer, modelType: 'zipformer2')
         ├─ NemoInferenceEngine      (OfflineRecognizer, modelType: 'nemo_transducer')
         └─ CanaryInferenceEngine    (OfflineRecognizer, modelType: 'canary')

  AudioCaptureService → Float32List stream → VAD → SttEngine.transcribeBuffer
  LanguageDetector (optional, sherpa_onnx SpokenLanguageIdentification) →
    used as fallback when the engine doesn't return a lang
```

All inference runs via `sherpa_onnx` native FFI. Audio preprocessing (resampling)
runs in one-shot `Isolate.run()` background isolates. `sherpa_onnx.initBindings()`
is called once by `SttEngine` on first model load.

See [PLAN.md](PLAN.md) for the full architecture document.

---

## Requirements

- Flutter 3.7+
- Dart 3.0+
- Android minSdk 24, iOS 14+

---

## License

MIT
