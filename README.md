# stt_flutter

Fully local, on-device speech-to-text for Flutter using `sherpa_onnx`.

Runs inference on the **main isolate** (native `sherpa_onnx` FFI calls are non-blocking).
Supports four model families — all ONNX, all via `sherpa_onnx`.

---

## Features

- **Local only** — no network calls during transcription
- **Multi-language** — German, English, French, Spanish (and 99+ more via Whisper)
- **4 model families** — Whisper, Sherpa-ONNX (Zipformer transducer), NeMo Parakeet, Canary
- **Language detection** — detected language returned in `SttResult.lang` (Whisper, Canary, Parakeet)
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

// 3. Initialize engine
final stt = SttFlutter();
await stt.initialize(model: model, language: 'de');

// 4. Transcribe
final result = await stt.transcribeFile('/path/to/audio.wav');
print(result.text); // "Guten Tag, wie geht es Ihnen?"

// 5. Cleanup
await stt.dispose();
```

Or use the singleton `SttEngine` for repeated use:

```dart
await SttEngine.instance.loadModel(model);
final result = await SttEngine.instance.transcribeFile('/path/to/audio.wav');
```

---

## Seeded models

| ID | Type | Languages | Size |
|----|------|-----------|------|
| `whisper-tiny` | Whisper | 99 langs | ~220 MB |
| `whisper-tiny.en` | Whisper | en | ~220 MB |
| `whisper-base` | Whisper | 99 langs | ~370 MB |
| `whisper-small` | Whisper | 99 langs | ~1.1 GB |
| `whisper-medium` | Whisper | 99 langs | ~2.5 GB |
| `whisper-large-v3` | Whisper | 99 langs | ~4.5 GB |
| `whisper-large-v3-turbo` | Whisper | 99 langs | ~2.5 GB |
| `sherpa-zipformer-en` | Sherpa | en | ~35 MB |
| `parakeet-tdt-0.6b-multilingual` | NeMo Parakeet | 25 langs | ~400 MB |
| `canary-180m-flash` | Canary | en | ~180 MB |

Add your own model:

```dart
ModelRegistry.register(ModelDescriptor(
  id: 'my-custom-model',
  type: SttModelType.whisper,
  languages: ['ja'],
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
```

All inference runs via `sherpa_onnx` native FFI. Audio preprocessing (resampling)
runs in ephemeral `Isolate.run()` background isolates. `sherpa_onnx.initBindings()`
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
