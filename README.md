# stt_flutter

Fully local, on-device speech-to-text for Flutter using ONNX models.

Runs inference on a **background isolate** so the UI thread is never blocked.
Supports three model families — all ONNX, all via `flutter_onnxruntime`.

---

## Features

- **Local only** — no network calls during transcription
- **Multi-language** — German, English, French, Spanish (and 99+ more via Whisper)
- **3 model families** — Whisper, Sherpa-ONNX, Voxtral (Mistral)
- **Runtime download** — models downloaded and cached on first use
- **Extensible registry** — add any ONNX model in one line of code
- **Background isolate** — all audio preprocessing and ONNX inference off the UI thread

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
| `voxtral-mini` | Voxtral | 8 langs | ~2.7 GB |

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
Main isolate                     Background isolate
  SttFlutter ──SendPort──►      InferenceWorker
                               ├─ WhisperEngine
                               ├─ SherpaEngine
                               └─ VoxtralEngine
```

See [PLAN.md](PLAN.md) for the full architecture document.

---

## Requirements

- Flutter 3.7+ (for background isolate channel support)
- Dart 3.0+

---

## License

MIT
