# stt_flutter

Fully local, on-device speech-to-text for Flutter using `sherpa_onnx`.

Runs inference on the **main isolate** (native `sherpa_onnx` FFI calls are non-blocking).
Supports seven model families — all ONNX, all via `sherpa_onnx`.

[![CI](https://github.com/sebbom/stt_flutter_package/actions/workflows/ci.yaml/badge.svg)](https://github.com/sebbom/stt_flutter_package/actions/workflows/ci.yaml)

---

## Features

- **Local only** — no network calls during transcription
- **Multi-language** — every language supported by the loaded model is available
  (99+ via Whisper, 25 via Parakeet, en/es/de/fr via Canary, en via Zipformer,
  zh/en/ja/ko/yue via SenseVoice, 1600 via Omnilingual, multilingual via Qwen3)
- **7 model families** — Whisper, Sherpa-ONNX (Zipformer), NeMo Parakeet, Canary, SenseVoice, Omnilingual ASR, Qwen3-ASR
- **Three language modes** — auto-detect, default from `loadModel`, forced per-call
- **Language detection** — detected language returned in `SttResult.lang` (Whisper, Canary, Parakeet, plus a Whisper-tiny SLI fallback)
- **Long-form audio** — all engines chunk audio at 30 s with overlap and dedup
- **Beam search toggle** — `beamSearch: true` on `transcribeFile`/`transcribeBuffer` (Nemo falls back to greedy with warning)
- **Audio preprocessing** — denoiser (GTCRN/DPDFNet), high-pass, gain, normalize
- **Runtime download** — models downloaded and cached on first use
- **Extensible registry** — add any ONNX model in one line of code
- **Native ONNX Runtime** — via `sherpa_onnx` (no `flutter_onnxruntime`)
- **Silero VAD support** — optional `SherpaOnnxVadEngine` wrapper for speech/noise gating
- **Hotwords** — boost accuracy for specific words (Zipformer file-based, Qwen3 comma-separated)
---

## Requirements

- Flutter **3.44+** (Dart 3.12+) — recommended. Minimum: Flutter 3.7+ / Dart 3.4+
- Android minSdk 24, iOS 14+
- Linux: `clang`, `cmake`, `ninja-build`, `libgtk-3-dev`, `liblzma-dev`, `libpulse-dev`

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
| SenseVoice | ✅ | from `ModelDescriptor.languages` (zh, en, ja, ko, yue) |
| Omnilingual | ✅ | from `ModelDescriptor.languages` (1600) |
| Qwen3-ASR | ✅ | from `ModelDescriptor.languages` (multilingual) |

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
| `sensevoice-small` | SenseVoice | zh, en, ja, ko, yue | ~250 MB |
| `omnilingual-300m-ctc-v2` | Omnilingual | 1600 langs | ~1.3 GB |
| `omnilingual-1b-ctc` | Omnilingual | 1600 langs | ~3.9 GB |
| `qwen3-asr-0.6b` | Qwen3-ASR | multilingual | ~1 GB |

### Choosing a Model

| Use Case | Recommended Model | Size | Languages | Notes |
|----------|------------------|------|-----------|-------|
| Fast, English-only | `sherpa-zipformer-en` | 300 MB | en | Best for English-only, fast inference |
| General multilingual | `whisper-tiny` | 150 MB | 99+ | Good balance of size and accuracy |
| Better accuracy | `whisper-small` | 460 MB | 99+ | Higher quality than tiny |
| Best accuracy | `whisper-medium` | 960 MB | 99+ | Best quality/size tradeoff |
| High accuracy, fast | `whisper-large-v3-turbo` | 550 MB | 99+ | Optimized for speed |
| European languages | `canary-180m-en-es-de-fr` | 200 MB | en, es, de, fr | Optimized for these 4 languages |
| Asian languages | `sensevoice-small` | 250 MB | zh, en, ja, ko, yue | Optimized for Asian languages |
| 1600+ languages | `omnilingual-300m-ctc-v2` | 1.3 GB | 1600 | Supports the most languages |
| Multilingual, modern | `qwen3-asr-0.6b` | 1 GB | multilingual | State-of-the-art model |

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

## Audio preprocessing

All preprocessing is optional and controlled via `PreprocessConfig`. When none is
set (`PreprocessConfig.none`), audio is passed to the engine as-is.

### Pipeline order

Denoiser runs first to remove noise before gain/normalize amplify it:

```
raw audio → denoise → high-pass → gain → normalize → engine
```

### Available options

| Option | What it does | Default |
|--------|-------------|---------|
| **Denoiser** (GTCRN / DPDFNet) | Neural speech enhancement via sherpa-onnx `OfflineSpeechDenoiser`. Removes background noise, fan hum, keyboard clatter. GTCRN is lighter (535 KB); DPDFNet is heavier (10 MB) but higher quality. | Off |
| **High-pass filter** | First-order IIR RC filter at 80 Hz cutoff. Removes DC offset and low-frequency rumble (HVAC, mic handling). | Off |
| **Gain** | Multiplies every sample by a factor (e.g. `1.5`). Useful for quiet recordings. Clamps to [-1, 1] to prevent clipping. | 1.0 (no change) |
| **Normalize — Peak** | Scales audio so `max|sample|` equals 0.95. Preserves dynamics while preventing clipping. No-op on silent or already-loud audio. | Off |
| **Normalize — RMS** | Scales audio so the root-mean-square equals 0.1. Loudness-normalized output for consistent volume across recordings. Clamps to [-1, 1]. | Off |
| **Noise suppression flag** | Surfaces a `noiseSuppression` boolean for platform plugins (e.g. `noise_suppression`). The library only stores the flag; actual suppression is applied by your platform code. | Off |

### Quick-start example

```dart
import 'package:stt_flutter/stt_flutter.dart';

final result = await SttEngine.instance.transcribeFile(
  path,
  preprocess: PreprocessConfig(
    denoiserType: DenoiserType.gtcrn,
    denoiserModelDir: '/path/to/gtcrn/', // or use DenoiserBundle in the example app
    highPass: true,
    gain: 1.3,
    normalize: NormalizeMode.peak,
    noiseSuppression: true,
  ),
);
```

### Hotwords

Boost recognition accuracy for domain-specific words:

| Engine | Mechanism | How to set |
|--------|-----------|-----------|
| **Zipformer** | `hotwordsFile` (one word per line) + `hotwordsScore` (boost factor, e.g. `1.5`) | `SttEngine.setHotwords(text)` writes to `<modelDir>/hotwords.txt` and reloads |
| **Qwen3-ASR** | `hotwords` (comma-separated string) | `loadModel(hotwords: 'hello,world')` or per-call via `PreprocessConfig` |

---

## Architecture

```
Main isolate
  SttEngine (singleton)
    └─ SttFlutter
         ├─ WhisperInferenceEngine   (OfflineRecognizer, modelType: 'whisper')
         ├─ SherpaInferenceEngine    (OfflineRecognizer, modelType: 'zipformer2')
         ├─ NemoInferenceEngine      (OfflineRecognizer, modelType: 'nemo_transducer')
         ├─ CanaryInferenceEngine    (OfflineRecognizer, modelType: 'canary')
         ├─ SenseVoiceInferenceEngine(OfflineRecognizer, modelType: 'sense_voice')
         ├─ OmnilingualInferenceEngine(OfflineRecognizer, modelType: 'omnilingual_asr_ctc')
         └─ Qwen3AsrInferenceEngine  (OfflineRecognizer, modelType: 'qwen3_asr')

  AudioCaptureService → Float32List stream → VAD → SttEngine.transcribeBuffer
  AudioProcessor → denoiser (GTCRN/DPDFNet) → high-pass → gain → normalize
  LanguageDetector (optional, sherpa_onnx SpokenLanguageIdentification) →
    used as fallback when the engine doesn't return a lang
```

All inference runs via `sherpa_onnx` native FFI. Audio preprocessing (resampling)
runs in one-shot `Isolate.run()` background isolates. `sherpa_onnx.initBindings()`
is called once by `SttEngine` on first model load.

See [PLAN.md](PLAN.md) for the full architecture document.

---

## Language auto-detect

The library has three language modes passed through `transcribeFile` / `transcribeBuffer`:

| Mode | How |
|---|---|
| **Auto** | Pass `language: null` (or omit). Whisper auto-detects; other engines use their native tokenizer. |
| **Default** | Set on `SttEngine.loadModel(defaultLanguage: 'de')`. Used when no per-call override. |
| **Forced** | Pass `language: 'fr'` at call-site. Always wins over default. |

`SttResult.lang` reports what the engine actually produced. An optional `LanguageDetector` fallback (Whisper-tiny SLI) populates `lang` when the engine returns empty.

---

## License

MIT
