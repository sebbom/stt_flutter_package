# CHANGELOG

## 0.2.1 (unreleased)
- Fix: resample audio to 16kHz before passing to engines (fixes truncated transcriptions from 8kHz files)
- Fix: boundary-safe `resampleSync` with `clamp()` to prevent index out-of-bounds
- Feat: `beamSearch` parameter on `transcribeFile`/`transcribeBuffer` (default `false`; Nemo transducer falls back to greedy)
- Feat: `Map<String, dynamic>? options` on `InferenceEngine.transcribe()` for engine-specific flags
- Feat: MIT `LICENSE` file
- CI: upgrade to Flutter 3.44.x / Dart 3.12+
- Chore: lint fixes across tests and example

## 0.2.0
- Initial release
- Support for 7 model families: Whisper, Sherpa Zipformer, NeMo Parakeet, Canary, SenseVoice, Omnilingual, Qwen3-ASR
- 99+ languages supported
- Audio preprocessing pipeline (denoiser, high-pass, gain, normalize)
- Long-form audio chunking with deduplication
- Hotwords support for Zipformer and Qwen3-ASR
- Silero VAD support for streaming
- Runtime model download and caching
- Extensible model registry

## 0.1.0
- Initial development version
