import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../stt_logger.dart';

/// Spoken-language identifier built on top of `sherpa_onnx.SpokenLanguageIdentification`.
///
/// Reuses a Whisper-tiny encoder/decoder pair to identify the language of a 16 kHz
/// mono PCM clip. The result is used as a fallback for engines whose native
/// recognizer does not return a language (Parakeet, Zipformer, etc.).
class LanguageDetector {
  SpokenLanguageIdentification? _sli;
  String? _loadedEncoder;
  String? _loadedDecoder;

  /// Create a detector that will lazily load [encoderPath] / [decoderPath] on
  /// the first call to [detect]. Both paths must point at the files of a
  /// Whisper-tiny model (e.g. `tiny-encoder.onnx` / `tiny-decoder.onnx`).
  LanguageDetector();

  bool get isReady => _sli != null;

  Future<void> _ensureLoaded({
    required String encoderPath,
    required String decoderPath,
  }) async {
    if (_sli != null && _loadedEncoder == encoderPath && _loadedDecoder == decoderPath) {
      return;
    }
    _sli?.free();
    _loadedEncoder = encoderPath;
    _loadedDecoder = decoderPath;
    _sli = SpokenLanguageIdentification(
      SpokenLanguageIdentificationConfig(
        whisper: SpokenLanguageIdentificationWhisperConfig(
          encoder: encoderPath,
          decoder: decoderPath,
        ),
        numThreads: 1,
        provider: 'cpu',
        debug: false,
      ),
    );
    SttLogger.d('LanguageDetector: loaded SLI model');
  }

  /// Detect the language of [samples] (16 kHz mono Float32).
  /// Returns an empty string if detection fails.
  Future<String> detect(
    Float32List samples, {
    required int sampleRate,
    required String encoderPath,
    required String decoderPath,
  }) async {
    if (samples.isEmpty) return '';
    await _ensureLoaded(encoderPath: encoderPath, decoderPath: decoderPath);
    final sli = _sli;
    if (sli == null) return '';

    final stream = sli.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: sampleRate);
      final result = sli.compute(stream);
      return result.lang;
    } finally {
      stream.free();
    }
  }

  Future<void> dispose() async {
    _sli?.free();
    _sli = null;
    _loadedEncoder = null;
    _loadedDecoder = null;
  }
}
