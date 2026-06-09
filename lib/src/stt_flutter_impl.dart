import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show visibleForTesting;
import 'stt_result.dart';
import 'stt_exception.dart';
import 'stt_logger.dart';
import 'cancellation_token.dart';
import 'model_registry.dart';
import 'model_downloader.dart';
import 'audio/audio_buffer.dart';
import 'audio/audio_processor.dart';
import 'engines/engine_factory.dart';
import 'engines/inference_engine.dart';
import 'language/language_detector.dart';

/// Main entry point for speech-to-text functionality.
///
/// This class provides the core API for initializing the STT engine,
/// transcribing audio files or buffers, and managing resources.
///
/// Example usage:
/// ```dart
/// final stt = SttFlutter();
/// await stt.initialize(model: ModelRegistry.get('whisper-tiny'));
/// final result = await stt.transcribeFile('/path/to/audio.wav');
/// print(result.text);
/// await stt.dispose();
/// ```
class SttFlutter {
  InferenceEngine? _engine;
  ModelDescriptor? _model;
  bool _initialized = false;
  String? _defaultLanguage;
  CancellationToken? _currentToken;
  LanguageDetector? _detector;
  String? _detectorEncoderPath;
  String? _detectorDecoderPath;

  SttFlutter();

  /// Test-only: create a [SttFlutter] with [engine] pre-loaded for [model].
  /// This bypasses the real model-load / file-discovery step.
  @visibleForTesting
  SttFlutter.withEngine({
    required ModelDescriptor model,
    required InferenceEngine engine,
    String? language,
  })  : _engine = engine,
        _model = model,
        _defaultLanguage =
            (language != null && language.isNotEmpty) ? language : null,
        _initialized = true;

  /// Test-only: inject a [LanguageDetector] (e.g. a fake) used to auto-detect
  /// the language in [_transcribe] when no per-call or default language is
  /// provided and the engine requires explicit language.
  @visibleForTesting
  set detector(LanguageDetector? value) {
    _detector = value;
    _detectorEncoderPath = 'fake-encoder.onnx';
    _detectorDecoderPath = 'fake-decoder.onnx';
  }

  /// Initializes the STT engine with the specified model.
  ///
  /// [model]: The model descriptor to load (e.g., from [ModelRegistry.get])
  /// [modelDir]: Optional directory containing pre-downloaded model files.
  ///            If not provided, uses the default cache directory.
  /// [language]: Optional default language for transcription (ISO 639-1 code).
  ///            If not provided, the engine will use auto-detection where supported.
  ///
  /// Throws [SttException] if initialization fails or if already initialized.
  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,
    String? language,
  }) async {
    if (_initialized) throw SttException.notInitialized('SttFlutter');
    if (model.id.isEmpty) throw SttException.invalidArgument('model.id must not be empty');
    _model = model;
    _defaultLanguage = (language != null && language.isNotEmpty) ? language : null;

    final dir = modelDir ?? await ModelDownloader.defaultStoragePath(model);
    final modelFiles = <String, String>{};

    for (final f in model.files) {
      final path = '$dir/${f.filename}';
      final file = File(path);
      if (await file.exists()) {
        modelFiles[f.filename] = path;
      }
    }

    final modelDir_ = Directory(dir);
    if (await modelDir_.exists()) {
      await for (final entry in modelDir_.list()) {
        if (entry is File) {
          final name = entry.uri.pathSegments.last;
          if (name.endsWith('.onnx') ||
              name.endsWith('.onnx_data') ||
              name.endsWith('.txt') ||
              name.endsWith('.json') ||
              name.endsWith('.weights')) {
            modelFiles[name] = entry.path;
          }
        }
      }
    }

    if (modelFiles.isEmpty) {
      throw SttException.modelLoadFailed('No model files found in $dir');
    }

    try {
      _engine = createEngine(model);
      await _engine!.load(modelFiles);
      _initialized = true;
      SttLogger.i(
        'Initialized with model: ${model.id} (langs=${model.languages.length})',
      );
    } catch (e) {
      await _cleanup();
      rethrow;
    }
  }

  /// Transcribes an audio file to text.
  ///
  /// [path]: Path to the audio file (WAV format recommended)
  /// [language]: Optional language override for this transcription (ISO 639-1 code).
  ///            If not provided, uses the default language set during initialization.
  /// [token]: Optional cancellation token to abort the transcription
  /// [preprocess]: Optional audio preprocessing configuration
  ///
  /// Returns: [SttResult] containing the transcribed text, detected language,
  ///          confidence score, and timing information
  ///
  /// Throws [SttException] if not initialized or if the file doesn't exist
  Future<SttResult> transcribeFile(
    String path, {
    String? language,
    CancellationToken? token,
    PreprocessConfig preprocess = PreprocessConfig.none,
  }) async {
    if (!_initialized) throw SttException.notInitialized('SttFlutter');
    final file = File(path);
    if (!await file.exists()) throw SttException.fileNotFound(path);
    final audio = await AudioProcessor.loadWav(path, preprocess: preprocess);
    if (audio.samples.isEmpty) {
      throw SttException.invalidArgument('Audio file contains no samples');
    }
    return _transcribe(audio, language: language, token: token);
  }

  /// Transcribes raw audio samples to text.
  ///
  /// [samples]: Raw PCM audio samples as Float32 values in range [-1.0, 1.0]
  /// [sampleRate]: Sample rate of the audio in Hz (will be resampled to 16kHz internally)
  /// [language]: Optional language override for this transcription (ISO 639-1 code).
  ///            If not provided, uses the default language set during initialization.
  /// [token]: Optional cancellation token to abort the transcription
  /// [preprocess]: Optional audio preprocessing configuration
  ///
  /// Returns: [SttResult] containing the transcribed text, detected language,
  ///          confidence score, and timing information
  ///
  /// Throws [SttException] if not initialized or if parameters are invalid
  Future<SttResult> transcribeBuffer(
    Float32List samples,
    int sampleRate, {
    String? language,
    CancellationToken? token,
    PreprocessConfig preprocess = PreprocessConfig.none,
  }) async {
    if (!_initialized) throw SttException.notInitialized('SttFlutter');
    if (sampleRate <= 0 || sampleRate > 192000) {
      throw SttException.invalidArgument('sampleRate must be between 1 and 192000');
    }
    if (samples.isEmpty) {
      throw SttException.invalidArgument('samples buffer must not be empty');
    }
    var audio = AudioBuffer(samples: samples, sampleRate: sampleRate);
    if (!preprocess.isNoOp) {
      audio = AudioProcessor.applyPreprocess(audio, preprocess);
    }
    return _transcribe(audio, language: language, token: token);
  }

  Future<SttResult> _transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
  }) async {
    _currentToken = token;
    token?.throwIfCancelled();

    // All engines expect 16kHz mono – resample before forwarding.
    if (audio.sampleRate != AudioProcessor.targetSampleRate) {
      audio = AudioProcessor.resampleSync(audio);
    }

    String? effective = language ?? _defaultLanguage;
    String? detectedLang;

    final needAutoDetect = (effective == null || effective.isEmpty) &&
        _engine!.supportsExplicitLanguage;

    if (needAutoDetect && _detector != null) {
      try {
        final detected = await _detector!.detect(
          audio.samples,
          sampleRate: audio.sampleRate,
          encoderPath: _detectorEncoderPath!,
          decoderPath: _detectorDecoderPath!,
        );
        if (detected.isNotEmpty) {
          detectedLang = detected;
          effective = detected;
          SttLogger.i('Auto-detected language: $effective');
        }
      } catch (e) {
        SttLogger.d('LanguageDetector auto-detect failed: $e');
      }
    }

    if (effective != null && effective.isNotEmpty) {
      _warnIfUnsupported(effective);
    }

    final result = await _engine!.transcribe(
      audio,
      language: effective,
      token: token,
    );

    String? lang = result.lang;
    if ((lang == null || lang.isEmpty) && detectedLang != null) {
      lang = detectedLang;
    } else if ((lang == null || lang.isEmpty) && _detector != null) {
      try {
        lang = await _detector!.detect(
          audio.samples,
          sampleRate: audio.sampleRate,
          encoderPath: _detectorEncoderPath!,
          decoderPath: _detectorDecoderPath!,
        );
      } catch (e) {
        SttLogger.d('LanguageDetector fallback failed: $e');
      }
    }

    SttLogger.d(
      'transcription completed in ${result.inferenceTimeMs.toStringAsFixed(1)}ms '
      'lang=${lang ?? "-"}',
    );
    return SttResult(
      text: result.text,
      inferenceTimeMs: result.inferenceTimeMs,
      lang: lang,
      confidence: result.confidence,
      durationMs: (audio.samples.length / audio.sampleRate) * 1000.0,
    );
  }

  void _warnIfUnsupported(String language) {
    final m = _model;
    if (m == null) return;
    if (m.languages.isEmpty) return;
    if (m.languages.contains(language)) return;
    SttLogger.w(
      'Model ${m.id} does not declare language="$language" '
      '(supported: ${m.languages.join(", ")}). Forwarding to engine anyway.',
    );
  }

  /// Configure an optional [LanguageDetector] used to populate [SttResult.lang]
  /// when the underlying recognizer does not return one. The [encoderPath] and
  /// [decoderPath] must point at the Whisper-tiny encoder/decoder ONNX files.
  void configureLanguageDetector({
    required String encoderPath,
    required String decoderPath,
  }) {
    _detector ??= LanguageDetector();
    _detectorEncoderPath = encoderPath;
    _detectorDecoderPath = decoderPath;
  }

  /// Cancels any ongoing transcription.
  void cancel() {
    _currentToken?.cancel();
  }

  Future<void> _cleanup() async {
    await _engine?.dispose();
    _engine = null;
    _model = null;
    _initialized = false;
  }

  /// Releases all resources held by the STT engine.
  ///
  /// Call this when you're done using the engine to free native resources.
  /// After calling dispose(), the engine cannot be used again until re-initialized.
  Future<void> dispose() async {
    cancel();
    if (!_initialized) return;
    await _engine?.dispose();
    _engine = null;
    _model = null;
    _initialized = false;
    await _detector?.dispose();
    _detector = null;
  }
}
