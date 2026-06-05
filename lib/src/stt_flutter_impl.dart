import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show visibleForTesting;
import 'model_registry.dart';
import 'model_downloader.dart';
import 'stt_result.dart';
import 'stt_exception.dart';
import 'stt_logger.dart';
import 'cancellation_token.dart';
import 'compute_worker.dart';
import 'audio/audio_buffer.dart';
import 'audio/audio_processor.dart';
import 'engines/inference_engine.dart';
import 'engines/engine_factory.dart';
import 'language/language_detector.dart';

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

  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,
    String? language,
  }) async {
    if (_initialized) throw SttException.notInitialized('SttFlutter');
    if (model.id.isEmpty) throw SttException.invalidArgument('model.id must not be empty');
    _model = model;
    _defaultLanguage = (language != null && language.isNotEmpty) ? language : null;

    await ComputeWorker.instance.initialize();

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

    final AudioBuffer resampled;
    if (audio.sampleRate == 16000) {
      resampled = audio;
    } else {
      resampled = await ComputeWorker.instance.resample(audio);
    }
    token?.throwIfCancelled();

    String? effective = language ?? _defaultLanguage;
    String? detectedLang;

    final needAutoDetect = (effective == null || effective.isEmpty) &&
        _engine!.supportsExplicitLanguage;

    if (needAutoDetect && _detector != null) {
      try {
        final detected = await _detector!.detect(
          resampled.samples,
          sampleRate: 16000,
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
      resampled,
      language: effective,
      token: token,
    );

    String? lang = result.lang;
    if ((lang == null || lang.isEmpty) && detectedLang != null) {
      lang = detectedLang;
    } else if ((lang == null || lang.isEmpty) && _detector != null) {
      try {
        lang = await _detector!.detect(
          resampled.samples,
          sampleRate: 16000,
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
      durationMs: (resampled.samples.length / 16000) * 1000.0,
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

  void cancel() {
    _currentToken?.cancel();
  }

  Future<void> _cleanup() async {
    await _engine?.dispose();
    _engine = null;
    _model = null;
    _initialized = false;
  }

  Future<void> dispose() async {
    cancel();
    if (!_initialized) return;
    await _engine?.dispose();
    _engine = null;
    _model = null;
    _initialized = false;
    await _detector?.dispose();
    _detector = null;
    await ComputeWorker.instance.dispose();
  }
}
