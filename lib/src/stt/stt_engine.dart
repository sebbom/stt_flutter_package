import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../stt_flutter_impl.dart';
import '../stt_result.dart';
import '../stt_exception.dart';
import '../model_registry.dart';
import '../model_downloader.dart';
import '../stt_logger.dart';

class SttEngine {
  static bool _initialized = false;

  SttFlutter? _stt;
  ModelDescriptor? _model;
  String? _defaultLanguage;
  bool _cancelRequested = false;

  SttEngine._();

  static final SttEngine instance = SttEngine._();

  bool get isReady => _stt != null && !_cancelRequested;
  ModelDescriptor? get currentModel => _model;
  String? get currentDefaultLanguage => _defaultLanguage;

  /// Load [model] into the engine. If [defaultLanguage] is provided, it is
  /// used as the default for subsequent transcribe calls when no per-call
  /// `language` override is supplied. Pass `null` (the default) to enable
  /// auto-detect mode.
  ///
  /// Returns `null` on success, or an [SttException] on failure.
  Future<SttException?> loadModel(
    ModelDescriptor model, {
    String? modelDir,
    String? defaultLanguage,
  }) async {
    try {
      if (!_initialized) {
        initBindings();
        _initialized = true;
      }
      _cancelRequested = false;
      await _stt?.dispose();
      _stt = SttFlutter();
      if (modelDir == null) {
        modelDir = await ModelDownloader.defaultStoragePath(model);
        if (!await ModelDownloader.isDownloaded(model, storagePath: modelDir)) {
          await ModelDownloader.download(model, storagePath: modelDir);
        }
      }
      final stt = _stt!;
      await stt.initialize(
        model: model,
        modelDir: modelDir,
        language: defaultLanguage,
      );
      _model = model;
      _defaultLanguage =
          (defaultLanguage != null && defaultLanguage.isNotEmpty)
              ? defaultLanguage
              : null;
      return null;
    } catch (e, st) {
      await _stt?.dispose();
      _stt = null;
      _model = null;
      _defaultLanguage = null;
      SttLogger.e('SttEngine.loadModel failed', e, st);
      if (e is SttException) return e;
      return SttException.modelLoadFailed(e.toString());
    }
  }

  Future<SttResult> transcribeFile(
    String path, {
    String? language,
  }) async {
    final stt = _stt;
    if (stt == null) {
      throw SttException.notInitialized('SttEngine');
    }
    return stt.transcribeFile(path, language: language);
  }

  Future<SttResult> transcribeBuffer(
    Float32List samples,
    int sampleRate, {
    String? language,
  }) async {
    final stt = _stt;
    if (stt == null) {
      throw SttException.notInitialized('SttEngine');
    }
    return stt.transcribeBuffer(samples, sampleRate, language: language);
  }

  /// Set the language-detector fallback (uses a Whisper-tiny SLI model).
  void configureLanguageDetector({
    required String encoderPath,
    required String decoderPath,
  }) {
    final stt = _stt;
    if (stt == null) {
      throw SttException.notInitialized('SttEngine');
    }
    stt.configureLanguageDetector(
      encoderPath: encoderPath,
      decoderPath: decoderPath,
    );
  }

  void cancel() {
    _cancelRequested = true;
    _stt?.cancel();
  }

  Future<void> destroy() async {
    await _stt?.dispose();
    _stt = null;
    _model = null;
    _defaultLanguage = null;
    _cancelRequested = false;
  }
}
