import 'dart:io';
import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../stt_flutter_impl.dart';
import '../stt_result.dart';
import '../stt_exception.dart';
import '../model_registry.dart';
import '../model_downloader.dart';
import '../stt_logger.dart';
import '../audio/audio_processor.dart';

/// Singleton wrapper for [SttFlutter] that provides a convenient global access point
/// for speech-to-text functionality.
///
/// This class is a singleton, so use [SttEngine.instance] to access it.
///
/// Example usage:
/// ```dart
/// await SttEngine.instance.loadModel(ModelRegistry.get('whisper-tiny'));
/// final result = await SttEngine.instance.transcribeFile('/path/to/audio.wav');
/// print(result.text);
/// await SttEngine.instance.destroy();
/// ```
class SttEngine {
  static const String _hotwordsFilename = 'hotwords.txt';
  static const String _hotwordsTmpFilename = 'hotwords.txt.tmp';

  static bool _initialized = false;

  SttFlutter? _stt;
  ModelDescriptor? _model;
  String? _defaultLanguage;
  String? _modelDir;
  bool _cancelRequested = false;

  SttEngine._();

  static final SttEngine instance = SttEngine._();

  /// Whether the engine is ready to perform transcription.
  bool get isReady => _stt != null && !_cancelRequested;

  /// The currently loaded model, or null if no model is loaded.
  ModelDescriptor? get currentModel => _model;

  /// The current default language for transcription, or null if using auto-detection.
  String? get currentDefaultLanguage => _defaultLanguage;

  /// The directory containing the currently loaded model files.
  String? get currentModelDir => _modelDir;

  /// Load [model] into the engine. If [defaultLanguage] is provided, it is
  /// used as the default for subsequent transcribe calls when no per-call
  /// `language` override is supplied. Pass `null` (the default) to enable
  /// auto-detect mode.
  ///
  /// If [hotwords] is non-empty, it is written to `<modelDir>/hotwords.txt`
  /// and consumed by Zipformer (or any other model that supports hotwords).
  /// One entry per line, formatted as `"word score"`.
  ///
  /// Returns `null` on success, or an [SttException] on failure.
  Future<SttException?> loadModel(
    ModelDescriptor model, {
    String? modelDir,
    String? defaultLanguage,
    String? hotwords,
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
      if (hotwords != null) {
        await _writeHotwords(modelDir, hotwords);
      }
      final stt = _stt!;
      await stt.initialize(
        model: model,
        modelDir: modelDir,
        language: defaultLanguage,
      );
      _model = model;
      _modelDir = modelDir;
      _defaultLanguage =
          (defaultLanguage != null && defaultLanguage.isNotEmpty)
              ? defaultLanguage
              : null;
      return null;
    } catch (e, st) {
      await _stt?.dispose();
      _stt = null;
      _model = null;
      _modelDir = null;
      _defaultLanguage = null;
      SttLogger.e('SttEngine.loadModel failed', e, st);
      if (e is SttException) return e;
      return SttException.modelLoadFailed(e.toString());
    }
  }

  /// Update the hotwords text used by the currently loaded model. Writes
  /// `<modelDir>/hotwords.txt` and reloads the engine. The model must already
  /// be loaded via [loadModel].
  Future<SttException?> setHotwords(String? text) async {
    final m = _model;
    final dir = _modelDir;
    if (m == null || dir == null) {
      return SttException.notInitialized('SttEngine');
    }
    final defaultLang = _defaultLanguage;
    await _writeHotwords(dir, text ?? '');
    return loadModel(
      m,
      modelDir: dir,
      defaultLanguage: defaultLang,
      hotwords: text,
    );
  }

  static Future<void> _writeHotwords(String modelDir, String text) async {
    final file = File('$modelDir/$_hotwordsFilename');
    final tmp = File('$modelDir/$_hotwordsTmpFilename');
    final trimmed = text.trim();
    if (trimmed.isEmpty) {
      if (await file.exists()) await file.delete();
      if (await tmp.exists()) await tmp.delete();
      return;
    }
    await tmp.writeAsString(trimmed.endsWith('\n') ? trimmed : '$trimmed\n');
    await tmp.rename(file.path);
  }

  Future<SttResult> transcribeFile(
    String path, {
    String? language,
    PreprocessConfig preprocess = PreprocessConfig.none,
  }) async {
    final stt = _stt;
    if (stt == null) {
      throw SttException.notInitialized('SttEngine');
    }
    return stt.transcribeFile(path,
        language: language, preprocess: preprocess);
  }

  Future<SttResult> transcribeBuffer(
    Float32List samples,
    int sampleRate, {
    String? language,
    PreprocessConfig preprocess = PreprocessConfig.none,
  }) async {
    final stt = _stt;
    if (stt == null) {
      throw SttException.notInitialized('SttEngine');
    }
    return stt.transcribeBuffer(samples, sampleRate,
        language: language, preprocess: preprocess);
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
