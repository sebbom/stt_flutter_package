import 'dart:io';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../stt_logger.dart';
import '../model_registry.dart';
import 'inference_engine.dart';

/// Common scaffolding for engines that wrap a `sherpa_onnx.OfflineRecognizer`.
///
/// Provides:
/// - the loose file-by-name lookup used to map a [ModelDescriptor] file list
///   to the encoder/decoder/joiner/tokens paths sherpa expects;
/// - a CPU thread-count heuristic;
/// - the `recognizer` reference and language-related metadata derived from
///   the [ModelDescriptor] passed at construction time.
abstract class OfflineEngineBase implements InferenceEngine {
  OfflineRecognizer? _recognizer;
  final ModelDescriptor model;

  OfflineEngineBase(this.model);

  @override
  Set<String> get supportedLanguages => model.languages.toSet();

  OfflineRecognizer get recognizer {
    final r = _recognizer;
    if (r == null) {
      throw StateError('$runtimeType not loaded');
    }
    return r;
  }

  String findFile(Map<String, String> files, List<String> patterns) {
    for (final p in patterns) {
      if (files.containsKey(p)) return files[p]!;
    }
    for (final p in patterns) {
      for (final entry in files.entries) {
        if (entry.key.contains(p)) return entry.value;
      }
    }
    throw FileSystemException('Model file not found for patterns: $patterns');
  }

  void setRecognizer(OfflineRecognizer r) {
    _recognizer?.free();
    _recognizer = r;
  }

  void freeRecognizer() {
    _recognizer?.free();
    _recognizer = null;
  }

  void warnIfLanguageUnsupported(
    String? language, {
    required bool supportsExplicitLanguage,
  }) {
    if (language == null || language.isEmpty) return;
    if (model.languages.isEmpty) return;
    if (model.languages.contains(language)) return;
    final supported = model.languages.join(', ');
    if (supportsExplicitLanguage) {
      SttLogger.w(
        'Engine $runtimeType (model=$model.id) does not declare '
        'language="$language" in its supported list ($supported). '
        'Forwarding to native engine — it will reject if invalid.',
      );
    } else {
      SttLogger.w(
        'Engine $runtimeType (model=$model.id) does not support '
        'language="$language" (supported: $supported). '
        'Transcribing with the model\'s native language.',
      );
    }
  }

  static int optimalThreadCount() {
    final cores = Platform.numberOfProcessors;
    if (cores >= 8) return 4;
    if (cores >= 6) return 3;
    if (cores >= 4) return 2;
    return 1;
  }
}
