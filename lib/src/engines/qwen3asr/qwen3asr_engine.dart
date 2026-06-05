import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../offline_engine_base.dart';

class Qwen3AsrInferenceEngine extends OfflineEngineBase {
  /// Qwen3-ASR defaults to Chinese output when no language hint is provided,
  /// so we fall back to a sensible default from the model descriptor's
  /// supported list. Callers can override via [SttEngine.setDefaultLanguage]
  /// or the per-call `language` argument.
  static const String _fallbackLanguage = 'en';

  Qwen3AsrInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    final conv = findFile(modelFiles, ['conv_frontend.onnx', 'conv_frontend']);
    final encoder = findFile(modelFiles, ['encoder.int8.onnx', 'encoder']);
    final decoder = findFile(modelFiles, ['decoder.int8.onnx', 'decoder']);

    String? tokenizerDir;
    for (final entry in modelFiles.entries) {
      if (entry.key.endsWith('tokenizer/vocab.json')) {
        final filePath = entry.value;
        final lastSlash = filePath.lastIndexOf('/');
        if (lastSlash > 0) {
          tokenizerDir = filePath.substring(0, lastSlash);
        }
        break;
      }
    }
    if (tokenizerDir == null) {
      for (final entry in modelFiles.entries) {
        if (entry.key == 'tokenizer') {
          tokenizerDir = entry.value;
          break;
        }
      }
    }
    if (tokenizerDir == null) {
      throw StateError(
        'Qwen3-ASR model requires a tokenizer/ subdirectory containing '
        'vocab.json, merges.txt, and tokenizer_config.json. '
        'No such directory was found in the model files.',
      );
    }

    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        qwen3Asr: OfflineQwen3AsrModelConfig(
          convFrontend: conv,
          encoder: encoder,
          decoder: decoder,
          tokenizer: tokenizerDir,
        ),
        tokens: '',
        numThreads: OfflineEngineBase.optimalThreadCount(),
        provider: 'cpu',
        debug: false,
        modelType: 'qwen3_asr',
      ),
      decodingMethod: 'greedy_search',
    );

    setRecognizer(OfflineRecognizer(config));
    SttLogger.d('Qwen3AsrInferenceEngine: loaded qwen3-asr model (${model.id})');
  }

  @override
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
  }) async {
    final stopwatch = Stopwatch()..start();
    final rec = recognizer;
    token?.throwIfCancelled();

    final effective = _resolveLanguage(language);
    warnIfLanguageUnsupported(
      effective,
      supportsExplicitLanguage: supportsExplicitLanguage,
    );

    final stream = rec.createStream();
    try {
      stream.setOption(key: 'language', value: effective);
      stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate);
      rec.decode(stream);
      final result = rec.getResult(stream);
      stopwatch.stop();
      SttLogger.d(
        'Qwen3-ASR result (lang=$effective): "${result.text}" '
        'in ${stopwatch.elapsedMilliseconds}ms',
      );
      return SttResult(
        text: result.text,
        inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
        lang: effective,
      );
    } finally {
      stream.free();
    }
  }

  String _resolveLanguage(String? language) {
    if (language != null && language.isNotEmpty) return language;
    SttLogger.w(
      'Qwen3-ASR received no language hint — falling back to "$_fallbackLanguage". '
      'Use SttEngine.setDefaultLanguage() or pass language: "..." to override. '
      'Without a language, the model defaults to Chinese output.',
    );
    return _fallbackLanguage;
  }

  @override
  Future<void> dispose() async {
    freeRecognizer();
  }
}
