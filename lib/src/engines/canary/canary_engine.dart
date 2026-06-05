import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../../audio/audio_chunker.dart';
import '../offline_engine_base.dart';

class CanaryInferenceEngine extends OfflineEngineBase {
  static const ChunkingConfig _chunking = ChunkingConfig.defaultForTransducer;
  String? _lastConfiguredLang;
  Map<String, String>? _modelFiles;

  CanaryInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    _modelFiles = Map.unmodifiable(modelFiles);
    final defaultLang = model.languages.isNotEmpty ? model.languages.first : '';
    _lastConfiguredLang = defaultLang;
    _recreate(defaultLang);
    SttLogger.d(
      'CanaryInferenceEngine: loaded canary model (${model.id}); '
      'default lang=$defaultLang',
    );
  }

  void _recreate(String lang) {
    final files = _modelFiles;
    if (files == null) return;
    final encoder = findFile(files, ['encoder.onnx', 'encoder']);
    final decoder = findFile(files, ['decoder.onnx', 'decoder']);
    final tokens = findFile(files, ['tokens.txt', 'tokens']);
    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        canary: OfflineCanaryModelConfig(
          encoder: encoder,
          decoder: decoder,
          srcLang: lang,
          tgtLang: lang,
          usePnc: true,
        ),
        tokens: tokens,
        numThreads: OfflineEngineBase.optimalThreadCount(),
        provider: 'cpu',
        debug: false,
        modelType: 'canary',
      ),
      decodingMethod: 'greedy_search',
    );
    setRecognizer(OfflineRecognizer(config));
    _lastConfiguredLang = lang;
  }

  @override
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
  }) async {
    final stopwatch = Stopwatch()..start();
    token?.throwIfCancelled();

    final hasExplicit = language != null && language.isNotEmpty;
    final target = hasExplicit ? language : (_lastConfiguredLang ?? '');

    warnIfLanguageUnsupported(
      hasExplicit ? target : null,
      supportsExplicitLanguage: supportsExplicitLanguage,
    );

    if (hasExplicit && target != _lastConfiguredLang) {
      _recreate(target);
    }

    final rec = recognizer;
    final chunks = chunkBuffer(audio, config: _chunking);
    final textParts = <String>[];
    for (final chunk in chunks) {
      token?.throwIfCancelled();
      final stream = rec.createStream();
      try {
        stream.acceptWaveform(
            samples: chunk.samples, sampleRate: chunk.sampleRate);
        rec.decode(stream);
        final result = rec.getResult(stream);
        if (result.text.isNotEmpty) textParts.add(result.text);
      } finally {
        stream.free();
      }
    }
    stopwatch.stop();
    SttLogger.d(
      'Canary result (${chunks.length} chunk${chunks.length == 1 ? "" : "s"}, '
      'lang=$target): "${textParts.join(" ")}" '
      'in ${stopwatch.elapsedMilliseconds}ms',
    );
    return SttResult(
      text: dedupJoinedText(textParts),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
      lang: target,
    );
  }

  @override
  Future<void> dispose() async {
    _modelFiles = null;
    freeRecognizer();
  }
}
