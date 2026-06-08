import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../../audio/audio_chunker.dart';
import '../offline_engine_base.dart';

class OmnilingualInferenceEngine extends OfflineEngineBase {
  static const ChunkingConfig _chunking = ChunkingConfig.defaultForTransducer;
  OmnilingualInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => false;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    final modelFile = findFile(modelFiles, ['model.onnx', 'model']);
    final tokens = findFile(modelFiles, ['tokens.txt', 'tokens']);

    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        omnilingual: OfflineOmnilingualAsrCtcModelConfig(model: modelFile),
        tokens: tokens,
        numThreads: OfflineEngineBase.optimalThreadCount(),
        provider: 'cpu',
        debug: false,
        modelType: 'omnilingual',
      ),
      decodingMethod: 'greedy_search',
    );

    setRecognizer(OfflineRecognizer(config));
    SttLogger.d(
      'OmnilingualInferenceEngine: loaded omnilingual model (${model.id})',
    );
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

    warnIfLanguageUnsupported(
      language,
      supportsExplicitLanguage: supportsExplicitLanguage,
    );

    final chunks = chunkBuffer(audio, config: _chunking);
    final textParts = <String>[];
    String? detectedLang;
    for (final chunk in chunks) {
      token?.throwIfCancelled();
      final stream = rec.createStream();
      try {
        stream.acceptWaveform(
            samples: chunk.samples, sampleRate: chunk.sampleRate);
        rec.decode(stream);
        final result = rec.getResult(stream);
        if (result.text.isNotEmpty) textParts.add(result.text);
        if (detectedLang == null && result.lang.isNotEmpty) {
          detectedLang = result.lang;
        }
      } finally {
        stream.free();
      }
    }
    stopwatch.stop();
    SttLogger.d(
      'Omnilingual result (${chunks.length} chunk${chunks.length == 1 ? "" : "s"}): '
      '"${textParts.join(" ")}" in ${stopwatch.elapsedMilliseconds}ms',
    );
    return SttResult(
      text: dedupJoinedText(textParts),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
      lang: detectedLang ??
          (model.languages.isNotEmpty ? model.languages.first : null),
    );
  }

  @override
  Future<void> dispose() async {
    freeRecognizer();
  }
}
