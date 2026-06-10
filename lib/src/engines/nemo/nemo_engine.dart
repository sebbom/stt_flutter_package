import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../../audio/audio_chunker.dart';
import '../offline_engine_base.dart';

class NemoInferenceEngine extends OfflineEngineBase {
  static const ChunkingConfig _chunking = ChunkingConfig.defaultForTransducer;
  NemoInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    final encoder = findFile(modelFiles, ['encoder.onnx', 'encoder']);
    final decoder = findFile(modelFiles, ['decoder.onnx', 'decoder']);
    final joiner = findFile(modelFiles, ['joiner.onnx', 'joiner']);
    final tokens = findFile(modelFiles, ['tokens.txt', 'tokens']);

    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        transducer: OfflineTransducerModelConfig(
          encoder: encoder,
          decoder: decoder,
          joiner: joiner,
        ),
        tokens: tokens,
        numThreads: OfflineEngineBase.optimalThreadCount(),
        provider: 'cpu',
        debug: false,
        modelType: 'nemo_transducer',
      ),
      decodingMethod: 'greedy_search',
    );

    setRecognizer(OfflineRecognizer(config));
    SttLogger.d('NemoInferenceEngine: loaded nemo model (${model.id})');
  }

  @override
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
    Map<String, dynamic>? options,
  }) async {
    if (options?['beamSearch'] == true) {
      SttLogger.w(
        'Nemo transducer does not support beam search — falling back to greedy.',
      );
    }

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
        if (language != null && language.isNotEmpty) {
          stream.setOption(key: 'language', value: language);
        }
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
      'Nemo result (${chunks.length} chunk${chunks.length == 1 ? "" : "s"}): '
      '"${textParts.join(" ")}" in ${stopwatch.elapsedMilliseconds}ms',
    );
    return SttResult(
      text: dedupJoinedText(textParts),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
      lang: detectedLang ??
          (language != null && language.isNotEmpty ? language : null),
    );
  }

  @override
  Future<void> dispose() async => freeRecognizer();
}
