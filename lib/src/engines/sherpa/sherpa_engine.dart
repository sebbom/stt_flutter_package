import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../offline_engine_base.dart';

class SherpaInferenceEngine extends OfflineEngineBase {
  SherpaInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => false;

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
        modelType: 'zipformer2',
      ),
      decodingMethod: 'greedy_search',
    );

    setRecognizer(OfflineRecognizer(config));
    SttLogger.d(
      'SherpaInferenceEngine: loaded zipformer2 model (${model.id}); '
      'supportsExplicitLanguage=$supportsExplicitLanguage',
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

    final stream = rec.createStream();
    try {
      stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate);
      rec.decode(stream);
      final result = rec.getResult(stream);
      stopwatch.stop();
      SttLogger.d('Sherpa result: "${result.text}" in ${stopwatch.elapsedMilliseconds}ms');
      return SttResult(
        text: result.text,
        inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
        lang: model.languages.isNotEmpty ? model.languages.first : null,
      );
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> dispose() async => freeRecognizer();
}
