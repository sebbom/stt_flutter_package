import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../offline_engine_base.dart';

class OmnilingualInferenceEngine extends OfflineEngineBase {
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

    final stream = rec.createStream();
    try {
      stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate);
      rec.decode(stream);
      final result = rec.getResult(stream);
      stopwatch.stop();
      SttLogger.d(
        'Omnilingual result: "${result.text}" in ${stopwatch.elapsedMilliseconds}ms',
      );
      return SttResult(
        text: result.text,
        inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
        lang: result.lang.isNotEmpty
            ? result.lang
            : (model.languages.isNotEmpty ? model.languages.first : null),
      );
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> dispose() async {
    freeRecognizer();
  }
}
