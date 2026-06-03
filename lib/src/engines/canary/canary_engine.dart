import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../offline_engine_base.dart';

class CanaryInferenceEngine extends OfflineEngineBase {
  String? _lastConfiguredLang;

  CanaryInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    final encoder = findFile(modelFiles, ['encoder.onnx', 'encoder']);
    final decoder = findFile(modelFiles, ['decoder.onnx', 'decoder']);
    final tokens = findFile(modelFiles, ['tokens.txt', 'tokens']);

    final defaultLang = model.languages.isNotEmpty ? model.languages.first : '';
    _lastConfiguredLang = defaultLang;

    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        canary: OfflineCanaryModelConfig(
          encoder: encoder,
          decoder: decoder,
          srcLang: defaultLang,
          tgtLang: defaultLang,
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
    SttLogger.d(
      'CanaryInferenceEngine: loaded canary model (${model.id}); '
      'default lang=$defaultLang',
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

    final effective = (language != null && language.isNotEmpty)
        ? language
        : (_lastConfiguredLang ?? '');

    warnIfLanguageUnsupported(
      effective.isEmpty ? null : effective,
      supportsExplicitLanguage: supportsExplicitLanguage,
    );

    final stream = rec.createStream();
    try {
      if (effective.isNotEmpty && effective != _lastConfiguredLang) {
        stream.setOption(key: 'srcLang', value: effective);
        stream.setOption(key: 'tgtLang', value: effective);
      }
      stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate);
      rec.decode(stream);
      final result = rec.getResult(stream);
      stopwatch.stop();
      SttLogger.d('Canary result: "${result.text}" in ${stopwatch.elapsedMilliseconds}ms');
      return SttResult(
        text: result.text,
        inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
        lang: result.lang.isNotEmpty ? result.lang : effective,
      );
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> dispose() async => freeRecognizer();
}
