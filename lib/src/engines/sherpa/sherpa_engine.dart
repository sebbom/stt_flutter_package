import 'dart:io';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../inference_engine.dart';

class SherpaInferenceEngine implements InferenceEngine {
  OfflineRecognizer? _recognizer;

  SherpaInferenceEngine();

  String _findFile(Map<String, String> files, List<String> patterns) {
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

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    final encoder = _findFile(modelFiles, ['encoder.onnx', 'encoder']);
    final decoder = _findFile(modelFiles, ['decoder.onnx', 'decoder']);
    final joiner = _findFile(modelFiles, ['joiner.onnx', 'joiner']);
    final tokens = _findFile(modelFiles, ['tokens.txt', 'tokens']);

    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        transducer: OfflineTransducerModelConfig(
          encoder: encoder,
          decoder: decoder,
          joiner: joiner,
        ),
        tokens: tokens,
        numThreads: _optimalThreadCount(),
        provider: 'cpu',
        debug: false,
        modelType: 'zipformer2',
      ),
      decodingMethod: 'greedy_search',
    );

    _recognizer = OfflineRecognizer(config);
    SttLogger.d('SherpaInferenceEngine: loaded offline zipformer2 model');
  }

  @override
  Future<SttResult> transcribe(AudioBuffer audio,
      {String? language, CancellationToken? token}) async {
    final stopwatch = Stopwatch()..start();
    final recognizer = _recognizer;
    if (recognizer == null) {
      throw StateError('SherpaInferenceEngine not loaded');
    }

    token?.throwIfCancelled();

    final stream = recognizer.createStream();
    try {
      stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate);
      recognizer.decode(stream);

      final result = recognizer.getResult(stream);
      final text = result.text;

      stopwatch.stop();
      SttLogger.d(
          'Sherpa result: "$text" in ${stopwatch.elapsedMilliseconds}ms');

      return SttResult(
        text: text,
        inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
      );
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> dispose() async {
    _recognizer?.free();
    _recognizer = null;
  }

  static int _optimalThreadCount() {
    final cores = Platform.numberOfProcessors;
    if (cores >= 8) return 4;
    if (cores >= 6) return 3;
    if (cores >= 4) return 2;
    return 1;
  }
}
