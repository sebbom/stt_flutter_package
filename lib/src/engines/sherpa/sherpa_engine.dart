import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../../audio/audio_chunker.dart';
import '../offline_engine_base.dart';

class SherpaInferenceEngine extends OfflineEngineBase {
  static const ChunkingConfig _chunking = ChunkingConfig.defaultForTransducer;
  SherpaInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => false;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    final encoder = findFile(modelFiles, ['encoder.onnx', 'encoder']);
    final decoder = findFile(modelFiles, ['decoder.onnx', 'decoder']);
    final joiner = findFile(modelFiles, ['joiner.onnx', 'joiner']);
    final tokens = findFile(modelFiles, ['tokens.txt', 'tokens']);

    final hotwordsFile = _resolveHotwordsFile(modelFiles);
    final hotwordsScore = _resolveHotwordsScore();

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
      hotwordsFile: hotwordsFile,
      hotwordsScore: hotwordsScore,
    );

    setRecognizer(OfflineRecognizer(config));
    SttLogger.d(
      'SherpaInferenceEngine: loaded zipformer2 model (${model.id}); '
      'supportsExplicitLanguage=$supportsExplicitLanguage, '
      'hotwordsFile=${hotwordsFile.isEmpty ? "<none>" : hotwordsFile}, '
      'hotwordsScore=$hotwordsScore',
    );
  }

  String _resolveHotwordsFile(Map<String, String> modelFiles) {
    for (final file in model.files) {
      final path = file.hotwordsFile;
      if (path == null || path.isEmpty) continue;
      final resolved = modelFiles[file.filename];
      if (resolved != null && resolved.isNotEmpty) return resolved;
      final inDir = file.filename;
      for (final entry in modelFiles.entries) {
        if (entry.key.endsWith(inDir) || entry.value.endsWith(inDir)) {
          return entry.value;
        }
      }
      return path;
    }
    return '';
  }

  double _resolveHotwordsScore() {
    for (final file in model.files) {
      final score = file.hotwordsScore;
      if (score != null) return score;
    }
    return 1.5;
  }

  @override
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
    Map<String, dynamic>? options,
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
      'Sherpa result (${chunks.length} chunk${chunks.length == 1 ? "" : "s"}): '
      '"${textParts.join(" ")}" in ${stopwatch.elapsedMilliseconds}ms',
    );
    return SttResult(
      text: dedupJoinedText(textParts),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
      lang: model.languages.isNotEmpty ? model.languages.first : null,
    );
  }

  @override
  Future<void> dispose() async => freeRecognizer();
}
