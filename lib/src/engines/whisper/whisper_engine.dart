import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../../audio/audio_chunker.dart';
import '../offline_engine_base.dart';

class WhisperInferenceEngine extends OfflineEngineBase {
  static const ChunkingConfig _chunking = ChunkingConfig.whisper;

  WhisperInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    final encoder = findFile(modelFiles, ['encoder.onnx', 'encoder']);
    final decoder = findFile(modelFiles, ['decoder.onnx', 'decoder']);
    final tokens = findFile(modelFiles, ['tokens.txt', 'tokens']);

    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        whisper: OfflineWhisperModelConfig(
          encoder: encoder,
          decoder: decoder,
          language: '',
          task: 'transcribe',
          tailPaddings: -1,
          enableTokenTimestamps: true,
          enableSegmentTimestamps: true,
        ),
        tokens: tokens,
        numThreads: OfflineEngineBase.optimalThreadCount(),
        provider: 'cpu',
        debug: false,
        modelType: 'whisper',
      ),
      decodingMethod: 'greedy_search',
    );

    setRecognizer(OfflineRecognizer(config));
    SttLogger.d('WhisperInferenceEngine: loaded whisper model (${model.id})');
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
    String? detectedLang;
    for (final chunk in chunks) {
      token?.throwIfCancelled();
      final result = await _decodeChunk(
        rec,
        chunk.samples,
        chunk.sampleRate,
        language: language,
        token: token,
      );
      if (result.text.isNotEmpty) textParts.add(result.text);
      if (detectedLang == null && result.lang.isNotEmpty) {
        detectedLang = result.lang;
      }
    }

    stopwatch.stop();
    return SttResult(
      text: dedupJoinedText(textParts),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
      lang: detectedLang ?? _normalize(language),
    );
  }

  String? _normalize(String? language) =>
      (language != null && language.isNotEmpty) ? language : null;

  Future<OfflineRecognizerResult> _decodeChunk(
    OfflineRecognizer rec,
    Float32List chunk,
    int sampleRate, {
    String? language,
    CancellationToken? token,
  }) async {
    token?.throwIfCancelled();
    final stream = rec.createStream();
    try {
      if (language != null && language.isNotEmpty) {
        stream.setOption(key: 'language', value: language);
      }
      stream.acceptWaveform(samples: chunk, sampleRate: sampleRate);
      rec.decode(stream);
      return rec.getResult(stream);
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> dispose() async {
    freeRecognizer();
  }
}
