import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../offline_engine_base.dart';

class WhisperInferenceEngine extends OfflineEngineBase {
  static const int _whisperSampleRate = 16000;
  static const int _maxChunkSamples = 30 * _whisperSampleRate;
  static const int _overlapSamples = 5 * _whisperSampleRate;

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
  }) async {
    final stopwatch = Stopwatch()..start();
    final rec = recognizer;
    token?.throwIfCancelled();

    warnIfLanguageUnsupported(
      language,
      supportsExplicitLanguage: supportsExplicitLanguage,
    );

    final samples = audio.samples;
    final sampleRate = audio.sampleRate;
    final totalSamples = samples.length;

    if (totalSamples <= _maxChunkSamples) {
      final result = await _decodeChunk(
        rec,
        samples,
        sampleRate,
        language: language,
        token: token,
      );
      stopwatch.stop();
      return SttResult(
        text: result.text,
        inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
        lang: result.lang.isNotEmpty ? result.lang : _normalize(language),
      );
    }

    final textParts = <String>[];
    String? detectedLang;
    int offset = 0;
    while (offset < totalSamples) {
      token?.throwIfCancelled();
      final end = (offset + _maxChunkSamples).clamp(0, totalSamples);
      final chunk = Float32List.sublistView(samples, offset, end);
      final result = await _decodeChunk(
        rec,
        chunk,
        sampleRate,
        language: language,
        token: token,
      );
      if (result.text.isNotEmpty) textParts.add(result.text);
      if (detectedLang == null && result.lang.isNotEmpty) {
        detectedLang = result.lang;
      }
      if (end >= totalSamples) break;
      offset += _maxChunkSamples - _overlapSamples;
    }

    stopwatch.stop();
    return SttResult(
      text: _dedupJoinedText(textParts),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
      lang: detectedLang ?? _normalize(language),
    );
  }

  String? _normalize(String? language) =>
      (language != null && language.isNotEmpty) ? language : null;

  /// Joins chunk transcriptions and removes duplicated words at chunk
  /// boundaries caused by the overlap window.
  String _dedupJoinedText(List<String> parts) {
    if (parts.length <= 1) return parts.join(' ').trim();
    final out = StringBuffer();
    String prevTail = '';
    for (var i = 0; i < parts.length; i++) {
      final p = parts[i].trim();
      if (p.isEmpty) continue;
      if (i == 0) {
        out.write(p);
      } else {
        final deduped = _stripOverlapPrefix(p, prevTail);
        if (deduped.isNotEmpty) {
          if (out.isNotEmpty) out.write(' ');
          out.write(deduped);
        }
      }
      prevTail = _tailWords(p, 8);
    }
    return out.toString();
  }

  String _stripOverlapPrefix(String current, String previousTail) {
    if (previousTail.isEmpty) return current;
    final prevWords = previousTail.split(RegExp(r'\s+'));
    final curWords = current.split(RegExp(r'\s+'));
    final maxN = prevWords.length < curWords.length
        ? prevWords.length
        : curWords.length;
    for (var n = maxN; n > 0; n--) {
      final tail = prevWords.sublist(prevWords.length - n).join(' ');
      final head = curWords.sublist(0, n).join(' ');
      if (_fuzzyEqual(tail, head)) {
        return curWords.sublist(n).join(' ');
      }
    }
    return current;
  }

  String _tailWords(String s, int n) {
    final words = s.split(RegExp(r'\s+'));
    if (words.length <= n) return s;
    return words.sublist(words.length - n).join(' ');
  }

  bool _fuzzyEqual(String a, String b) {
    if (a == b) return true;
    final la = a.toLowerCase();
    final lb = b.toLowerCase();
    if (la == lb) return true;
    final strippedA = la.replaceAll(RegExp(r'[^\w\s]'), '');
    final strippedB = lb.replaceAll(RegExp(r'[^\w\s]'), '');
    if (strippedA == strippedB) return true;
    if (strippedA.length > 4 && strippedB.length > 4) {
      return strippedA.substring(strippedA.length - 4) ==
          strippedB.substring(strippedB.length - 4);
    }
    return false;
  }

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
