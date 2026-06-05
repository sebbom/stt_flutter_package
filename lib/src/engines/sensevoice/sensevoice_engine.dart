import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../offline_engine_base.dart';

class SenseVoiceInferenceEngine extends OfflineEngineBase {
  String? _lastConfiguredLang;
  String? _lastConfiguredTextNorm;
  Map<String, String>? _modelFiles;

  SenseVoiceInferenceEngine(super.model);

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    _modelFiles = Map.unmodifiable(modelFiles);
    _lastConfiguredLang = 'auto';
    _lastConfiguredTextNorm = 'withitn';
    _recreate(_lastConfiguredLang!, _lastConfiguredTextNorm!);
    SttLogger.d(
      'SenseVoiceInferenceEngine: loaded sense-voice model (${model.id}); '
      'default lang=$_lastConfiguredLang, itn=$_lastConfiguredTextNorm',
    );
  }

  void _recreate(String language, String textNorm) {
    final files = _modelFiles;
    if (files == null) return;
    final modelFile = findFile(files, ['model.onnx', 'model']);
    final tokens = findFile(files, ['tokens.txt', 'tokens']);

    final config = OfflineRecognizerConfig(
      model: OfflineModelConfig(
        senseVoice: OfflineSenseVoiceModelConfig(
          model: modelFile,
          language: language,
          useInverseTextNormalization: textNorm == 'withitn',
        ),
        tokens: tokens,
        numThreads: OfflineEngineBase.optimalThreadCount(),
        provider: 'cpu',
        debug: false,
        modelType: 'sense_voice',
      ),
      decodingMethod: 'greedy_search',
    );

    setRecognizer(OfflineRecognizer(config));
    _lastConfiguredLang = language;
    _lastConfiguredTextNorm = textNorm;
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
    final target = hasExplicit ? language : (_lastConfiguredLang ?? 'auto');

    warnIfLanguageUnsupported(
      hasExplicit ? target : null,
      supportsExplicitLanguage: supportsExplicitLanguage,
    );

    if (hasExplicit && target != _lastConfiguredLang) {
      _recreate(target, _lastConfiguredTextNorm ?? 'withitn');
    }

    final rec = recognizer;
    final stream = rec.createStream();
    try {
      stream.acceptWaveform(samples: audio.samples, sampleRate: audio.sampleRate);
      rec.decode(stream);
      final result = rec.getResult(stream);
      stopwatch.stop();
      final parsed = parseSenseVoiceText(result.text);
      SttLogger.d(
        'SenseVoice result (lang=$target): "${parsed.text}" '
        'emotion=${parsed.emotion} events=${parsed.events} '
        'in ${stopwatch.elapsedMilliseconds}ms',
      );
      return SttResult(
        text: parsed.text,
        inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
        lang: result.lang.isNotEmpty
            ? result.lang
            : (target == 'auto' ? null : target),
        emotion: parsed.emotion,
        events: parsed.events,
      );
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> dispose() async {
    _modelFiles = null;
    freeRecognizer();
  }
}

/// Extracts the spoken text from a SenseVoice result, separating out the
/// language/emotion/event special tokens (`<|...|>`).
///
/// SenseVoice emits a string like `<|en|><|NEUTRAL|><|Speech|><|withitn|>hello`,
/// where the meaningful transcript is everything after the last `withitn`/
/// `woitn` token. Emotion is one of NEUTRAL/HAPPY/ANGRY/SAD/SURPRISED/UNKNOWN.
/// Events are tags such as `Speech`, `BGM`, `Laughter`, `Music`, etc.
SenseVoiceParsed parseSenseVoiceText(String raw) {
  if (raw.isEmpty) return const SenseVoiceParsed(text: '');
  const openToken = '<|';
  const closeToken = '|>';
  final tags = <String>[];
  final buffer = StringBuffer();
  var i = 0;
  while (i < raw.length) {
    final open = raw.indexOf(openToken, i);
    if (open < 0) {
      buffer.write(raw.substring(i));
      break;
    }
    if (open > i) {
      buffer.write(raw.substring(i, open));
    }
    final close = raw.indexOf(closeToken, open + openToken.length);
    if (close < 0) {
      buffer.write(raw.substring(open));
      break;
    }
    tags.add(raw.substring(open + openToken.length, close));
    i = close + closeToken.length;
  }
  final text = buffer.toString().trim();
  String? emotion;
  final events = <String>[];
  for (final tag in tags) {
    final lower = tag.toLowerCase();
    const emotions = {
      'neutral', 'happy', 'angry', 'sad', 'surprised', 'unknown',
    };
    if (emotion == null && emotions.contains(lower)) {
      emotion = lower;
    } else if (lower != 'auto' &&
        lower != 'zh' &&
        lower != 'en' &&
        lower != 'yue' &&
        lower != 'ja' &&
        lower != 'ko' &&
        lower != 'nospeech' &&
        lower != 'withitn' &&
        lower != 'woitn') {
      events.add(tag);
    }
  }
  return SenseVoiceParsed(text: text, emotion: emotion, events: events);
}

/// Lightweight public view-model of a SenseVoice parse result.
class SenseVoiceParsed {
  const SenseVoiceParsed({
    required this.text,
    this.emotion,
    this.events = const <String>[],
  });
  final String text;
  final String? emotion;
  final List<String> events;
}
