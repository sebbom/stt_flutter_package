import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

class _CapturingEngine implements InferenceEngine {
  AudioBuffer? receivedAudio;
  String? receivedLanguage;
  Map<String, dynamic>? receivedOptions;
  bool disposed = false;

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Set<String> get supportedLanguages => {'en', 'fr'};

  @override
  Future<void> load(Map<String, String> modelFiles) async {}

  @override
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
    Map<String, dynamic>? options,
  }) async {
    receivedAudio = audio;
    receivedLanguage = language;
    receivedOptions = options;
    return SttResult(
      text: 'transcribed: ${audio.samples.length} @ ${audio.sampleRate}Hz',
      inferenceTimeMs: 50.0,
      lang: language ?? 'en',
      confidence: 0.95,
      durationMs: (audio.samples.length / audio.sampleRate) * 1000.0,
    );
  }

  @override
  Future<void> dispose() async {
    disposed = true;
  }
}

ModelDescriptor _testModel({List<String>? languages}) {
  return ModelDescriptor(
    id: 'pipeline-test-model',
    name: 'Pipeline Test Model',
    type: SttModelType.whisper,
    languages: languages ?? ['en', 'fr'],
    files: [
      ModelFile(
        url: 'https://example.com/encoder.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: 'https://example.com/decoder.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 1,
  );
}

void main() {
  // Verify asset files exist before running tests
  setUpAll(() {
    expect(File('example/assets/hello_en.wav').existsSync(), isTrue);
    expect(File('example/assets/podcast_fr.wav').existsSync(), isTrue);
    expect(File('example/assets/hello_en_tr.md').existsSync(), isTrue);
    expect(File('example/assets/podcast_fr.md').existsSync(), isTrue);
  });

  group('WAV pipeline integration', () {
    late SttFlutter stt;
    late _CapturingEngine engine;

    tearDown(() async {
      try {
        await stt.dispose();
      } catch (_) {}
    });

    test('loads hello_en.wav (8kHz) and resamples to 16kHz', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'en',
      );

      final result = await stt.transcribeFile('example/assets/hello_en.wav');

      expect(engine.receivedAudio, isNotNull);
      expect(engine.receivedAudio!.sampleRate, AudioProcessor.targetSampleRate);
      expect(engine.receivedAudio!.samples.length, greaterThan(0));
      expect(engine.receivedLanguage, 'en');
      expect(result.lang, 'en');
      expect(result.confidence, 0.95);
      expect(result.inferenceTimeMs, 50.0);
    });

    test('loads podcast_fr.wav (16kHz) without resampling', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'fr',
      );

      final result = await stt.transcribeFile('example/assets/podcast_fr.wav');

      expect(engine.receivedAudio, isNotNull);
      expect(engine.receivedAudio!.sampleRate, AudioProcessor.targetSampleRate);
      expect(engine.receivedLanguage, 'fr');
      expect(result.lang, 'fr');
    });

    test('forwarding per-call language override', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'en',
      );

      await stt.transcribeFile('example/assets/podcast_fr.wav', language: 'fr');
      expect(engine.receivedLanguage, 'fr');

      await stt.transcribeFile('example/assets/hello_en.wav', language: 'en');
      expect(engine.receivedLanguage, 'en');
    });

    test('forwarding default language when per-call not specified', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'fr',
      );

      await stt.transcribeFile('example/assets/podcast_fr.wav');
      expect(engine.receivedLanguage, 'fr');
    });

    test('duration matches expected WAV length', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'en',
      );

      final result = await stt.transcribeFile('example/assets/hello_en.wav');
      expect(result.durationMs, greaterThan(33000.0));
      expect(result.durationMs, lessThan(34000.0));
    });

    test('transcribe with preprocessing config', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'en',
      );

      await stt.transcribeFile(
        'example/assets/hello_en.wav',
        preprocess: const PreprocessConfig(gain: 2.0, highPass: true),
      );

      expect(engine.receivedAudio, isNotNull);
      expect(engine.receivedAudio!.sampleRate, AudioProcessor.targetSampleRate);
    });

    test('throws on non-existent file', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
      );

      expect(
        () => stt.transcribeFile('example/assets/non_existent.wav'),
        throwsA(isA<SttException>()),
      );
    });

    test('beamSearch option forwarded to engine', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'en',
      );

      await stt.transcribeFile('example/assets/hello_en.wav', beamSearch: true);
      expect(engine.receivedOptions, {'beamSearch': true});
    });

    test('default beamSearch is null', () async {
      engine = _CapturingEngine();
      stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'en',
      );

      await stt.transcribeFile('example/assets/hello_en.wav');
      expect(engine.receivedOptions, isNull);
    });
  });

  group('Expected transcript files', () {
    test('hello_en_tr.md contains expected English text', () {
      final text = File('example/assets/hello_en_tr.md').readAsStringSync();
      expect(text.trim().isNotEmpty, isTrue);
      expect(text, contains('The birch canoe slid on the smooth planks'));
      expect(text, contains('It is easy to tell the depth of a well'));
    });

    test('podcast_fr.md contains expected French text', () {
      final text = File('example/assets/podcast_fr.md').readAsStringSync();
      expect(text.trim().isNotEmpty, isTrue);
      expect(text, contains('France Inter'));
      expect(text, contains('Pierre Astier'));
      expect(text, contains('Netanyahou'));
    });
  });

  group('CancellationToken with WAV pipeline', () {
    test('cancelled token before transcribeFile throws', () async {
      final engine = _CapturingEngine();
      final stt = SttFlutter.withEngine(
        model: _testModel(),
        engine: engine,
        language: 'en',
      );

      final token = CancellationToken();
      token.cancel();

      expect(
        () => stt.transcribeFile('example/assets/hello_en.wav', token: token),
        throwsA(isA<OperationCancelledException>()),
      );

      await stt.dispose();
    });
  });
}
