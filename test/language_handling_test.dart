import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'package:stt_flutter/src/engines/offline_engine_base.dart';

class _FakeEngine extends OfflineEngineBase {
  String? lastLanguage;
  final List<SttResult> scriptedResults;

  _FakeEngine(super.model, {this.scriptedResults = const []});

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Future<void> load(Map<String, String> modelFiles) async {}

  @override
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
  }) async {
    lastLanguage = language;
    if (scriptedResults.isNotEmpty) {
      return scriptedResults.removeAt(0);
    }
    return SttResult(
      text: 'fake',
      inferenceTimeMs: 1.0,
      lang: language,
    );
  }

  @override
  Future<void> dispose() async => freeRecognizer();
}

AudioBuffer _buffer() => AudioBuffer(
      samples: Float32List(16000),
      sampleRate: 16000,
    );

void main() {
  group('Language handling plumbing (no real model)', () {
    test('voxtral is removed from SttModelType', () {
      expect(SttModelType.values, hasLength(4));
      expect(SttModelType.values.any((e) => e.name == 'voxtral'), false);
    });

    test('SttException.unsupportedLanguage carries a code and a message', () {
      final e = SttException.unsupportedLanguage('xx', 'm');
      expect(e.code, 6001);
      expect(e.message, contains('xx'));
      expect(e.message, contains('m'));
    });

    test('per-call language override is forwarded to the engine', () async {
      final model = ModelRegistry.get('whisper-tiny');
      final engine = _FakeEngine(model);
      final stt = SttFlutter.withEngine(model: model, engine: engine);

      await stt.transcribeBuffer(
        _buffer().samples,
        16000,
        language: 'de',
      );
      expect(engine.lastLanguage, 'de');

      await stt.dispose();
    });

    test('default language is used when no per-call override is given', () async {
      final model = ModelRegistry.get('whisper-tiny');
      final engine = _FakeEngine(model);
      final stt = SttFlutter.withEngine(
        model: model,
        engine: engine,
        language: 'fr',
      );

      await stt.transcribeBuffer(_buffer().samples, 16000);
      expect(engine.lastLanguage, 'fr');

      // Per-call wins.
      await stt.transcribeBuffer(_buffer().samples, 16000, language: 'es');
      expect(engine.lastLanguage, 'es');

      await stt.dispose();
    });

    test('no language anywhere → null forwarded to engine (auto-detect)', () async {
      final model = ModelRegistry.get('whisper-tiny');
      final engine = _FakeEngine(model);
      final stt = SttFlutter.withEngine(model: model, engine: engine);

      await stt.transcribeBuffer(_buffer().samples, 16000);
      expect(engine.lastLanguage, isNull);
      await stt.dispose();
    });

    test('SttResult.lang is preserved from the engine result', () async {
      final model = ModelRegistry.get('whisper-tiny');
      final engine = _FakeEngine(
        model,
        scriptedResults: [
          const SttResult(
            text: 'bonjour',
            inferenceTimeMs: 5.0,
            lang: 'fr',
          ),
        ],
      );
      final stt = SttFlutter.withEngine(
        model: model,
        engine: engine,
        language: 'fr',
      );

      final result = await stt.transcribeBuffer(
        _buffer().samples,
        16000,
        language: 'fr',
      );
      expect(result.lang, 'fr');
      expect(result.text, 'bonjour');
      await stt.dispose();
    });

    test('Per-call empty string still wins over the default', () async {
      final model = ModelRegistry.get('whisper-tiny');
      final engine = _FakeEngine(model);
      final stt = SttFlutter.withEngine(
        model: model,
        engine: engine,
        language: 'de',
      );
      await stt.transcribeBuffer(_buffer().samples, 16000, language: '');
      // Empty string is a deliberate "auto" override, so it should be passed
      // through as-is to the engine (which can interpret it as "use default").
      expect(engine.lastLanguage, '');
      await stt.dispose();
    });

    test(
      'auto mode + engine requires explicit lang → LanguageDetector drives input',
      () async {
        final model = ModelRegistry.get('whisper-tiny');
        final engine = _FakeEngine(model);
        final stt = SttFlutter.withEngine(model: model, engine: engine);
        stt.detector = _FakeDetector('es');

        final result = await stt.transcribeBuffer(
          _buffer().samples,
          16000,
        );

        // The detector's result is forwarded to the engine.
        expect(engine.lastLanguage, 'es');
        // And surfaced on the result.
        expect(result.lang, 'es');
        await stt.dispose();
      },
    );

    test(
      'auto mode + engine requires explicit lang + no detector → null forwarded',
      () async {
        final model = ModelRegistry.get('whisper-tiny');
        final engine = _FakeEngine(model);
        final stt = SttFlutter.withEngine(model: model, engine: engine);

        await stt.transcribeBuffer(_buffer().samples, 16000);

        expect(engine.lastLanguage, isNull);
        await stt.dispose();
      },
    );

    test(
      'per-call language wins over LanguageDetector in auto-capable engines',
      () async {
        final model = ModelRegistry.get('whisper-tiny');
        final engine = _FakeEngine(model);
        final stt = SttFlutter.withEngine(model: model, engine: engine);
        stt.detector = _FakeDetector('es');

        await stt.transcribeBuffer(_buffer().samples, 16000, language: 'de');

        expect(engine.lastLanguage, 'de');
        await stt.dispose();
      },
    );
  });
}

class _FakeDetector extends LanguageDetector {
  final String lang;
  int callCount = 0;
  _FakeDetector(this.lang);

  @override
  Future<String> detect(
    Float32List samples, {
    required int sampleRate,
    required String encoderPath,
    required String decoderPath,
  }) async {
    callCount++;
    return lang;
  }
}
