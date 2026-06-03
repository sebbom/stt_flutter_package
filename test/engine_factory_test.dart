import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'package:stt_flutter/src/engines/engine_factory.dart';

void main() {
  group('EngineFactory', () {
    test('only four model types are exposed', () {
      expect(SttModelType.values, hasLength(4));
      expect(
        SttModelType.values.map((e) => e.name).toSet(),
        {'whisper', 'sherpa', 'nemo', 'canary'},
      );
    });

    test('whisper model produces a WhisperInferenceEngine with languages', () {
      final model = ModelRegistry.get('whisper-tiny');
      final engine = createEngine(model);
      expect(engine, isA<WhisperInferenceEngine>());
      expect(engine.supportsExplicitLanguage, true);
      expect(engine.supportedLanguages, contains('en'));
      expect(engine.supportedLanguages, contains('de'));
    });

    test('sherpa-zipformer-en is monolingual and does not support explicit language',
        () {
      final model = ModelRegistry.get('sherpa-zipformer-en');
      final engine = createEngine(model);
      expect(engine, isA<SherpaInferenceEngine>());
      expect(engine.supportsExplicitLanguage, false);
      expect(engine.supportedLanguages, {'en'});
    });

    test('parakeet multilingual declares its 25 languages and supports explicit language',
        () {
      final model = ModelRegistry.get('parakeet-tdt-0.6b-multilingual');
      final engine = createEngine(model);
      expect(engine, isA<NemoInferenceEngine>());
      expect(engine.supportsExplicitLanguage, true);
      expect(engine.supportedLanguages, contains('de'));
      expect(engine.supportedLanguages, contains('fr'));
      expect(engine.supportedLanguages.length, greaterThanOrEqualTo(20));
    });

    test('canary model has restricted language set', () {
      // Find any registered canary model
      final canaryModels = ModelRegistry.available(type: SttModelType.canary);
      expect(canaryModels, isNotEmpty);
      final model = canaryModels.first;
      final engine = createEngine(model);
      expect(engine, isA<CanaryInferenceEngine>());
      expect(engine.supportsExplicitLanguage, true);
      expect(engine.supportedLanguages, isNotEmpty);
    });
  });
}
