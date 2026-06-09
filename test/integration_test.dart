import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

void main() {
  group('Integration tests', () {
    test('ModelRegistry returns whisper-tiny model', () {
      final model = ModelRegistry.get('whisper-tiny');
      expect(model.id, 'whisper-tiny');
      expect(model.type, SttModelType.whisper);
      expect(model.languages, isNotEmpty);
      expect(model.files, isNotEmpty);
    });

    test('ModelRegistry available returns all models', () {
      final models = ModelRegistry.available();
      expect(models.length, greaterThanOrEqualTo(10));
    });

    test('ModelRegistry filters by type', () {
      final whisperModels = ModelRegistry.available(type: SttModelType.whisper);
      expect(whisperModels.every((m) => m.type == SttModelType.whisper), true);
      expect(whisperModels.length, greaterThanOrEqualTo(1));
    });

    test('SttResult has all expected fields', () {
      const result = SttResult(
        text: 'Hello world',
        inferenceTimeMs: 100.0,
        lang: 'en',
        confidence: 0.95,
        durationMs: 1000.0,
        emotion: 'neutral',
        events: ['speech'],
      );
      expect(result.text, 'Hello world');
      expect(result.lang, 'en');
      expect(result.confidence, 0.95);
      expect(result.durationMs, 1000.0);
      expect(result.emotion, 'neutral');
      expect(result.events, ['speech']);
    });

    test('PreprocessConfig.none is a no-op', () {
      expect(PreprocessConfig.none.isNoOp, true);
    });

    test('PreprocessConfig with gain is not a no-op', () {
      final config = PreprocessConfig(gain: 2.0);
      expect(config.isNoOp, false);
    });

    test('PreprocessConfig with high-pass is not a no-op', () {
      final config = PreprocessConfig(highPass: true);
      expect(config.isNoOp, false);
    });

    test('SttException has message', () {
      final exception = SttException.modelLoadFailed('Test error');
      expect(exception.message, 'Failed to load model: Test error');
    });

    test('CancellationToken can be cancelled', () {
      final token = CancellationToken();
      expect(token.isCancelled, false);
      token.cancel();
      expect(token.isCancelled, true);
    });

    test('CancellationToken throws when cancelled', () {
      final token = CancellationToken();
      token.cancel();
      expect(() => token.throwIfCancelled(), throwsA(isA<OperationCancelledException>()));
    });

    test('AudioBuffer has correct duration', () {
      final samples = Float32List.fromList(List.filled(16000, 0.0));
      final buffer = AudioBuffer(samples: samples, sampleRate: 16000);
      expect(buffer.durationSec, 1.0);
      expect(buffer.length, 16000);
    });

    test('ChunkingConfig has correct values', () {
      expect(ChunkingConfig.whisper.windowSeconds, 30);
      expect(ChunkingConfig.whisper.overlapSeconds, 5);
      expect(ChunkingConfig.defaultForTransducer.windowSeconds, 30);
      expect(ChunkingConfig.defaultForTransducer.overlapSeconds, 2);
    });

    test('SttModelType has all expected values', () {
      expect(SttModelType.values.length, 7);
      expect(SttModelType.values.map((e) => e.name).toSet(), {
        'whisper',
        'sherpa',
        'nemo',
        'canary',
        'sensevoice',
        'omnilingual',
        'qwen3asr',
      });
    });
  });
}
