import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

void main() {
  group('ModelRegistry', () {
    test('available returns all models', () {
      expect(ModelRegistry.available().length, greaterThanOrEqualTo(12));
    });

    test('available filters by type', () {
      final whisper = ModelRegistry.available(type: SttModelType.whisper);
      expect(whisper.every((m) => m.type == SttModelType.whisper), true);

      final sherpa = ModelRegistry.available(type: SttModelType.sherpa);
      expect(sherpa.every((m) => m.type == SttModelType.sherpa), true);

      final voxtral = ModelRegistry.available(type: SttModelType.voxtral);
      expect(voxtral.every((m) => m.type == SttModelType.voxtral), true);
    });

    test('get returns correct model by id', () {
      final model = ModelRegistry.get('whisper-tiny');
      expect(model.id, 'whisper-tiny');
      expect(model.name, contains('Tiny'));
    });

    test('register adds custom model', () {
      ModelRegistry.register(ModelDescriptor(
        id: 'custom-test',
        name: 'Custom Test',
        type: SttModelType.whisper,
        languages: ['en'],
        files: [ModelFile(url: 'https://example.com/m.onnx', filename: 'm.onnx')],
        sizeMb: 10,
      ));

      expect(ModelRegistry.isRegistered('custom-test'), true);
      expect(ModelRegistry.get('custom-test').files.length, 1);
    });

    test('throws for unknown model', () {
      expect(() => ModelRegistry.get('does-not-exist'), throwsArgumentError);
    });
  });
}
