import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

void main() {
  test('ModelRegistry is seeded with default models', () {
    final models = ModelRegistry.available();
    expect(models.isNotEmpty, true);

    final whisperModels = ModelRegistry.available(type: SttModelType.whisper);
    expect(whisperModels.length, greaterThanOrEqualTo(5));

    final sherpaModels = ModelRegistry.available(type: SttModelType.sherpa);
    expect(sherpaModels.length, greaterThanOrEqualTo(1));

    final voxtralModels = ModelRegistry.available(type: SttModelType.voxtral);
    expect(voxtralModels.length, greaterThanOrEqualTo(1));
  });

  test('ModelRegistry.get returns correct models', () {
    final tiny = ModelRegistry.get('whisper-tiny');
    expect(tiny.type, SttModelType.whisper);
    expect(tiny.files.length, 3);
    expect(tiny.files[0].filename, 'encoder.onnx');
    expect(tiny.files[2].filename, 'vocab.json');

    final zipformer = ModelRegistry.get('sherpa-zipformer-en');
    expect(zipformer.type, SttModelType.sherpa);
    expect(zipformer.languages, contains('en'));

    final voxtral = ModelRegistry.get('voxtral-mini');
    expect(voxtral.type, SttModelType.voxtral);
    expect(voxtral.languages, contains('de'));
  });

  test('ModelRegistry.register adds custom model', () {
    ModelRegistry.register(ModelDescriptor(
      id: 'test-model',
      name: 'Test Model',
      type: SttModelType.whisper,
      languages: ['ja'],
      files: [
        ModelFile(url: 'https://example.com/encoder.onnx', filename: 'encoder.onnx'),
        ModelFile(url: 'https://example.com/decoder.onnx', filename: 'decoder.onnx'),
      ],
      sizeMb: 100,
    ));

    final model = ModelRegistry.get('test-model');
    expect(model.languages, contains('ja'));

    // Verify original models still accessible
    expect(ModelRegistry.isRegistered('whisper-tiny'), true);
  });

  test('ModelRegistry throws for unknown model', () {
    expect(
      () => ModelRegistry.get('nonexistent-model'),
      throwsArgumentError,
    );
  });

  test('ModelDescriptor files contain valid URLs', () {
    for (final model in ModelRegistry.available()) {
      for (final file in model.files) {
        expect(file.url.startsWith('https://'), true,
            reason: '${model.id}: ${file.filename} URL must be HTTPS');
        expect(file.filename.contains('/'), false,
            reason: '${model.id}: filename must not contain path separators');
      }
    }
  });
}
