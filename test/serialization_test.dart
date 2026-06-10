import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

void main() {
  group('JSON Serialization', () {
    group('ModelFile', () {
      test('toJson and fromJson round-trip', () {
        const original = ModelFile(
          url: 'https://example.com/model.onnx',
          filename: 'model.onnx',
          sha256: 'abc123',
          sizeBytes: 1000000,
          hotwordsFile: 'hotwords.txt',
          hotwordsScore: 1.5,
          hotwordsString: 'hello,world',
        );

        final json = original.toJson();
        final deserialized = ModelFile.fromJson(json);

        expect(deserialized.url, original.url);
        expect(deserialized.filename, original.filename);
        expect(deserialized.sha256, original.sha256);
        expect(deserialized.sizeBytes, original.sizeBytes);
        expect(deserialized.hotwordsFile, original.hotwordsFile);
        expect(deserialized.hotwordsScore, original.hotwordsScore);
        expect(deserialized.hotwordsString, original.hotwordsString);
      });

      test('toJson and fromJson with minimal fields', () {
        const original = ModelFile(
          url: 'https://example.com/model.onnx',
          filename: 'model.onnx',
        );

        final json = original.toJson();
        final deserialized = ModelFile.fromJson(json);

        expect(deserialized.url, original.url);
        expect(deserialized.filename, original.filename);
        expect(deserialized.sha256, isNull);
        expect(deserialized.sizeBytes, isNull);
      });
    });

    group('ModelDescriptor', () {
      test('toJson and fromJson round-trip', () {
        const original = ModelDescriptor(
          id: 'test-model',
          name: 'Test Model',
          type: SttModelType.whisper,
          languages: ['en', 'fr', 'de'],
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
          sizeMb: 150,
        );

        final json = original.toJson();
        final deserialized = ModelDescriptor.fromJson(json);

        expect(deserialized.id, original.id);
        expect(deserialized.name, original.name);
        expect(deserialized.type, original.type);
        expect(deserialized.languages, original.languages);
        expect(deserialized.files.length, original.files.length);
        expect(deserialized.files[0].filename, 'encoder.onnx');
        expect(deserialized.files[1].filename, 'decoder.onnx');
        expect(deserialized.sizeMb, original.sizeMb);
      });

      test('toJsonString and fromJsonString round-trip', () {
        const original = ModelDescriptor(
          id: 'test-model',
          name: 'Test Model',
          type: SttModelType.sherpa,
          languages: ['en'],
          files: [
            ModelFile(
              url: 'https://example.com/model.onnx',
              filename: 'model.onnx',
            ),
          ],
          sizeMb: 300,
        );

        final jsonString = original.toJsonString();
        final deserialized = ModelDescriptor.fromJsonString(jsonString);

        expect(deserialized.id, original.id);
        expect(deserialized.type, original.type);
        expect(deserialized.toJsonString(), jsonString);
      });

      test('fromJson handles all SttModelType values', () {
        for (final type in SttModelType.values) {
          final json = {
            'id': 'test',
            'name': 'Test',
            'type': type.name,
            'languages': ['en'],
            'files': [],
            'sizeMb': 100,
          };
          final descriptor = ModelDescriptor.fromJson(json);
          expect(descriptor.type, type);
        }
      });

      test('fromJson throws for invalid type', () {
        final json = {
          'id': 'test',
          'name': 'Test',
          'type': 'invalid_type',
          'languages': ['en'],
          'files': [],
          'sizeMb': 100,
        };
        expect(
          () => ModelDescriptor.fromJson(json),
          throwsA(isA<FormatException>()),
        );
      });
    });

    group('SttResult', () {
      test('toJson and fromJson round-trip with all fields', () {
        const original = SttResult(
          text: 'Hello world',
          inferenceTimeMs: 150.5,
          lang: 'en',
          confidence: 0.95,
          durationMs: 1000.0,
          emotion: 'neutral',
          events: ['speech_start', 'speech_end'],
        );

        final json = original.toJson();
        final deserialized = SttResult.fromJson(json);

        expect(deserialized.text, original.text);
        expect(deserialized.inferenceTimeMs, original.inferenceTimeMs);
        expect(deserialized.lang, original.lang);
        expect(deserialized.confidence, original.confidence);
        expect(deserialized.durationMs, original.durationMs);
        expect(deserialized.emotion, original.emotion);
        expect(deserialized.events, original.events);
      });

      test('toJson and fromJson with minimal fields', () {
        const original = SttResult(
          text: 'Test',
          inferenceTimeMs: 100.0,
        );

        final json = original.toJson();
        final deserialized = SttResult.fromJson(json);

        expect(deserialized.text, original.text);
        expect(deserialized.inferenceTimeMs, original.inferenceTimeMs);
        expect(deserialized.lang, isNull);
        expect(deserialized.confidence, isNull);
        expect(deserialized.durationMs, isNull);
        expect(deserialized.emotion, isNull);
        expect(deserialized.events, isEmpty);
      });

      test('toJsonString and fromJsonString round-trip', () {
        const original = SttResult(
          text: 'Test transcription',
          inferenceTimeMs: 200.0,
          lang: 'en',
        );

        final jsonString = original.toJsonString();
        final deserialized = SttResult.fromJsonString(jsonString);

        expect(deserialized.text, original.text);
        expect(deserialized.inferenceTimeMs, original.inferenceTimeMs);
        expect(deserialized.lang, original.lang);
      });
    });

    group('SttConfig', () {
      test('toJson and fromJson round-trip with default values', () {
        const original = SttConfig();

        final json = original.toJson();
        final deserialized = SttConfig.fromJson(json);

        expect(deserialized.sampleRate, original.sampleRate);
        expect(deserialized.chunkSize, original.chunkSize);
        expect(deserialized.numChannels, original.numChannels);
        expect(deserialized.numThreads, original.numThreads);
        expect(
            deserialized.useParakeetForHighEnd, original.useParakeetForHighEnd);
        expect(deserialized.useVAD, original.useVAD);
        expect(deserialized.maxModelSizeMB, original.maxModelSizeMB);
        expect(
            deserialized.languageCacheDuration, original.languageCacheDuration);
        expect(deserialized.maxLoadedModels, original.maxLoadedModels);
      });

      test('toJson and fromJson with custom values', () {
        const original = SttConfig(
          sampleRate: 44100,
          chunkSize: 4096,
          numChannels: 2,
          numThreads: 8,
          useParakeetForHighEnd: false,
          useVAD: false,
          maxModelSizeMB: 1000,
          languageCacheDuration: Duration(minutes: 10),
          maxLoadedModels: 4,
        );

        final json = original.toJson();
        final deserialized = SttConfig.fromJson(json);

        expect(deserialized.sampleRate, original.sampleRate);
        expect(deserialized.chunkSize, original.chunkSize);
        expect(deserialized.numChannels, original.numChannels);
        expect(deserialized.numThreads, original.numThreads);
        expect(
            deserialized.useParakeetForHighEnd, original.useParakeetForHighEnd);
        expect(deserialized.useVAD, original.useVAD);
        expect(deserialized.maxModelSizeMB, original.maxModelSizeMB);
        expect(
            deserialized.languageCacheDuration, original.languageCacheDuration);
        expect(deserialized.maxLoadedModels, original.maxLoadedModels);
      });

      test('toJsonString and fromJsonString round-trip', () {
        const original = SttConfig(
          sampleRate: 16000,
          numThreads: 2,
        );

        final jsonString = original.toJsonString();
        final deserialized = SttConfig.fromJsonString(jsonString);

        expect(deserialized.sampleRate, original.sampleRate);
        expect(deserialized.numThreads, original.numThreads);
      });

      test('fromJson uses default values for missing fields', () {
        final json = <String, dynamic>{'sampleRate': 22050};
        final config = SttConfig.fromJson(json);

        expect(config.sampleRate, 22050);
        expect(config.chunkSize, 1600); // default
        expect(config.numThreads, 4); // default
      });
    });
  });
}
