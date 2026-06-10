import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

// Mock engine for testing
class MockInferenceEngine implements InferenceEngine {
  bool _disposed = false;

  @override
  bool get supportsExplicitLanguage => true;

  @override
  Set<String> get supportedLanguages => {'en', 'fr', 'de'};

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    // Mock loading - no actual model loading needed for tests
  }

  @override
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
    Map<String, dynamic>? options,
  }) async {
    if (_disposed) throw StateError('Engine disposed');
    token?.throwIfCancelled();

    // Return mock result based on input
    return SttResult(
      text: 'Mock transcription of: ${audio.samples.length} samples',
      inferenceTimeMs: 100.0,
      lang: language ?? 'en',
      confidence: 0.95,
      durationMs: (audio.samples.length / audio.sampleRate) * 1000.0,
    );
  }

  @override
  Future<void> dispose() async {
    _disposed = true;
  }
}

void main() {
  group('STT Pipeline Integration Tests', () {
    late SttFlutter stt;
    late ModelDescriptor model;

    setUp(() {
      model = ModelDescriptor(
        id: 'mock-test-model',
        name: 'Mock Test Model',
        type: SttModelType.whisper,
        languages: ['en', 'fr', 'de'],
        files: [
          ModelFile(
            url: 'https://example.com/mock-encoder.onnx',
            filename: 'mock-encoder.onnx',
          ),
          ModelFile(
            url: 'https://example.com/mock-decoder.onnx',
            filename: 'mock-decoder.onnx',
          ),
        ],
        sizeMb: 1,
      );
    });

    tearDown(() async {
      try {
        await stt.dispose();
      } catch (_) {
        // Ignore errors during cleanup
      }
    });

    group('Initialization and Transcription', () {
      test('can initialize with mock engine and transcribe buffer', () async {
        // Create STT instance with mock engine
        final mockEngine = MockInferenceEngine();
        stt = SttFlutter.withEngine(
          model: model,
          engine: mockEngine,
          language: 'en',
        );

        // Create test audio buffer
        final samples = Float32List.fromList(
          List.generate(16000, (i) => (i % 100) / 100.0 - 0.5),
        );

        // Transcribe
        final result = await stt.transcribeBuffer(
          samples,
          16000,
          language: 'en',
        );

        // Verify result
        expect(result.text, contains('16000 samples'));
        expect(result.lang, 'en');
        expect(result.inferenceTimeMs, 100.0);
        expect(result.confidence, 0.95);
        expect(result.durationMs, closeTo(1000.0, 0.1));
      });

      test('can transcribe with different languages', () async {
        final mockEngine = MockInferenceEngine();
        stt = SttFlutter.withEngine(
          model: model,
          engine: mockEngine,
        );

        final samples = Float32List.fromList(
          List.generate(8000, (i) => (i % 100) / 100.0 - 0.5),
        );

        // Test with French
        final resultFr = await stt.transcribeBuffer(
          samples,
          16000,
          language: 'fr',
        );
        expect(resultFr.lang, 'fr');

        // Test with German
        final resultDe = await stt.transcribeBuffer(
          samples,
          16000,
          language: 'de',
        );
        expect(resultDe.lang, 'de');
      });

      test('can transcribe with default language', () async {
        final mockEngine = MockInferenceEngine();
        stt = SttFlutter.withEngine(
          model: model,
          engine: mockEngine,
          language: 'fr',
        );

        final samples = Float32List.fromList(
          List.generate(8000, (i) => (i % 100) / 100.0 - 0.5),
        );

        // Transcribe without specifying language - should use default
        final result = await stt.transcribeBuffer(samples, 16000);
        expect(result.lang, 'fr');
      });

      test('can transcribe with resampling', () async {
        final mockEngine = MockInferenceEngine();
        stt = SttFlutter.withEngine(
          model: model,
          engine: mockEngine,
        );

        // Create audio at 44100 Hz
        final samples = Float32List.fromList(
          List.generate(44100, (i) => (i % 100) / 100.0 - 0.5),
        );

        // Should resample to 16kHz internally
        final result = await stt.transcribeBuffer(
          samples,
          44100,
          language: 'en',
        );

        expect(result.text, contains('samples'));
        // Duration should be ~1 second (44100 samples at 44100 Hz)
        expect(result.durationMs, closeTo(1000.0, 1.0));
      });

      test('cancellation token stops transcription', () async {
        final mockEngine = MockInferenceEngine();
        stt = SttFlutter.withEngine(
          model: model,
          engine: mockEngine,
        );

        final token = CancellationToken();
        token.cancel();

        final samples = Float32List.fromList(
          List.generate(16000, (i) => (i % 100) / 100.0 - 0.5),
        );

        // Should throw because token is already cancelled
        expect(
          () => stt.transcribeBuffer(samples, 16000, token: token),
          throwsA(isA<OperationCancelledException>()),
        );
      });
    });

    group('Model Registry Serialization Integration', () {
      test('can serialize and deserialize built-in model descriptor', () {
        final descriptor = ModelRegistry.get('whisper-tiny');

        // Serialize to JSON
        final jsonString = descriptor.toJsonString();

        // Deserialize back
        final deserialized = ModelDescriptor.fromJsonString(jsonString);

        expect(deserialized.id, descriptor.id);
        expect(deserialized.name, descriptor.name);
        expect(deserialized.type, descriptor.type);
        expect(deserialized.languages, descriptor.languages);
        expect(deserialized.files.length, descriptor.files.length);
        expect(deserialized.sizeMb, descriptor.sizeMb);
      });

      test('can serialize and deserialize all built-in models', () {
        final models = ModelRegistry.available();

        for (final model in models) {
          final json = model.toJson();
          final deserialized = ModelDescriptor.fromJson(json);

          expect(deserialized.id, model.id);
          expect(deserialized.type, model.type);
          expect(deserialized.name, model.name);
        }
      });
    });

    group('SttResult Serialization Integration', () {
      test('can create and serialize SttResult from transcription', () async {
        final mockEngine = MockInferenceEngine();
        stt = SttFlutter.withEngine(
          model: model,
          engine: mockEngine,
          language: 'en',
        );

        final samples = Float32List.fromList(
          List.generate(16000, (i) => (i % 100) / 100.0 - 0.5),
        );

        final result = await stt.transcribeBuffer(samples, 16000);

        // Serialize the result
        final jsonString = result.toJsonString();

        // Deserialize back
        final deserialized = SttResult.fromJsonString(jsonString);

        expect(deserialized.text, result.text);
        expect(deserialized.inferenceTimeMs, result.inferenceTimeMs);
        expect(deserialized.lang, result.lang);
        expect(deserialized.confidence, result.confidence);
        expect(deserialized.durationMs, result.durationMs);
      });
    });

    group('SttConfig Serialization Integration', () {
      test('can serialize and deserialize default SttConfig', () {
        const config = SttConfig();

        final jsonString = config.toJsonString();
        final deserialized = SttConfig.fromJsonString(jsonString);

        expect(deserialized.sampleRate, config.sampleRate);
        expect(deserialized.chunkSize, config.chunkSize);
        expect(deserialized.numChannels, config.numChannels);
        expect(deserialized.numThreads, config.numThreads);
        expect(
            deserialized.useParakeetForHighEnd, config.useParakeetForHighEnd);
        expect(deserialized.useVAD, config.useVAD);
        expect(deserialized.maxModelSizeMB, config.maxModelSizeMB);
        expect(
            deserialized.languageCacheDuration, config.languageCacheDuration);
        expect(deserialized.maxLoadedModels, config.maxLoadedModels);
      });

      test('can serialize and deserialize custom SttConfig', () {
        const config = SttConfig(
          sampleRate: 44100,
          chunkSize: 4096,
          numThreads: 8,
          maxModelSizeMB: 1000,
        );

        final jsonString = config.toJsonString();
        final deserialized = SttConfig.fromJsonString(jsonString);

        expect(deserialized.sampleRate, config.sampleRate);
        expect(deserialized.chunkSize, config.chunkSize);
        expect(deserialized.numThreads, config.numThreads);
        expect(deserialized.maxModelSizeMB, config.maxModelSizeMB);
      });
    });
  });
}
