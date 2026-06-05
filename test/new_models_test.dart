import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'package:stt_flutter/src/engines/engine_factory.dart';
import 'package:stt_flutter/src/engines/sensevoice/sensevoice_engine.dart';

void main() {
  group('Default model registrations', () {
    test('SenseVoice Small is registered with declared sizes and SHAs', () {
      final m = ModelRegistry.get('sensevoice-small-zh-en-ja-ko-yue');
      expect(m.type, SttModelType.sensevoice);
      expect(m.languages, contains('zh'));
      expect(m.languages, contains('en'));
      expect(m.languages, contains('ja'));
      expect(m.languages, contains('ko'));
      expect(m.languages, contains('yue'));
      expect(m.files.length, 2);
      final modelFile = m.files.firstWhere((f) => f.filename == 'model.int8.onnx');
      expect(modelFile.sizeBytes, 237115547);
      expect(modelFile.sha256,
          '12ca1a2ae7ecf3e0019ef2822307ee0b5cadc9196569e379b4c4026f8205276d');
    });

    test('Omnilingual 300M is registered', () {
      final m = ModelRegistry.get('omnilingual-300m-ctc-1600lang');
      expect(m.type, SttModelType.omnilingual);
      expect(m.files.length, 2);
      final modelFile = m.files.firstWhere((f) => f.filename == 'model.onnx');
      expect(modelFile.sizeBytes, 1304579963);
      expect(modelFile.sha256,
          '32432a0afa53180d08553de33baec1a36a43141bded193a26bcf2bed0bcd9c13');
    });

    test('Omnilingual 1B is registered with external weights', () {
      final m = ModelRegistry.get('omnilingual-1b-ctc-1600lang');
      expect(m.type, SttModelType.omnilingual);
      final weights = m.files.firstWhere((f) => f.filename == 'model.weights');
      expect(weights.sizeBytes, 3900260688);
      expect(weights.sha256,
          'd0954ba0b14d84b68b375989b56c00f1c86390d585a54bfddac77465adca1afd');
    });

    test('Qwen3-ASR 0.6B is registered with files in a tokenizer/ subdir', () {
      final m = ModelRegistry.get('qwen3-asr-0.6b-int8');
      expect(m.type, SttModelType.qwen3asr);
      expect(
        m.files.map((f) => f.filename).toSet(),
        {
          'conv_frontend.onnx',
          'encoder.int8.onnx',
          'decoder.int8.onnx',
          'tokenizer/tokenizer_config.json',
          'tokenizer/merges.txt',
          'tokenizer/vocab.json',
        },
      );
    });
  });

  group('EngineFactory dispatch', () {
    test('SenseVoice descriptor maps to SenseVoiceInferenceEngine', () {
      final m = ModelRegistry.get('sensevoice-small-zh-en-ja-ko-yue');
      final engine = createEngine(m);
      expect(engine, isA<SenseVoiceInferenceEngine>());
      expect(engine.supportsExplicitLanguage, true);
    });

    test('Omnilingual descriptor maps to OmnilingualInferenceEngine', () {
      final m = ModelRegistry.get('omnilingual-300m-ctc-1600lang');
      final engine = createEngine(m);
      expect(engine.runtimeType.toString(), 'OmnilingualInferenceEngine');
      expect(engine.supportsExplicitLanguage, false);
    });

    test('Qwen3-ASR descriptor maps to Qwen3AsrInferenceEngine', () {
      final m = ModelRegistry.get('qwen3-asr-0.6b-int8');
      final engine = createEngine(m);
      expect(engine.runtimeType.toString(), 'Qwen3AsrInferenceEngine');
      expect(engine.supportsExplicitLanguage, true);
    });
  });

  group('parseSenseVoiceText', () {
    test('strips language, emotion, event, and text-norm tags', () {
      final r = parseSenseVoiceText('<|en|><|HAPPY|><|Speech|><|withitn|>hello');
      expect(r.text, 'hello');
      expect(r.emotion, 'happy');
      expect(r.events, ['Speech']);
    });

    test('handles multiple events', () {
      final r = parseSenseVoiceText(
        '<|zh|><|NEUTRAL|><|Speech|><|BGM|><|withitn|>你好',
      );
      expect(r.text, '你好');
      expect(r.emotion, 'neutral');
      expect(r.events, contains('Speech'));
      expect(r.events, contains('BGM'));
    });

    test('returns empty text for empty input', () {
      final r = parseSenseVoiceText('');
      expect(r.text, '');
      expect(r.emotion, isNull);
      expect(r.events, isEmpty);
    });

    test('extracts emotion from a known set', () {
      final r = parseSenseVoiceText(
        '<|en|><|SAD|><|Speech|><|woitn|>goodbye',
      );
      expect(r.text, 'goodbye');
      expect(r.emotion, 'sad');
    });

    test('emotion is null when no recognized emotion tag is present', () {
      final r = parseSenseVoiceText(
        '<|en|><|Speech|><|withitn|>something else',
      );
      expect(r.emotion, isNull);
      expect(r.events, ['Speech']);
    });
  });
}
