import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'package:stt_flutter/src/audio/audio_processor.dart';

void main() {
  group('AudioProcessor', () {
    test('loadWav parses a valid WAV file', () async {
      final buffer = await AudioProcessor.loadWav('test/fixtures/hello_en.wav');
      expect(buffer.sampleRate, 16000);
      expect(buffer.length, 16000);
      expect(buffer.durationSec, closeTo(1.0, 0.01));
    });

    test('loadWav returns samples in [-1, 1] range', () async {
      final buffer = await AudioProcessor.loadWav('test/fixtures/hello_en.wav');
      for (final s in buffer.samples) {
        expect(s, greaterThanOrEqualTo(-1.0));
        expect(s, lessThanOrEqualTo(1.0));
      }
    });

    test('resampleSync returns same buffer when rate matches', () {
      final input = AudioBuffer(
        samples: Float32List.fromList([0.0, 0.5, 1.0, 0.5, 0.0]),
        sampleRate: 16000,
      );
      final result = AudioProcessor.resampleSync(input);
      expect(result.sampleRate, 16000);
      expect(result.length, 5);
      expect(result.samples[0], 0.0);
      expect(result.samples[2], 1.0);
    });

    test('resampleSync handles downsampling', () {
      final input = AudioBuffer(
        samples: Float32List.fromList(List.generate(48, (i) => i / 47.0)),
        sampleRate: 48000,
      );
      final result = AudioProcessor.resampleSync(input, targetRate: 16000);
      expect(result.sampleRate, 16000);
      expect(result.length, closeTo(16, 1));
    });

    test('resampleSync handles upsampling', () {
      final input = AudioBuffer(
        samples: Float32List.fromList(List.generate(8, (i) => i / 7.0)),
        sampleRate: 8000,
      );
      final result = AudioProcessor.resampleSync(input, targetRate: 16000);
      expect(result.sampleRate, 16000);
      expect(result.length, closeTo(16, 1));
    });

    test('loadWav throws on non-existent file', () async {
      expect(
        () => AudioProcessor.loadWav('test/fixtures/nonexistent.wav'),
        throwsA(isA<FileSystemException>()),
      );
    });
  });
}
