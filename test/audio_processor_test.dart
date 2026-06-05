import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

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

  group('AudioProcessor preprocessing', () {
    AudioBuffer silent() => AudioBuffer(
          samples: Float32List.fromList(List.filled(1000, 0.0)),
          sampleRate: 16000,
        );

    AudioBuffer constant(double v) => AudioBuffer(
          samples: Float32List.fromList(List.filled(1000, v)),
          sampleRate: 16000,
        );

    double peakOf(AudioBuffer b) {
      var p = 0.0;
      for (final s in b.samples) {
        final a = s.abs();
        if (a > p) p = a;
      }
      return p;
    }

    double rmsOf(AudioBuffer b) {
      double s = 0;
      for (final v in b.samples) {
        s += v * v;
      }
      return s == 0 ? 0 : math.sqrt(s / b.samples.length);
    }

    test('applyGain multiplies and clamps to [-1, 1]', () {
      final b = AudioBuffer(
        samples: Float32List.fromList([0.1, 0.5, -0.7, 0.9, -0.95]),
        sampleRate: 16000,
      );
      final g = AudioProcessor.applyGain(b, 2.0);
      expect(g.samples[0], closeTo(0.2, 1e-6));
      expect(g.samples[1], closeTo(1.0, 1e-6));
      expect(g.samples[2], closeTo(-1.0, 1e-6));
      expect(g.samples[3], closeTo(1.0, 1e-6));
      expect(g.samples[4], closeTo(-1.0, 1e-6));
    });

    test('applyGain is a no-op when gain=1', () {
      final b = AudioBuffer(
        samples: Float32List.fromList([0.1, -0.2]),
        sampleRate: 16000,
      );
      expect(identical(AudioProcessor.applyGain(b, 1.0), b), isTrue);
    });

    test('peakNormalize scales quiet buffer to target peak', () {
      final b = AudioBuffer(
        samples: Float32List.fromList([0.01, -0.005, 0.02, -0.015, 0.008]),
        sampleRate: 16000,
      );
      final n = AudioProcessor.peakNormalize(b);
      expect(peakOf(n), closeTo(0.95, 1e-6));
      for (final s in n.samples) {
        expect(s.abs(), lessThanOrEqualTo(1.0));
      }
    });

    test('peakNormalize is a no-op when already at/above target', () {
      final b = AudioBuffer(
        samples: Float32List.fromList([0.96, -0.97, 0.95]),
        sampleRate: 16000,
      );
      expect(identical(AudioProcessor.peakNormalize(b), b), isTrue);
    });

    test('peakNormalize on silent buffer returns input unchanged', () {
      final s = silent();
      expect(identical(AudioProcessor.peakNormalize(s), s), isTrue);
    });

    test('rmsNormalize scales buffer to target RMS', () {
      final b = AudioBuffer(
        samples: Float32List.fromList([0.05, -0.05, 0.05, -0.05]),
        sampleRate: 16000,
      );
      final n = AudioProcessor.rmsNormalize(b, target: 0.2);
      expect(rmsOf(n), closeTo(0.2, 1e-6));
    });

    test('rmsNormalize on silent buffer returns input unchanged', () {
      final s = silent();
      expect(identical(AudioProcessor.rmsNormalize(s), s), isTrue);
    });

    test('highPass removes DC offset', () {
      final b = constant(0.5);
      final hp = AudioProcessor.highPass(b);
      final tail =
          hp.samples.sublist(hp.samples.length - 100).reduce((a, c) => a + c) /
              100;
      expect(tail.abs(), lessThan(1e-3),
          reason: 'mean of the last 100 samples should be ~0 after DC removal');
    });

    test('highPass passes through high-frequency content', () {
      final sr = 16000;
      final n = sr;
      final samples = Float32List(n);
      for (var i = 0; i < n; i++) {
        samples[i] = math.sin(2 * math.pi * 1000 * i / sr);
      }
      final b = AudioBuffer(samples: samples, sampleRate: sr);
      final hp = AudioProcessor.highPass(b, cutoffHz: 80);
      expect(peakOf(hp), greaterThan(0.9),
          reason: 'a 1 kHz sine should be preserved by an 80 Hz HP filter');
    });

    test('highPass attenuates low-frequency content', () {
      final sr = 16000;
      final n = sr;
      final samples = Float32List(n);
      for (var i = 0; i < n; i++) {
        samples[i] = math.sin(2 * math.pi * 30 * i / sr);
      }
      final b = AudioBuffer(samples: samples, sampleRate: sr);
      final hp = AudioProcessor.highPass(b, cutoffHz: 80);
      expect(peakOf(hp), lessThan(0.5),
          reason: 'a 30 Hz sine should be strongly attenuated by an 80 Hz HP');
    });

    test('applyPreprocess chains high-pass → gain → peak-normalize', () {
      final b = AudioBuffer(
        samples: Float32List.fromList(
          List.generate(8000, (i) {
            final t = i / 16000.0;
            return 0.01 * math.sin(2 * math.pi * 1000 * t) + 0.05;
          }),
        ),
        sampleRate: 16000,
      );
      final cfg = const PreprocessConfig(
        gain: 2.0,
        normalize: NormalizeMode.peak,
        highPass: true,
      );
      final out = AudioProcessor.applyPreprocess(b, cfg);
      expect(peakOf(out), closeTo(0.95, 1e-6),
          reason: 'peak must hit target after high-pass + gain + normalize');
      final mean = out.samples.reduce((a, c) => a + c) / out.samples.length;
      expect(mean.abs(), lessThan(0.02),
          reason: 'DC offset should be removed by the high-pass stage');
    });

    test('loadWav with PreprocessConfig applies peak normalization', () async {
      final buffer = await AudioProcessor.loadWav(
        'test/fixtures/hello_en.wav',
        preprocess: const PreprocessConfig(normalize: NormalizeMode.peak),
      );
      expect(peakOf(buffer), closeTo(0.95, 0.05));
    });

    test('loadWav with default config is byte-identical to before', () async {
      final a = await AudioProcessor.loadWav('test/fixtures/hello_en.wav');
      final b = await AudioProcessor.loadWav(
        'test/fixtures/hello_en.wav',
        preprocess: PreprocessConfig.none,
      );
      expect(b.sampleRate, a.sampleRate);
      expect(b.length, a.length);
      for (var i = 0; i < a.length; i++) {
        expect(b.samples[i], closeTo(a.samples[i], 1e-9));
      }
    });
  });
}

