import 'dart:math';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

void main() {
  group('MelSpectrogram', () {
    test('compute returns correct shape for 1s audio', () {
      // 1 second at 16kHz
      final samples = Float32List(16000);
      for (int i = 0; i < 16000; i++) {
        samples[i] = 0.3 * sin(2 * pi * 440 * i / 16000);
      }

      final mel = MelSpectrogram.compute(samples);
      // 1s @ 16kHz hop 160 → 98 frames × 80 bins = 7840
      expect(mel.length % MelSpectrogram.nMels, 0);
      expect(mel.length ~/ MelSpectrogram.nMels, greaterThan(0));
      expect(mel.length, lessThanOrEqualTo(MelSpectrogram.maxFrames * MelSpectrogram.nMels));
    });

    test('compute handles empty audio gracefully', () {
      final samples = Float32List(0);
      final mel = MelSpectrogram.compute(samples);
      // Should produce at least 1 frame of silence
      expect(mel.length, MelSpectrogram.nMels);
    });

    test('compute produces finite values', () {
      final samples = Float32List(48000);
      for (int i = 0; i < 48000; i++) {
        samples[i] = 0.1 * sin(2 * pi * 1000 * i / 16000);
      }

      final mel = MelSpectrogram.compute(samples);
      for (int i = 0; i < mel.length; i++) {
        expect(mel[i].isNaN, false);
        expect(mel[i].isInfinite, false);
      }
    });

    test('compute gives higher energy at the expected frequency bin', () {
      // 1kHz sine wave — should excite mel band around 1000 Hz
      final samples = Float32List(16000);
      for (int i = 0; i < 16000; i++) {
        samples[i] = 0.5 * sin(2 * pi * 1000 * i / 16000);
      }

      final mel = MelSpectrogram.compute(samples);

      // Average energy per mel band across all frames
      final avgBands = Float64List(MelSpectrogram.nMels);
      final nFrames = mel.length ~/ MelSpectrogram.nMels;
      for (int f = 0; f < nFrames; f++) {
        for (int b = 0; b < MelSpectrogram.nMels; b++) {
          avgBands[b] += mel[f * MelSpectrogram.nMels + b];
        }
      }
      for (int b = 0; b < MelSpectrogram.nMels; b++) {
        avgBands[b] /= nFrames;
      }

      // Find the band with maximum energy
      double maxEnergy = -1e10;
      int maxBand = 0;
      for (int b = 0; b < MelSpectrogram.nMels; b++) {
        if (avgBands[b] > maxEnergy) {
          maxEnergy = avgBands[b];
          maxBand = b;
        }
      }

      // 1kHz ~ mel band ~18 (mel scale: 2595*log10(1+1000/700) ≈ 1000 mel)
      // nMels=80, range 0-8kHz → each band ~100 mel → band ~10
      expect(maxBand, greaterThanOrEqualTo(5));
      expect(maxBand, lessThanOrEqualTo(30));
    });
  });
}
