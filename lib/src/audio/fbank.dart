import 'dart:math';
import 'dart:typed_data';

class Fbank {
  static const int nFft = 512;
  static const int hopLength = 160;
  static const int winLength = 400;
  static const int nFreqBins = nFft ~/ 2 + 1;

  final int nMels;
  final Float64List _hannWindow;
  final Float64List _melFilterbank;
  final double lowHz;
  final double highHz;

  Fbank({this.nMels = 80, this.lowHz = 0.0, this.highHz = 8000.0, int sampleRate = 16000})
      : _hannWindow = _createWindow(sampleRate),
        _melFilterbank = _createMelFilterbank(nMels, lowHz, highHz, sampleRate);

  static Float64List _createWindow(int sampleRate) {
    final size = (25.0 / 1000 * sampleRate).round(); // 25ms = 400 at 16kHz
    final w = Float64List(size);
    for (int i = 0; i < size; i++) {
      w[i] = 0.54 - 0.46 * cos(2 * pi * i / (size - 1)); // Hamming
    }
    return w;
  }

  static Float64List _createMelFilterbank(int nMels, double lowHz, double highHz, int sampleRate) {
    final fb = Float64List(nMels * nFreqBins);
    final lowMel = _hzToMel(max(lowHz, 0.0));
    final highMel = _hzToMel(min(highHz, sampleRate / 2));
    final spacing = (highMel - lowMel) / (nMels + 1);

    final centers = Float64List(nMels + 2);
    for (int i = 0; i < nMels + 2; i++) {
      centers[i] = _melToHz(lowMel + i * spacing);
    }

    for (int i = 0; i < nMels; i++) {
      final start = centers[i];
      final peak = centers[i + 1];
      final end = centers[i + 2];
      for (int j = 0; j < nFreqBins; j++) {
        final hz = j * sampleRate / nFft;
        if (hz >= start && hz <= peak) {
          fb[i * nFreqBins + j] = (hz - start) / (peak - start);
        } else if (hz > peak && hz <= end) {
          fb[i * nFreqBins + j] = (end - hz) / (end - peak);
        }
      }
    }
    return fb;
  }

  static double _hzToMel(double hz) => 2595 * log(1 + hz / 700);
  static double _melToHz(double mel) => 700 * (exp(mel / 2595) - 1);

  Float64List compute(Float32List samples) {
    final winSize = _hannWindow.length;
    final nFrames = max(1, (samples.length - winSize) ~/ hopLength + 1);
    final n = min(nFrames, 3000);
    final mel = Float64List(n * nMels);

    // Pre-emphasis
    final emphasized = Float32List(samples.length);
    emphasized[0] = samples[0];
    for (int i = 1; i < samples.length; i++) {
      emphasized[i] = samples[i] - 0.97 * samples[i - 1];
    }

    final padded = Float64List(nFft);

    for (int t = 0; t < n; t++) {
      for (int i = 0; i < winSize; i++) {
        final idx = t * hopLength + i;
        padded[i] = (idx < emphasized.length ? emphasized[idx] : 0.0) * _hannWindow[i];
      }
      for (int i = winSize; i < nFft; i++) {
        padded[i] = 0.0;
      }

      // Power spectrum
      for (int k = 0; k < nFreqBins; k++) {
        double real = 0, imag = 0;
        for (int i = 0; i < nFft; i++) {
          final angle = 2 * pi * k * i / nFft;
          real += padded[i] * cos(angle);
          imag -= padded[i] * sin(angle);
        }
        padded[k] = real * real + imag * imag; // reuse padded for spectrum
      }

      // Mel filterbank + log
      for (int m = 0; m < nMels; m++) {
        double sum = 0;
        final fbOffset = m * nFreqBins;
        for (int k = 0; k < nFreqBins; k++) {
          sum += padded[k] * _melFilterbank[fbOffset + k];
        }
        mel[t * nMels + m] = log(max(sum, 1e-10)) / ln10;
      }
    }

    // Global CMVN: normalize each mel band to zero mean, unit variance over time
    for (int m = 0; m < nMels; m++) {
      double sum = 0, sqSum = 0;
      for (int t = 0; t < n; t++) {
        final v = mel[t * nMels + m];
        sum += v;
        sqSum += v * v;
      }
      final mean = sum / n;
      final std = sqrt(max(sqSum / n - mean * mean, 1e-10));
      for (int t = 0; t < n; t++) {
        mel[t * nMels + m] = (mel[t * nMels + m] - mean) / std;
      }
    }

    return mel;
  }
}
