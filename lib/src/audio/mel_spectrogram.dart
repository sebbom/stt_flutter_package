import 'dart:math';
import 'dart:typed_data';

/// Log-Mel spectrogram for Whisper models.
///
/// Uses 400-sample Hann window, 160-sample hop, configurable mel bins.
/// Output shape: [1, nMels, 3000] as flattened Float64List (frames * nMels).
///
/// Safe to call from [Isolate.run] (pass a closure capturing the instance).
class MelSpectrogram {
  final int nMels;
  final Float64List _melFilterbank;

  static const int nFft = 400;
  static const int hopLength = 160;
  static const int maxFrames = 3000;
  static const int sampleRate = 16000;

  static final Float64List _hannWindow = _createHannWindow();

  MelSpectrogram({this.nMels = 80})
      : _melFilterbank = _createMelFilterbank(nMels);

  int get nFreqBins => nFft ~/ 2 + 1;

  static Float64List _createHannWindow() {
    final w = Float64List(nFft);
    for (int i = 0; i < nFft; i++) {
      w[i] = 0.5 * (1 - cos(2 * pi * i / (nFft - 1)));
    }
    return w;
  }

  static Float64List _createMelFilterbank(int nMels) {
    final nFreqBins = nFft ~/ 2 + 1;
    final fb = Float64List(nMels * nFreqBins);
    final lowMel = _hzToMel(0);
    final highMel = _hzToMel(sampleRate / 2);
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

  /// Compute log-mel spectrogram from 16kHz mono PCM samples.
  /// Returns flattened Float64List (frames * nMels).
  Float64List compute(Float32List samples) {
    final nFreqBins = nFft ~/ 2 + 1;
    final nFrames = max(1, (samples.length - nFft) ~/ hopLength + 1);

    // STFT magnitude spectrum
    final stft = Float64List(nFrames * nFreqBins);
    final padded = Float64List(nFft);

    for (int t = 0; t < nFrames; t++) {
      for (int i = 0; i < nFft; i++) {
        final idx = t * hopLength + i;
        padded[i] = (idx < samples.length ? samples[idx] : 0.0) * _hannWindow[i];
      }

      for (int k = 0; k < nFreqBins; k++) {
        double real = 0, imag = 0;
        for (int i = 0; i < nFft; i++) {
          final angle = 2 * pi * k * i / nFft;
          real += padded[i] * cos(angle);
          imag -= padded[i] * sin(angle);
        }
        stft[t * nFreqBins + k] = real * real + imag * imag;
      }
    }

    // Mel filterbank
    final n = min(nFrames, maxFrames);
    final mel = Float64List(n * nMels);

    for (int t = 0; t < n; t++) {
      for (int m = 0; m < nMels; m++) {
        double sum = 0;
        final fbOffset = m * nFreqBins;
        final stftOffset = t * nFreqBins;
        for (int k = 0; k < nFreqBins; k++) {
          sum += stft[stftOffset + k] * _melFilterbank[fbOffset + k];
        }
        mel[t * nMels + m] = max(sum, 1e-10);
      }
    }

    // log10
    for (int i = 0; i < mel.length; i++) {
      mel[i] = log(mel[i]) / ln10;
    }

    // Per-band zero-mean unit-variance normalization
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

  /// Convenience for default 80-band computation (backward compat).
  static Float64List compute80(Float32List samples) {
    return MelSpectrogram().compute(samples);
  }
}
