import 'dart:math';
import 'dart:typed_data';

/// Log-Mel spectrogram for Whisper models.
///
/// Uses 400-sample Hann window, 160-sample hop, 80 mel bins.
/// Output shape: [1, 80, 3000] as flattened Float64List (frames * nMels).
///
/// Designed to be stateless — safe to call from [Isolate.run].
class MelSpectrogram {
  static const int nMels = 80;
  static const int nFft = 400;
  static const int hopLength = 160;
  static const int maxFrames = 3000;
  static const int sampleRate = 16000;
  static const int nFreqBins = nFft ~/ 2 + 1;

  static final Float64List _hannWindow = _createHannWindow();
  static final Float64List _melFilterbank = _createMelFilterbank();

  static Float64List _createHannWindow() {
    final w = Float64List(nFft);
    for (int i = 0; i < nFft; i++) {
      w[i] = 0.5 * (1 - cos(2 * pi * i / (nFft - 1)));
    }
    return w;
  }

  static Float64List _createMelFilterbank() {
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
  /// Returns flattened Float64List (frames * nMels) transposed to [frames, nMels].
  static Float64List compute(Float32List samples) {
    final nFrames = max(1, (samples.length - nFft) ~/ hopLength + 1);
    final nFreqBins = nFft ~/ 2 + 1;

    // STFT magnitude spectrum
    final stft = Float64List(nFrames * nFreqBins);
    final padded = Float64List(nFft);

    for (int t = 0; t < nFrames; t++) {
      // Frame extraction + windowing
      for (int i = 0; i < nFft; i++) {
        final idx = t * hopLength + i;
        padded[i] = (idx < samples.length ? samples[idx] : 0.0) * _hannWindow[i];
      }

      // DFT magnitude (simplified direct computation)
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

    // Mel filterbank + log
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
        mel[t * nMels + m] = log(max(sum, 1e-10));
      }
    }

    return mel;
  }
}
