import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import 'audio_buffer.dart';
import '../stt_logger.dart';

enum NormalizeMode { none, peak, rms }

/// Which sherpa-onnx denoiser to apply. The actual on-disk model files are
/// referenced by [PreprocessConfig.denoiserModelDir] + [denoiserModelFile].
enum DenoiserType { none, gtcrn, dpdfnet }

class PreprocessConfig {
  final double gain;
  final NormalizeMode normalize;
  final bool highPass;
  final double highPassCutoffHz;
  final double peakTarget;
  final double rmsTarget;

  /// Request live microphone noise suppression. The actual suppression is
  /// applied by an external Flutter plugin (e.g. `noise_suppression`); this
  /// flag is surfaced to consumers so they can wire the platform-side hook.
  final bool noiseSuppression;

  /// Directory containing the sherpa-onnx denoiser model files. When
  /// [denoiserType] is `gtcrn`, the loader looks for `model.onnx` inside this
  /// directory. When `dpdfnet`, it looks for `model.onnx` and may also
  /// require `model_post.onnx`. Empty disables denoising.
  final String? denoiserModelDir;

  /// Which sherpa-onnx denoiser family to use.
  final DenoiserType denoiserType;

  /// Number of threads to use for the denoiser session. 0 means
  /// "use engine default" (1).
  final int denoiserNumThreads;

  const PreprocessConfig({
    this.gain = 1.0,
    this.normalize = NormalizeMode.none,
    this.highPass = false,
    this.highPassCutoffHz = 80.0,
    this.peakTarget = 0.95,
    this.rmsTarget = 0.1,
    this.noiseSuppression = false,
    this.denoiserModelDir,
    this.denoiserType = DenoiserType.none,
    this.denoiserNumThreads = 0,
  });

  static const none = PreprocessConfig();

  bool get hasDenoiser =>
      denoiserType != DenoiserType.none &&
      denoiserModelDir != null &&
      denoiserModelDir!.isNotEmpty;

  bool get isNoOp =>
      gain == 1.0 &&
      normalize == NormalizeMode.none &&
      !highPass &&
      !hasDenoiser &&
      !noiseSuppression;
}

class AudioProcessor {
  static const int targetSampleRate = 16000;

  /// Load a WAV file, optionally apply preprocessing, and return a
  /// 16kHz mono Float32 AudioBuffer.
  static Future<AudioBuffer> loadWav(
    String path, {
    PreprocessConfig preprocess = PreprocessConfig.none,
  }) async {
    final file = File(path);
    final bytes = await file.readAsBytes();
    final buf = _parseWavBytes(bytes);
    if (preprocess.isNoOp) return buf;
    final afterDenoise = await _maybeDenoise(buf, preprocess);
    return applyPreprocess(afterDenoise, preprocess);
  }

  /// Apply a preprocessing pipeline in this order:
  ///   denoiser → high-pass → gain → normalize.
  /// Returns a new buffer; the input is not modified.
  static Future<AudioBuffer> applyPreprocessAsync(
    AudioBuffer buf,
    PreprocessConfig cfg,
  ) async {
    if (cfg.isNoOp) return buf;
    var b = await _maybeDenoise(buf, cfg);
    if (cfg.highPass) {
      b = highPass(b, cutoffHz: cfg.highPassCutoffHz);
    }
    if (cfg.gain != 1.0) {
      b = applyGain(b, cfg.gain);
    }
    switch (cfg.normalize) {
      case NormalizeMode.peak:
        b = peakNormalize(b, target: cfg.peakTarget);
      case NormalizeMode.rms:
        b = rmsNormalize(b, target: cfg.rmsTarget);
      case NormalizeMode.none:
        break;
    }
    return b;
  }

  /// Apply the synchronous portion of the preprocessing pipeline.
  /// The denoiser step is skipped (it requires async); use
  /// [applyPreprocessAsync] for the full pipeline.
  static AudioBuffer applyPreprocess(AudioBuffer buf, PreprocessConfig cfg) {
    if (cfg.isNoOp) return buf;
    var b = buf;
    if (cfg.highPass) {
      b = highPass(b, cutoffHz: cfg.highPassCutoffHz);
    }
    if (cfg.gain != 1.0) {
      b = applyGain(b, cfg.gain);
    }
    switch (cfg.normalize) {
      case NormalizeMode.peak:
        b = peakNormalize(b, target: cfg.peakTarget);
      case NormalizeMode.rms:
        b = rmsNormalize(b, target: cfg.rmsTarget);
      case NormalizeMode.none:
        break;
    }
    return b;
  }

  static Future<AudioBuffer> _maybeDenoise(
    AudioBuffer buf,
    PreprocessConfig cfg,
  ) async {
    if (!cfg.hasDenoiser) return buf;
    final dir = cfg.denoiserModelDir!;
    String? modelFile;
    if (cfg.denoiserType == DenoiserType.gtcrn) {
      modelFile = '$dir/model.onnx';
    } else if (cfg.denoiserType == DenoiserType.dpdfnet) {
      modelFile = '$dir/model.onnx';
    }
    if (modelFile == null || !File(modelFile).existsSync()) {
      SttLogger.w(
        'Denoiser model not found at $modelFile — skipping denoise step.',
      );
      return buf;
    }
    final config = OfflineSpeechDenoiserConfig(
      model: OfflineSpeechDenoiserModelConfig(
        gtcrn: cfg.denoiserType == DenoiserType.gtcrn
            ? OfflineSpeechDenoiserGtcrnModelConfig(model: modelFile)
            : const OfflineSpeechDenoiserGtcrnModelConfig(),
        dpdfnet: cfg.denoiserType == DenoiserType.dpdfnet
            ? OfflineSpeechDenoiserDpdfNetModelConfig(model: modelFile)
            : const OfflineSpeechDenoiserDpdfNetModelConfig(),
        numThreads: cfg.denoiserNumThreads > 0 ? cfg.denoiserNumThreads : 1,
        debug: false,
        provider: 'cpu',
      ),
    );
    final denoiser = OfflineSpeechDenoiser(config);
    try {
      final out = denoiser.run(samples: buf.samples, sampleRate: buf.sampleRate);
      if (out.samples.isEmpty) {
        SttLogger.w('Denoiser returned empty samples — skipping denoise step.');
        return buf;
      }
      return AudioBuffer(samples: out.samples, sampleRate: out.sampleRate);
    } finally {
      denoiser.free();
    }
  }

  /// Multiply every sample by [gain] and clamp to `[-1, 1]` to avoid wrap-around.
  static AudioBuffer applyGain(AudioBuffer buf, double gain) {
    if (gain == 1.0) return buf;
    final out = Float32List(buf.samples.length);
    for (var i = 0; i < buf.samples.length; i++) {
      final v = buf.samples[i] * gain;
      out[i] = v > 1.0 ? 1.0 : (v < -1.0 ? -1.0 : v);
    }
    return AudioBuffer(samples: out, sampleRate: buf.sampleRate);
  }

  /// Scale so `max|sample|` equals [target]. No-op if the buffer is silent
  /// (peak < 1e-9) or already at/above the target.
  static AudioBuffer peakNormalize(
    AudioBuffer buf, {
    double target = 0.95,
  }) {
    var peak = 0.0;
    for (final s in buf.samples) {
      final a = s.abs();
      if (a > peak) peak = a;
    }
    if (peak < 1e-9 || peak >= target) return buf;
    return applyGain(buf, target / peak);
  }

  /// Scale so RMS equals [target]. No-op if the buffer is silent.
  /// Output is clamped to `[-1, 1]`.
  static AudioBuffer rmsNormalize(
    AudioBuffer buf, {
    double target = 0.1,
  }) {
    if (buf.samples.isEmpty) return buf;
    double sumSq = 0.0;
    for (final s in buf.samples) {
      sumSq += s * s;
    }
    final rms = math.sqrt(sumSq / buf.samples.length);
    if (rms < 1e-9) return buf;
    return applyGain(buf, target / rms);
  }

  /// First-order IIR high-pass filter (RC → bilinear). Removes DC offset
  /// and attenuates frequencies below [cutoffHz] (default 80 Hz).
  static AudioBuffer highPass(
    AudioBuffer buf, {
    double cutoffHz = 80.0,
  }) {
    if (buf.samples.isEmpty) return buf;
    final dt = 1.0 / buf.sampleRate;
    final rc = 1.0 / (2.0 * math.pi * cutoffHz);
    final a = rc / (rc + dt);
    final out = Float32List(buf.samples.length);
    double yPrev = 0.0;
    double xPrev = 0.0;
    for (var i = 0; i < buf.samples.length; i++) {
      final x = buf.samples[i];
      final y = a * (yPrev + x - xPrev);
      out[i] = y;
      yPrev = y;
      xPrev = x;
    }
    return AudioBuffer(samples: out, sampleRate: buf.sampleRate);
  }

  /// Resample to [targetRate] Hz mono using linear interpolation.
  /// Lightweight enough to run on the main isolate, but for big buffers
  /// callers should prefer [Isolate.run].
  static AudioBuffer resampleSync(
    AudioBuffer input, {
    int targetRate = targetSampleRate,
  }) {
    if (input.sampleRate == targetRate) return input;
    final ratio = input.sampleRate / targetRate;
    final newLength = (input.length / ratio).round();
    final output = Float32List(newLength);
    final lastIdx = input.length - 1;

    for (int i = 0; i < newLength; i++) {
      final srcPos = i * ratio;
      final srcIdx = srcPos.floor();
      final frac = srcPos - srcIdx;
      final a = input.samples[srcIdx.clamp(0, lastIdx)];
      final b = input.samples[(srcIdx + 1).clamp(0, lastIdx)];
      output[i] = a * (1 - frac) + b * frac;
    }
    return AudioBuffer(samples: output, sampleRate: targetRate);
  }

  static AudioBuffer _parseWavBytes(Uint8List bytes) {
    if (bytes.length < 12 ||
        String.fromCharCodes(bytes.sublist(0, 4)) != 'RIFF' ||
        String.fromCharCodes(bytes.sublist(8, 12)) != 'WAVE') {
      throw const FormatException('Not a valid WAV file');
    }

    int channels = 1;
    int sampleRate = 16000;
    int bitsPerSample = 16;
    int audioFormat = 1; // 1 = PCM, 3 = IEEE float
    int dataOffset = 0;
    int dataSize = 0;

    int pos = 12;
    while (pos + 8 <= bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(pos, pos + 4));
      final chunkSize =
          bytes[pos + 4] |
          (bytes[pos + 5] << 8) |
          (bytes[pos + 6] << 16) |
          (bytes[pos + 7] << 24);
      pos += 8;

      if (chunkId == 'fmt ') {
        if (chunkSize >= 16) {
          audioFormat = bytes[pos] | (bytes[pos + 1] << 8);
          channels = bytes[pos + 2] | (bytes[pos + 3] << 8);
          sampleRate =
              bytes[pos + 4] |
              (bytes[pos + 5] << 8) |
              (bytes[pos + 6] << 16) |
              (bytes[pos + 7] << 24);
          bitsPerSample = bytes[pos + 14] | (bytes[pos + 15] << 8);
        }
      } else if (chunkId == 'data') {
        dataOffset = pos;
        dataSize = chunkSize;
        break;
      }

      pos += chunkSize + (chunkSize.isOdd ? 1 : 0); // pad to word boundary
    }

    if (dataOffset == 0) {
      throw const FormatException('No data chunk found in WAV file');
    }
    if (channels == 0 || sampleRate == 0) {
      throw FormatException(
          'Invalid WAV header (channels=$channels, rate=$sampleRate)');
    }
    if (audioFormat != 1 && audioFormat != 3) {
      throw FormatException(
          'Unsupported WAV format code=$audioFormat (only PCM=1 and IEEE float=3)');
    }

    final frameCount = dataSize > 0
        ? (dataSize * 8) ~/ (bitsPerSample * channels)
        : (bytes.length - dataOffset) * 8 ~/ (bitsPerSample * channels);
    if (frameCount <= 0) {
      throw const FormatException('No audio samples found in WAV file');
    }

    final samples = Float32List(frameCount);
    int p = dataOffset;
    final end = dataSize > 0 ? dataOffset + dataSize : bytes.length;
    final bytesPerSample = bitsPerSample ~/ 8;
    final stride = bytesPerSample * channels;
    int out = 0;

    if (audioFormat == 1 && bitsPerSample == 16) {
      while (p + stride <= end && out < frameCount) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
          final i = p + c * 2;
          final s = _toSigned16(bytes[i] | (bytes[i + 1] << 8));
          acc += s / 32768.0;
        }
        samples[out++] = acc / channels;
        p += stride;
      }
    } else if (audioFormat == 1 && bitsPerSample == 8) {
      while (p + stride <= end && out < frameCount) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
          acc += (bytes[p + c] - 128) / 128.0;
        }
        samples[out++] = acc / channels;
        p += stride;
      }
    } else if (audioFormat == 1 && bitsPerSample == 24) {
      while (p + stride <= end && out < frameCount) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
          final b0 = bytes[p + c * 3];
          final b1 = bytes[p + c * 3 + 1];
          final b2 = bytes[p + c * 3 + 2];
          final s24 = _toSigned24(b0 | (b1 << 8) | (b2 << 16));
          acc += s24 / 8388608.0;
        }
        samples[out++] = acc / channels;
        p += stride;
      }
    } else if (audioFormat == 1 && bitsPerSample == 32) {
      while (p + stride <= end && out < frameCount) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
          final i = p + c * 4;
          final s32 = _toSigned32(bytes[i] |
              (bytes[i + 1] << 8) |
              (bytes[i + 2] << 16) |
              (bytes[i + 3] << 24));
          acc += s32 / 2147483648.0;
        }
        samples[out++] = acc / channels;
        p += stride;
      }
    } else if (audioFormat == 3 && bitsPerSample == 32) {
      while (p + stride <= end && out < frameCount) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
          acc += _readFloat32LE(bytes, p + c * 4);
        }
        samples[out++] = acc / channels;
        p += stride;
      }
    } else {
      throw FormatException(
          'Unsupported WAV encoding: format=$audioFormat, bits=$bitsPerSample');
    }

    return AudioBuffer(samples: samples, sampleRate: sampleRate);
  }

  static double _readFloat32LE(Uint8List b, int offset) {
    final bd = ByteData.sublistView(b, offset, offset + 4);
    return bd.getFloat32(0, Endian.little);
  }

  static int _toSigned16(int v) {
    const signBit = 1 << 15;
    return (v & (signBit - 1)) - (v & signBit);
  }

  static int _toSigned24(int v) {
    const signBit = 1 << 23;
    return (v & (signBit - 1)) - (v & signBit);
  }

  static int _toSigned32(int v) {
    const signBit = 1 << 31;
    return (v & (signBit - 1)) - (v & signBit);
  }
}
