import 'dart:io';
import 'dart:typed_data';
import 'audio_buffer.dart';

class AudioProcessor {
  static const int targetSampleRate = 16000;

  /// Load a WAV file and return a 16kHz mono Float32 AudioBuffer.
  static Future<AudioBuffer> loadWav(String path) async {
    final file = File(path);
    final bytes = await file.readAsBytes();
    return _parseWavBytes(bytes);
  }

  /// Resample to 16kHz mono. Lightweight enough to run on the main isolate,
  /// but for big buffers callers should prefer [Isolate.run].
  static AudioBuffer resampleSync(
    AudioBuffer input, {
    int targetRate = targetSampleRate,
  }) {
    if (input.sampleRate == targetRate) return input;
    final ratio = input.sampleRate / targetRate;
    final newLength = (input.length / ratio).round();
    final output = Float32List(newLength);

    for (int i = 0; i < newLength; i++) {
      final srcPos = i * ratio;
      final srcIdx = srcPos.floor();
      final frac = srcPos - srcIdx;
      if (srcIdx + 1 < input.length) {
        output[i] =
            input.samples[srcIdx] * (1 - frac) + input.samples[srcIdx + 1] * frac;
      } else {
        output[i] = input.samples[srcIdx];
      }
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
