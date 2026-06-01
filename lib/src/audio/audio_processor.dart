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

  /// Resample to 16kHz mono. Runs on main isolate (lightweight I/O).
  static Future<AudioBuffer> resample(AudioBuffer input, {int targetRate = targetSampleRate}) async {
    return Future.value(resampleSync(input, targetRate: targetRate));
  }

  /// Sync resample — safe for [Isolate.run].
  static AudioBuffer resampleSync(AudioBuffer input, {int targetRate = targetSampleRate}) {
    if (input.sampleRate == targetRate) return input;
    final ratio = input.sampleRate / targetRate;
    final newLength = (input.length / ratio).round();
    final output = Float32List(newLength);

    for (int i = 0; i < newLength; i++) {
      final srcPos = i * ratio;
      final srcIdx = srcPos.floor();
      final frac = srcPos - srcIdx;
      if (srcIdx + 1 < input.length) {
        output[i] = input.samples[srcIdx] * (1 - frac) + input.samples[srcIdx + 1] * frac;
      } else {
        output[i] = input.samples[srcIdx];
      }
    }
    return AudioBuffer(samples: output, sampleRate: targetRate);
  }

  static AudioBuffer _parseWavBytes(Uint8List bytes) {
    if (bytes.length < 12 || String.fromCharCodes(bytes.sublist(0, 4)) != 'RIFF') {
      throw FormatException('Not a valid WAV file');
    }

    // Read format info from the "fmt " chunk (always present)
    int channels = 1, sampleRate = 16000, bitsPerSample = 16;
    int dataOffset = 0;

    int pos = 12; // skip RIFF header
    while (pos + 8 <= bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(pos, pos + 4));
      final chunkSize = bytes[pos + 4] | (bytes[pos + 5] << 8) | (bytes[pos + 6] << 16) | (bytes[pos + 7] << 24);
      pos += 8;

      if (pos + chunkSize > bytes.length) break;

      if (chunkId == 'fmt ') {
        if (chunkSize >= 16) {
          channels = bytes[pos + 2] | (bytes[pos + 3] << 8);
          sampleRate = bytes[pos + 4] | (bytes[pos + 5] << 8) | (bytes[pos + 6] << 16) | (bytes[pos + 7] << 24);
          bitsPerSample = bytes[pos + 14] | (bytes[pos + 15] << 8);
        }
      } else if (chunkId == 'data') {
        dataOffset = pos;
        break;
      }

      pos += chunkSize + (chunkSize % 2); // pad to word boundary
    }

    if (dataOffset == 0) {
      throw FormatException('No data chunk found in WAV file');
    }
    if (channels == 0 || sampleRate == 0) {
      throw FormatException('Invalid WAV header (channels=$channels, rate=$sampleRate)');
    }

    final samples = <double>[];
    if (bitsPerSample == 16) {
      for (int i = dataOffset; i < bytes.length - 1; i += 2 * channels) {
        final s = (bytes[i] | (bytes[i + 1] << 8)).toSigned(16);
        samples.add(s / 32768.0);
      }
    } else if (bitsPerSample == 8) {
      for (int i = dataOffset; i < bytes.length; i += channels) {
        samples.add((bytes[i] - 128) / 128.0);
      }
    }
    if (samples.isEmpty) {
      throw FormatException('No audio samples found in WAV file');
    }
    return AudioBuffer(samples: Float32List.fromList(samples), sampleRate: sampleRate);
  }
}

extension on int {
  int toSigned(int bits) {
    final signBit = 1 << (bits - 1);
    return (this & (signBit - 1)) - (this & signBit);
  }
}
