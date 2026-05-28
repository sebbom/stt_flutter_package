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
    final channels = bytes[22] | (bytes[23] << 8);
    final sampleRate = bytes[24] | (bytes[25] << 8) | (bytes[26] << 16) | (bytes[27] << 24);
    final bitsPerSample = bytes[34] | (bytes[35] << 8);
    final headerSize = 44;

    final samples = <double>[];
    if (bitsPerSample == 16) {
      for (int i = headerSize; i < bytes.length - 1; i += 2 * channels) {
        final s = (bytes[i] | (bytes[i + 1] << 8)).toSigned(16);
        samples.add(s / 32768.0);
      }
    } else if (bitsPerSample == 8) {
      for (int i = headerSize; i < bytes.length; i += channels) {
        samples.add((bytes[i] - 128) / 128.0);
      }
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
