import 'dart:typed_data';

class AudioBuffer {
  final Float32List samples;
  final int sampleRate;

  const AudioBuffer({required this.samples, required this.sampleRate});

  int get length => samples.length;
  double get durationSec => length / sampleRate;
}
