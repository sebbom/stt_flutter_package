// Generates test WAV files for unit testing.
// Run: dart tool/generate_fixtures.dart

import 'dart:io';
import 'dart:typed_data';
import 'dart:math';

void main() {
  final dir = Directory('test/fixtures');
  dir.createSync(recursive: true);

  // 1-second sine sweeps at 16kHz mono, 16-bit PCM
  _generateWav('${dir.path}/hello_en.wav', _tone(16000, 440));
  _generateWav('${dir.path}/guten_tag_de.wav', _tone(16000, 880));
  _generateWav('${dir.path}/bonjour_fr.wav', _tone(16000, 330));
  _generateWav('${dir.path}/hola_es.wav', _tone(16000, 660));

  print('Generated 4 WAV fixtures in ${dir.path}');
}

Float64List _tone(int sampleRate, double freqHz) {
  const durationSec = 1.0;
  final n = (sampleRate * durationSec).toInt();
  final samples = Float64List(n);
  for (int i = 0; i < n; i++) {
    final t = i / sampleRate;
    samples[i] = 0.3 * sin(2 * pi * freqHz * t) * (1 - t / durationSec); // fade out
  }
  return samples;
}

void _generateWav(String path, Float64List samples) {
  final sampleRate = 16000;
  final bitsPerSample = 16;
  final channels = 1;
  final dataSize = samples.length * (bitsPerSample ~/ 8);
  final headerSize = 44;

  final file = File(path);
  final bytes = BytesBuilder();

  // RIFF header
  bytes.add(_str('RIFF'));
  bytes.add(_le32(36 + dataSize));
  bytes.add(_str('WAVE'));

  // fmt chunk
  bytes.add(_str('fmt '));
  bytes.add(_le32(16));
  bytes.add(_le16(1));          // PCM
  bytes.add(_le16(channels));
  bytes.add(_le32(sampleRate));
  bytes.add(_le32(sampleRate * channels * bitsPerSample ~/ 8));
  bytes.add(_le16(channels * bitsPerSample ~/ 8));
  bytes.add(_le16(bitsPerSample));

  // data chunk
  bytes.add(_str('data'));
  bytes.add(_le32(dataSize));
  for (final s in samples) {
    final clamped = (s * 32767).clamp(-32767, 32767).toInt();
    bytes.add(_le16(clamped));
  }

  file.writeAsBytesSync(bytes.toBytes());
  print('  Wrote ${samples.length} samples → $path');
}

List<int> _str(String s) => s.codeUnits;
List<int> _le16(int v) => [v & 0xff, (v >> 8) & 0xff];
List<int> _le32(int v) => [v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff];
