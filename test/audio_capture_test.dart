import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/stt_flutter.dart';

void main() {
  group('AudioCaptureService.bytesToFloat32', () {
    test('empty input returns empty list', () {
      final out = AudioCaptureService.bytesToFloat32(Uint8List(0));
      expect(out, isEmpty);
    });

    test('odd-length input drops the trailing byte', () {
      final out = AudioCaptureService.bytesToFloat32(Uint8List.fromList([0, 0, 0]));
      expect(out.length, 1);
    });

    test('PCM16 little-endian conversion to [-1, 1]', () {
      final bytes = Uint8List.fromList([
        0x00, 0x00, // 0
        0x00, 0x80, // -32768 → -1.0
        0xff, 0x7f, // 32767 → ~1.0
        0x00, 0x40, // 16384 → 0.5
      ]);
      final out = AudioCaptureService.bytesToFloat32(bytes);
      expect(out.length, 4);
      expect(out[0], closeTo(0.0, 1e-6));
      expect(out[1], closeTo(-1.0, 1e-3));
      expect(out[2], closeTo(1.0 - 1 / 32768, 1e-3));
      expect(out[3], closeTo(0.5, 1e-3));
    });
  });
}
