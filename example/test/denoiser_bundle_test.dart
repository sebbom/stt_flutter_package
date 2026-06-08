import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter_example/utils/denoiser_bundle.dart';

void main() {
  group('DenoiserBundle.dirFor', () {
    test('returns parent directory of an .onnx file', () {
      expect(
        DenoiserBundle.dirFor('/tmp/abc/model.onnx'),
        '/tmp/abc',
      );
    });

    test('returns the input when it has no slashes', () {
      expect(DenoiserBundle.dirFor('model.onnx'), 'model.onnx');
    });

    test('returns null for null input', () {
      expect(DenoiserBundle.dirFor(null), isNull);
    });

    test('handles deeply nested paths', () {
      expect(
        DenoiserBundle.dirFor('/a/b/c/d/model.onnx'),
        '/a/b/c/d',
      );
    });
  });
}
