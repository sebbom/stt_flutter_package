import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/src/audio/audio_processor.dart';

void main() {
  group('PreprocessConfig.isNoOp', () {
    test('default config is a no-op', () {
      expect(PreprocessConfig.none.isNoOp, true);
    });

    test('config with only noiseSuppression=true is not a no-op', () {
      const c = PreprocessConfig(noiseSuppression: true);
      expect(c.isNoOp, false);
    });

    test('config with denoiser type but empty dir is a no-op', () {
      const c = PreprocessConfig(denoiserType: DenoiserType.gtcrn);
      expect(c.isNoOp, true);
      expect(c.hasDenoiser, false);
    });

    test('config with denoiser type and non-empty dir has denoiser', () {
      const c = PreprocessConfig(
        denoiserType: DenoiserType.gtcrn,
        denoiserModelDir: '/tmp/dn',
      );
      expect(c.isNoOp, false);
      expect(c.hasDenoiser, true);
    });

    test('config with gain=1.0 and no other knobs is a no-op', () {
      const c = PreprocessConfig(gain: 1.0);
      expect(c.isNoOp, true);
    });
  });
}
