import '../stt_config.dart';
import 'inference_engine.dart';
import 'whisper/whisper_engine.dart';
import 'sherpa/sherpa_engine.dart';
import 'canary/canary_engine.dart';
import 'nemo/nemo_engine.dart';

InferenceEngine createEngine(SttModelType type) {
  switch (type) {
    case SttModelType.whisper:
      return WhisperInferenceEngine();
    case SttModelType.sherpa:
      return SherpaInferenceEngine();
    case SttModelType.nemo:
      return NemoInferenceEngine();
    case SttModelType.canary:
      return CanaryInferenceEngine();
    case SttModelType.voxtral:
      throw UnsupportedError(
          'Voxtral models are not supported with sherpa_onnx backend. '
          'Use whisper or sherpa model types.');
  }
}
