import '../stt_config.dart';
import '../model_registry.dart';
import 'inference_engine.dart';
import 'whisper/whisper_engine.dart';
import 'sherpa/sherpa_engine.dart';
import 'canary/canary_engine.dart';
import 'nemo/nemo_engine.dart';

InferenceEngine createEngine(ModelDescriptor model) {
  switch (model.type) {
    case SttModelType.whisper:
      return WhisperInferenceEngine(model);
    case SttModelType.sherpa:
      return SherpaInferenceEngine(model);
    case SttModelType.nemo:
      return NemoInferenceEngine(model);
    case SttModelType.canary:
      return CanaryInferenceEngine(model);
  }
}
