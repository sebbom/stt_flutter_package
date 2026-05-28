import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import '../stt_config.dart';
import 'inference_engine.dart';
import 'whisper/whisper_engine.dart';
import 'sherpa/sherpa_engine.dart';
import 'voxtral/voxtral_engine.dart';

InferenceEngine createEngine(SttModelType type, ort.OnnxRuntime runtime) {
  switch (type) {
    case SttModelType.whisper:
      return WhisperInferenceEngine(runtime);
    case SttModelType.sherpa:
      return SherpaInferenceEngine(runtime);
    case SttModelType.voxtral:
      return VoxtralInferenceEngine(runtime);
  }
}
