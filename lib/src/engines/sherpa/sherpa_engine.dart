import 'dart:isolate';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import '../../stt_result.dart';
import '../../audio/audio_buffer.dart';
import '../inference_engine.dart';

class SherpaInferenceEngine implements InferenceEngine {
  final ort.OnnxRuntime _runtime;

  SherpaInferenceEngine(this._runtime);

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    // TODO: implement load
  }

  @override
  Future<SttResult> transcribe(AudioBuffer audio, {String? language}) async {
    final stopwatch = Stopwatch()..start();

    // TODO: offload Fbank extraction to isolate, run transducer decoder

    stopwatch.stop();
    return SttResult(text: '(sherpa stub)', inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000);
  }

  @override
  Future<void> dispose() async {
    // TODO: implement dispose
  }
}
