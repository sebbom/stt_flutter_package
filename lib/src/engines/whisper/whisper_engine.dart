import 'dart:isolate';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import '../../stt_result.dart';
import '../../audio/audio_buffer.dart';
import '../inference_engine.dart';
import 'mel_spectrogram.dart';

class WhisperInferenceEngine implements InferenceEngine {
  final ort.OnnxRuntime _runtime;
  ort.OrtSession? _encoderSession;
  ort.OrtSession? _decoderSession;

  WhisperInferenceEngine(this._runtime);

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    _encoderSession = await _runtime.createSession(modelFiles['encoder.onnx']!);
    _decoderSession = await _runtime.createSession(modelFiles['decoder.onnx']!);
  }

  @override
  Future<SttResult> transcribe(AudioBuffer audio, {String? language}) async {
    final stopwatch = Stopwatch()..start();

    // Offload mel spectrogram to background isolate
    final mel = await Isolate.run(() => MelSpectrogram.compute(audio.samples));

    // TODO: run encoder ONNX, decoder autoregressive loop
    // final encoderOut = await _encoderSession!.run({'mel': inputTensor});
    // final decoderOut = await _decoderSession!.run(decoderInput);

    stopwatch.stop();
    return SttResult(text: '(whisper stub)', inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000);
  }

  @override
  Future<void> dispose() async {
    await _encoderSession?.close();
    await _decoderSession?.close();
  }
}
