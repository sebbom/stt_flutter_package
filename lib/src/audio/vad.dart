import 'dart:math';
import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';

abstract class VadEngine {
  bool processChunk(List<int> audioChunk);
  bool get isSpeaking;
  void reset();
  void setSpeechThreshold(double threshold);
  void setSilenceThreshold(double threshold);
}

class EnergyVadEngine implements VadEngine {
  double _speechThreshold;
  double _silenceThreshold;
  final int minSpeechFrames;
  final int minSilenceFrames;

  int _speechFrameCount = 0;
  int _silenceFrameCount = 0;
  bool _isSpeaking = false;

  EnergyVadEngine({
    double speechThreshold = 0.01,
    double silenceThreshold = 0.001,
    this.minSpeechFrames = 3,
    this.minSilenceFrames = 10,
  })  : _speechThreshold = speechThreshold,
        _silenceThreshold = silenceThreshold;

  @override
  bool processChunk(List<int> audioChunk) {
    final energy = _calculateEnergy(audioChunk);

    if (energy > _speechThreshold) {
      _speechFrameCount++;
      _silenceFrameCount = 0;
      if (_speechFrameCount >= minSpeechFrames && !_isSpeaking) {
        _isSpeaking = true;
        _speechFrameCount = 0;
        return true;
      }
    } else {
      _silenceFrameCount++;
      _speechFrameCount = 0;
      if (_silenceFrameCount >= minSilenceFrames && _isSpeaking) {
        _isSpeaking = false;
        _silenceFrameCount = 0;
        return false;
      }
    }

    return _isSpeaking;
  }

  double _calculateEnergy(List<int> audioChunk) {
    double sum = 0;
    for (final sample in audioChunk) {
      final normalized = sample / 32768.0;
      sum += normalized * normalized;
    }
    return sum / max(audioChunk.length, 1);
  }

  @override
  bool get isSpeaking => _isSpeaking;

  @override
  void reset() {
    _speechFrameCount = 0;
    _silenceFrameCount = 0;
    _isSpeaking = false;
  }

  @override
  void setSpeechThreshold(double threshold) {
    _speechThreshold = threshold;
  }

  @override
  void setSilenceThreshold(double threshold) {
    _silenceThreshold = threshold;
  }

  double get speechThreshold => _speechThreshold;
  double get silenceThreshold => _silenceThreshold;
}

class SherpaOnnxVadEngine implements VadEngine {
  final VoiceActivityDetector _detector;
  bool _isSpeaking = false;

  SherpaOnnxVadEngine({
    required String modelPath,
    double threshold = 0.5,
    double minSilenceDuration = 0.5,
    double minSpeechDuration = 0.25,
  }) : _detector = VoiceActivityDetector(
          config: VadModelConfig(
            sileroVad: SileroVadModelConfig(
              model: modelPath,
              threshold: threshold,
              minSilenceDuration: minSilenceDuration,
              minSpeechDuration: minSpeechDuration,
            ),
            sampleRate: 16000,
            numThreads: 1,
            provider: 'cpu',
            debug: false,
          ),
          bufferSizeInSeconds: 10,
        );

  @override
  bool processChunk(List<int> audioChunk) {
    final samples = Float32List(audioChunk.length);
    for (int i = 0; i < audioChunk.length; i++) {
      int s = audioChunk[i];
      if (s > 32767) s -= 65536;
      samples[i] = s / 32768.0;
    }
    _detector.acceptWaveform(samples);
    _isSpeaking = _detector.isDetected();
    return _isSpeaking;
  }

  @override
  bool get isSpeaking => _isSpeaking;

  @override
  void reset() {
    _detector.reset();
    _isSpeaking = false;
  }

  @override
  void setSpeechThreshold(double threshold) {
    // Not supported by sherpa_onnx VAD at runtime
  }

  @override
  void setSilenceThreshold(double threshold) {
    // Not supported by sherpa_onnx VAD at runtime
  }

  void free() {
    _detector.free();
  }
}
