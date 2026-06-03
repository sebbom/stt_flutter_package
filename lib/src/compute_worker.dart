import 'dart:isolate';
import 'audio/audio_buffer.dart';
import 'audio/audio_processor.dart';
import 'stt_logger.dart';

/// Off-main-isolate compute. We use one-shot [Isolate.run] for resampling
/// instead of a long-lived background isolate — each call spawns (or reuses)
/// a short-lived worker and waits for its result.
class ComputeWorker {
  ComputeWorker._();

  static final ComputeWorker instance = ComputeWorker._();

  Future<void> initialize() async {
    SttLogger.d('ComputeWorker ready (Isolate.run mode)');
  }

  Future<AudioBuffer> resample(
    AudioBuffer audio, {
    int targetRate = AudioProcessor.targetSampleRate,
  }) {
    return Isolate.run(() => AudioProcessor.resampleSync(audio, targetRate: targetRate));
  }

  Future<void> dispose() async {
    SttLogger.d('ComputeWorker disposed (nothing to release in Isolate.run mode)');
  }
}
