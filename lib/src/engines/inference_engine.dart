import '../stt_result.dart';
import '../audio/audio_buffer.dart';

abstract class InferenceEngine {
  Future<void> load(Map<String, String> modelFiles);
  Future<SttResult> transcribe(AudioBuffer audio, {String? language});
  Future<void> dispose();
}
