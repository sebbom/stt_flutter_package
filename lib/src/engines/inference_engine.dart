import '../stt_result.dart';
import '../audio/audio_buffer.dart';
import '../cancellation_token.dart';

abstract class InferenceEngine {
  Future<void> load(Map<String, String> modelFiles);
  Future<SttResult> transcribe(
    AudioBuffer audio, {
    String? language,
    CancellationToken? token,
    Map<String, dynamic>? options,
  });
  Future<void> dispose();

  bool get supportsExplicitLanguage;
  Set<String> get supportedLanguages;
}
