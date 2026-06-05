import 'whisper_models.dart';
import 'sherpa_models.dart';
import 'canary_models.dart';
import 'sensevoice_models.dart';
import 'omnilingual_models.dart';
import 'qwen_models.dart';

void registerDefaultModels() {
  registerWhisperModels();
  registerSherpaModels();
  registerCanaryModels();
  registerSenseVoiceModels();
  registerOmnilingualModels();
  registerQwen3AsrModels();
}
