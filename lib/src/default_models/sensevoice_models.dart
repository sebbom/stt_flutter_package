import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerSenseVoiceModels() {
  const hf =
      'https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main';

  ModelRegistry.register(ModelDescriptor(
    id: 'sensevoice-small-zh-en-ja-ko-yue',
    name: 'SenseVoice Small INT8 (zh, en, ja, ko, yue — emotion + events)',
    type: SttModelType.sensevoice,
    languages: ['zh', 'en', 'ja', 'ko', 'yue'],
    files: [
      ModelFile(
        url: '$hf/model.int8.onnx',
        filename: 'model.int8.onnx',
        sizeBytes: 237115547,
        sha256:
            '12ca1a2ae7ecf3e0019ef2822307ee0b5cadc9196569e379b4c4026f8205276d',
      ),
      ModelFile(
        url: '$hf/tokens.txt',
        filename: 'tokens.txt',
        sizeBytes: 315894,
      ),
    ],
    sizeMb: 230,
  ));
}
