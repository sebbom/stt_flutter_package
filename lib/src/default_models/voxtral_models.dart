import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerVoxtralModels() {
  const base =
      'https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main/onnx';

  ModelRegistry.register(ModelDescriptor(
    id: 'voxtral-mini',
    name: 'Voxtral Mini (4.7B params, 8 languages)',
    type: SttModelType.voxtral,
    languages: ['en', 'de', 'fr', 'es', 'pt', 'hi', 'nl', 'it'],
    files: [
      ModelFile(
        url: '$base/audio_encoder_q4f16.onnx',
        filename: 'audio_encoder.onnx',
      ),
      ModelFile(
        url: '$base/decoder_model_merged_q4f16.onnx',
        filename: 'decoder_model_merged.onnx',
      ),
      ModelFile(
        url: '$base/embed_tokens_q4.onnx',
        filename: 'embed_tokens.onnx',
      ),
      ModelFile(
        url:
            'https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main/tokenizer.json',
        filename: 'tokenizer.json',
      ),
      ModelFile(
        url:
            'https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main/config.json',
        filename: 'config.json',
      ),
      ModelFile(
        url:
            'https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main/preprocessor_config.json',
        filename: 'preprocessor_config.json',
      ),
    ],
    sizeMb: 2700,
  ));
}
