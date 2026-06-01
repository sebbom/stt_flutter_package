import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerVoxtralModels() {
  const base =
      'https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main/onnx';
  const root =
      'https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main';

  ModelRegistry.register(ModelDescriptor(
    id: 'voxtral-mini',
    name: 'Voxtral Mini q4f16 (3B params, 8 languages)',
    type: SttModelType.voxtral,
    languages: ['en', 'de', 'fr', 'es', 'pt', 'hi', 'nl', 'it'],
    files: [
      // Audio encoder (q4f16 quantized)
      ModelFile(
        url: '$base/audio_encoder_q4f16.onnx',
        filename: 'audio_encoder_q4f16.onnx',
      ),
      ModelFile(
        url: '$base/audio_encoder_q4f16.onnx_data',
        filename: 'audio_encoder_q4f16.onnx_data',
      ),
      // Decoder merged (q4f16 quantized)
      ModelFile(
        url: '$base/decoder_model_merged_q4f16.onnx',
        filename: 'decoder_model_merged_q4f16.onnx',
      ),
      ModelFile(
        url: '$base/decoder_model_merged_q4f16.onnx_data',
        filename: 'decoder_model_merged_q4f16.onnx_data',
      ),
      // Embed tokens (q4 quantized)
      ModelFile(
        url: '$base/embed_tokens_q4.onnx',
        filename: 'embed_tokens_q4.onnx',
      ),
      ModelFile(
        url: '$base/embed_tokens_q4.onnx_data',
        filename: 'embed_tokens_q4.onnx_data',
      ),
      // Config files
      ModelFile(
        url: '$root/tokenizer.json',
        filename: 'tokenizer.json',
      ),
      ModelFile(
        url: '$root/config.json',
        filename: 'config.json',
      ),
      ModelFile(
        url: '$root/preprocessor_config.json',
        filename: 'preprocessor_config.json',
      ),
    ],
    sizeMb: 2700,
  ));
}
