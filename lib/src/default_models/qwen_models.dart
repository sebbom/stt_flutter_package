import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerQwen3AsrModels() {
  const hf =
      'https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main';

  ModelRegistry.register(ModelDescriptor(
    id: 'qwen3-asr-0.6b-int8',
    name: 'Qwen3-ASR 0.6B INT8 (multilingual, autoregressive)',
    type: SttModelType.qwen3asr,
    languages: const [
      'zh', 'en', 'ja', 'ko', 'es', 'fr', 'de', 'pt', 'ru', 'ar',
      'hi', 'bn', 'ur', 'tr', 'vi', 'th', 'id', 'it', 'nl', 'pl',
    ],
    files: [
      ModelFile(
        url: '$hf/conv_frontend.onnx',
        filename: 'conv_frontend.onnx',
        sizeBytes: 44148281,
        sha256:
            'd22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e',
      ),
      ModelFile(
        url: '$hf/encoder.int8.onnx',
        filename: 'encoder.int8.onnx',
        sizeBytes: 182491662,
        sha256:
            '60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9',
      ),
      ModelFile(
        url: '$hf/decoder.int8.onnx',
        filename: 'decoder.int8.onnx',
        sizeBytes: 755914231,
        sha256:
            '4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771',
      ),
      ModelFile(
        url: '$hf/tokenizer/tokenizer_config.json',
        filename: 'tokenizer/tokenizer_config.json',
        sizeBytes: 12487,
        sha256:
            '4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c',
      ),
      ModelFile(
        url: '$hf/tokenizer/merges.txt',
        filename: 'tokenizer/merges.txt',
        sizeBytes: 1671853,
      ),
      ModelFile(
        url: '$hf/tokenizer/vocab.json',
        filename: 'tokenizer/vocab.json',
        sizeBytes: 2776833,
        sha256:
            'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
      ),
    ],
    sizeMb: 935,
  ));
}
