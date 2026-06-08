import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerOmnilingualModels() {
  const hf300M =
      'https://huggingface.co/csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05/resolve/main';
  const hf1B =
      'https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-2025-11-12/resolve/main';

  ModelRegistry.register(ModelDescriptor(
    id: 'omnilingual-300m-ctc-1600lang',
    name: 'Omnilingual 300M CTC (1600 languages)',
    type: SttModelType.omnilingual,
    languages: const [
      'en', 'zh', 'ja', 'ko', 'es', 'fr', 'de', 'pt', 'ru', 'ar',
      'hi', 'bn', 'ur', 'tr', 'vi', 'th', 'id', 'sw', 'yo', 'am',
    ],
    files: [
      ModelFile(
        url: '$hf300M/model.onnx',
        filename: 'model.onnx',
        sizeBytes: 1304579963,
        sha256:
            '32432a0afa53180d08553de33baec1a36a43141bded193a26bcf2bed0bcd9c13',
      ),
      ModelFile(
        url: '$hf300M/tokens.txt',
        filename: 'tokens.txt',
        sizeBytes: 90630,
      ),
    ],
    sizeMb: 1245,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'omnilingual-1b-ctc-1600lang',
    name: 'Omnilingual 1B CTC (1600 languages, external weights)',
    type: SttModelType.omnilingual,
    languages: const [
      'en', 'zh', 'ja', 'ko', 'es', 'fr', 'de', 'pt', 'ru', 'ar',
      'hi', 'bn', 'ur', 'tr', 'vi', 'th', 'id', 'sw', 'yo', 'am',
    ],
    files: [
      ModelFile(
        url: '$hf1B/model.onnx',
        filename: 'model.onnx',
        sizeBytes: 1417765,
        sha256:
            '49465dc6dbf82b86a8025f8b45d7be34d43d63944cb0ed022e45ff212d30d306',
      ),
      ModelFile(
        url: '$hf1B/model.weights',
        filename: 'model.weights',
        sizeBytes: 3900260688,
        sha256:
            'd0954ba0b14d84b68b375989b56c00f1c86390d585a54bfddac77465adca1afd',
      ),
      ModelFile(
        url: '$hf1B/tokens.txt',
        filename: 'tokens.txt',
        sizeBytes: 86423,
      ),
    ],
    sizeMb: 3725,
  ));
}
