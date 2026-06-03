import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerSherpaModels() {
  const base = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models';
  const hf = 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main';

  ModelRegistry.register(ModelDescriptor(
    id: 'sherpa-zipformer-en',
    name: 'Sherpa Zipformer EN INT8 (streaming, English)',
    type: SttModelType.sherpa,
    languages: ['en'],
    files: [
      ModelFile(
        url: '$base/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2',
        filename: 'sherpa-onnx-zipformer-en-2023-06-26.tar.bz2',
      ),
    ],
    sizeMb: 300,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'parakeet-tdt-0.6b-multilingual',
    name: 'Parakeet TDT 0.6B (multilingual, streaming)',
    type: SttModelType.nemo,
    languages: [
      'en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'pl', 'ru',
      'bg', 'hr', 'cs', 'da', 'et', 'fi', 'el', 'hu', 'lv',
      'lt', 'mt', 'ro', 'sk', 'sl', 'sv', 'uk',
    ],
    files: [
      ModelFile(
        url: '$hf/encoder.int8.onnx',
        filename: 'encoder.int8.onnx',
      ),
      ModelFile(
        url: '$hf/decoder.int8.onnx',
        filename: 'decoder.int8.onnx',
      ),
      ModelFile(
        url: '$hf/joiner.int8.onnx',
        filename: 'joiner.int8.onnx',
      ),
      ModelFile(
        url: '$hf/tokens.txt',
        filename: 'tokens.txt',
      ),
    ],
    sizeMb: 400,
  ));
}
