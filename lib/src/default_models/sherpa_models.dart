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
        sizeBytes: 652184281,
        sha256: 'acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247',
      ),
      ModelFile(
        url: '$hf/decoder.int8.onnx',
        filename: 'decoder.int8.onnx',
        sizeBytes: 11845275,
        sha256: '179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e',
      ),
      ModelFile(
        url: '$hf/joiner.int8.onnx',
        filename: 'joiner.int8.onnx',
        sizeBytes: 6355277,
        sha256: '3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3',
      ),
      ModelFile(
        url: '$hf/tokens.txt',
        filename: 'tokens.txt',
        sizeBytes: 93939,
      ),
    ],
    sizeMb: 400,
  ));
}
