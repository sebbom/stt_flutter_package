import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerCanaryModels() {
  const hf = 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/resolve/main';

  ModelRegistry.register(ModelDescriptor(
    id: 'canary-180m-en-es-de-fr',
    name: 'Canary 180M Flash (en, es, de, fr, offline)',
    type: SttModelType.canary,
    languages: ['en', 'es', 'de', 'fr'],
    files: [
      ModelFile(
        url: '$hf/encoder.int8.onnx',
        filename: 'encoder.int8.onnx',
        sizeBytes: 132678643,
        sha256: '7a75b4e2a5857a6dcc0819503bbe3fad66943db4a3ccf21d3f27c633667d303f',
      ),
      ModelFile(
        url: '$hf/decoder.int8.onnx',
        filename: 'decoder.int8.onnx',
        sizeBytes: 74437848,
        sha256: 'e41a2ab9c0c2fe81a1e8ade5a45fb02a74bc4db7d1f91b89a54a25e2cf79cba2',
      ),
      ModelFile(
        url: '$hf/tokens.txt',
        filename: 'tokens.txt',
        sizeBytes: 53555,
      ),
    ],
    sizeMb: 200,
  ));
}
