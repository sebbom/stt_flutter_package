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
      ),
      ModelFile(
        url: '$hf/decoder.int8.onnx',
        filename: 'decoder.int8.onnx',
      ),
      ModelFile(
        url: '$hf/tokens.txt',
        filename: 'tokens.txt',
      ),
    ],
    sizeMb: 200,
  ));
}
