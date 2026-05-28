import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerSherpaModels() {
  const base = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models';

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
    sizeMb: 35,
  ));
}
