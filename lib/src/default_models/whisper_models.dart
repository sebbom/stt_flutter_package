import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerWhisperModels() {
  const base = 'https://huggingface.co/onnx-community';

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-tiny',
    name: 'Whisper Tiny (39M params, multilingual)',
    type: SttModelType.whisper,
    languages: ['en', 'de', 'fr', 'es'],
    files: [
      ModelFile(
        url: '$base/whisper-tiny-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-tiny-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 220,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-tiny-en',
    name: 'Whisper Tiny (39M, English-only)',
    type: SttModelType.whisper,
    languages: ['en'],
    files: [
      ModelFile(
        url: '$base/whisper-tiny.en-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-tiny.en-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 220,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-base',
    name: 'Whisper Base (74M params, multilingual)',
    type: SttModelType.whisper,
    languages: ['en', 'de', 'fr', 'es'],
    files: [
      ModelFile(
        url: '$base/whisper-base-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-base-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 370,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-base-en',
    name: 'Whisper Base (74M, English-only)',
    type: SttModelType.whisper,
    languages: ['en'],
    files: [
      ModelFile(
        url: '$base/whisper-base.en-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-base.en-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 370,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-small',
    name: 'Whisper Small (244M params, multilingual)',
    type: SttModelType.whisper,
    languages: ['en', 'de', 'fr', 'es'],
    files: [
      ModelFile(
        url: '$base/whisper-small-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-small-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 1100,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-small-en',
    name: 'Whisper Small (244M, English-only)',
    type: SttModelType.whisper,
    languages: ['en'],
    files: [
      ModelFile(
        url: '$base/whisper-small.en-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-small.en-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 1100,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-medium',
    name: 'Whisper Medium (769M params, multilingual)',
    type: SttModelType.whisper,
    languages: ['en', 'de', 'fr', 'es'],
    files: [
      ModelFile(
        url: '$base/whisper-medium-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-medium-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 2500,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-medium-en',
    name: 'Whisper Medium (769M, English-only)',
    type: SttModelType.whisper,
    languages: ['en'],
    files: [
      ModelFile(
        url: '$base/whisper-medium.en-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-medium.en-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 2500,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-large-v3',
    name: 'Whisper Large v3 (1.55B params, multilingual)',
    type: SttModelType.whisper,
    languages: ['en', 'de', 'fr', 'es'],
    files: [
      ModelFile(
        url: '$base/whisper-large-v3-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-large-v3-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 4500,
  ));

  ModelRegistry.register(ModelDescriptor(
    id: 'whisper-large-v3-turbo',
    name: 'Whisper Large v3 Turbo (809M params, multilingual)',
    type: SttModelType.whisper,
    languages: ['en', 'de', 'fr', 'es'],
    files: [
      ModelFile(
        url: '$base/whisper-large-v3-turbo-ONNX/resolve/main/onnx/encoder_model.onnx',
        filename: 'encoder.onnx',
      ),
      ModelFile(
        url: '$base/whisper-large-v3-turbo-ONNX/resolve/main/onnx/decoder_model_merged.onnx',
        filename: 'decoder.onnx',
      ),
    ],
    sizeMb: 2500,
  ));
}
