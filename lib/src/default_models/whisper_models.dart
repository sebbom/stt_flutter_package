import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerWhisperModels() {
  const base = 'https://huggingface.co/onnx-community';

  // Large-v3 uses Xet external data files for FP32, but q4f16 is self-contained
  void addVariants(String id, String name, String repo, List<String> langs, int mb) {
    ModelRegistry.register(ModelDescriptor(
      id: id,
      name: name,
      type: SttModelType.whisper,
      languages: langs,
      files: [
        ModelFile(
          url: '$base/$repo/resolve/main/onnx/encoder_model_q4f16.onnx',
          filename: 'encoder.onnx',
        ),
        ModelFile(
          url: '$base/$repo/resolve/main/onnx/decoder_model_merged_q4f16.onnx',
          filename: 'decoder.onnx',
        ),
        ModelFile(
          url: '$base/$repo/resolve/main/vocab.json',
          filename: 'vocab.json',
        ),
      ],
      sizeMb: mb + 1,
    ));
  }

  addVariants(
    'whisper-tiny',
    'Whisper Tiny q4f16 (39M params)',
    'whisper-tiny-ONNX',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    55,
  );
  addVariants(
    'whisper-tiny-en',
    'Whisper Tiny English-only q4f16 (39M params)',
    'whisper-tiny.en-ONNX',
    ['en'],
    55,
  );

  addVariants(
    'whisper-base',
    'Whisper Base q4f16 (74M params)',
    'whisper-base-ONNX',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    85,
  );
  addVariants(
    'whisper-base-en',
    'Whisper Base English-only q4f16 (74M params)',
    'whisper-base.en-ONNX',
    ['en'],
    85,
  );

  addVariants(
    'whisper-small',
    'Whisper Small q4f16 (244M params)',
    'whisper-small-ONNX',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    200,
  );
  addVariants(
    'whisper-small-en',
    'Whisper Small English-only q4f16 (244M params)',
    'whisper-small.en-ONNX',
    ['en'],
    200,
  );

  addVariants(
    'whisper-medium',
    'Whisper Medium q4f16 (769M params)',
    'whisper-medium-ONNX',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    520,
  );
  addVariants(
    'whisper-medium-en',
    'Whisper Medium English-only q4f16 (769M params)',
    'whisper-medium.en-ONNX',
    ['en'],
    520,
  );

  addVariants(
    'whisper-large-v3',
    'Whisper Large v3 q4f16 (1.55B params)',
    'whisper-large-v3-ONNX',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    1000,
  );

  addVariants(
    'whisper-large-v3-turbo',
    'Whisper Large v3 Turbo q4f16 (809M params)',
    'whisper-large-v3-turbo-ONNX',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    570,
  );
}
