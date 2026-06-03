import 'package:stt_flutter/src/model_registry.dart';
import 'package:stt_flutter/src/stt_config.dart';

void registerWhisperModels() {
  const base = 'https://huggingface.co/csukuangfj';

  void addVariants(
    String id,
    String name,
    String repo,
    String prefix,
    List<String> langs,
    int mb, {
    List<ModelFile> extraFiles = const [],
  }) {
    ModelRegistry.register(ModelDescriptor(
      id: id,
      name: name,
      type: SttModelType.whisper,
      languages: langs,
      files: [
        ModelFile(
          url: '$base/$repo/resolve/main/$prefix-encoder.onnx',
          filename: '$prefix-encoder.onnx',
        ),
        ModelFile(
          url: '$base/$repo/resolve/main/$prefix-decoder.onnx',
          filename: '$prefix-decoder.onnx',
        ),
        ModelFile(
          url: '$base/$repo/resolve/main/$prefix-tokens.txt',
          filename: '$prefix-tokens.txt',
        ),
        ...extraFiles,
      ],
      sizeMb: mb,
    ));
  }

  addVariants(
    'whisper-tiny',
    'Whisper Tiny (39M params, multilingual)',
    'sherpa-onnx-whisper-tiny',
    'tiny',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    150,
  );
  addVariants(
    'whisper-tiny-en',
    'Whisper Tiny (39M params, English only)',
    'sherpa-onnx-whisper-tiny.en',
    'tiny.en',
    ['en'],
    150,
  );

  addVariants(
    'whisper-base',
    'Whisper Base (74M params, multilingual)',
    'sherpa-onnx-whisper-base',
    'base',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    240,
  );
  addVariants(
    'whisper-base-en',
    'Whisper Base (74M params, English only)',
    'sherpa-onnx-whisper-base.en',
    'base.en',
    ['en'],
    240,
  );

  addVariants(
    'whisper-small',
    'Whisper Small (244M params, multilingual)',
    'sherpa-onnx-whisper-small',
    'small',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    460,
  );
  addVariants(
    'whisper-small-en',
    'Whisper Small (244M params, English only)',
    'sherpa-onnx-whisper-small.en',
    'small.en',
    ['en'],
    460,
  );

  addVariants(
    'whisper-medium',
    'Whisper Medium (769M params, multilingual)',
    'sherpa-onnx-whisper-medium',
    'medium',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    960,
  );
  addVariants(
    'whisper-medium-en',
    'Whisper Medium (769M params, English only)',
    'sherpa-onnx-whisper-medium.en',
    'medium.en',
    ['en'],
    960,
  );

  addVariants(
    'whisper-large-v3',
    'Whisper Large v3 (1.55B params, multilingual)',
    'sherpa-onnx-whisper-large-v3',
    'large-v3',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    950,
    extraFiles: [
      ModelFile(
        url: '$base/sherpa-onnx-whisper-large-v3/resolve/main/large-v3-encoder.weights',
        filename: 'large-v3-encoder.weights',
      ),
      ModelFile(
        url: '$base/sherpa-onnx-whisper-large-v3/resolve/main/large-v3-decoder.weights',
        filename: 'large-v3-decoder.weights',
      ),
    ],
  );

  addVariants(
    'whisper-large-v3-turbo',
    'Whisper Turbo (809M params, multilingual)',
    'sherpa-onnx-whisper-turbo',
    'turbo',
    ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh', 'ar', 'ru', 'it', 'nl', 'pl', 'tr'],
    550,
    extraFiles: [
      ModelFile(
        url: '$base/sherpa-onnx-whisper-turbo/resolve/main/turbo-encoder.weights',
        filename: 'turbo-encoder.weights',
      ),
    ],
  );
}
