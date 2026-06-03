enum ModelType {
  kroko,
  zipformer,
  parakeet,
  whisperTiny,
  whisperSmall,
}

class SttModelConfig {
  final String id;
  final ModelType type;
  final String name;
  final List<String> languages;
  final int sizeMB;
  final double expectedWER;
  final double expectedRTF;
  final String encoderPath;
  final String decoderPath;
  final String? joinerPath;
  final String tokensPath;
  final int priority;

  const SttModelConfig({
    required this.id,
    required this.type,
    required this.name,
    required this.languages,
    required this.sizeMB,
    required this.expectedWER,
    required this.expectedRTF,
    required this.encoderPath,
    required this.decoderPath,
    this.joinerPath,
    required this.tokensPath,
    this.priority = 2,
  });

  static const List<SttModelConfig> models = [
    SttModelConfig(
      id: 'kroko_64l_en',
      type: ModelType.kroko,
      name: 'Kroko 64L English',
      languages: ['en'],
      sizeMB: 20,
      expectedWER: 5.5,
      expectedRTF: 0.08,
      encoderPath: 'models/en/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/en/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/en/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/en/kroko_64l/tokens.txt',
      priority: 1,
    ),
    SttModelConfig(
      id: 'kroko_64l_fr',
      type: ModelType.kroko,
      name: 'Kroko 64L French',
      languages: ['fr'],
      sizeMB: 20,
      expectedWER: 6.2,
      expectedRTF: 0.08,
      encoderPath: 'models/fr/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/fr/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/fr/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/fr/kroko_64l/tokens.txt',
      priority: 1,
    ),
    SttModelConfig(
      id: 'kroko_64l_es',
      type: ModelType.kroko,
      name: 'Kroko 64L Spanish',
      languages: ['es'],
      sizeMB: 20,
      expectedWER: 5.8,
      expectedRTF: 0.08,
      encoderPath: 'models/es/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/es/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/es/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/es/kroko_64l/tokens.txt',
      priority: 1,
    ),
    SttModelConfig(
      id: 'kroko_64l_de',
      type: ModelType.kroko,
      name: 'Kroko 64L German',
      languages: ['de'],
      sizeMB: 20,
      expectedWER: 6.5,
      expectedRTF: 0.08,
      encoderPath: 'models/de/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/de/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/de/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/de/kroko_64l/tokens.txt',
      priority: 1,
    ),
    SttModelConfig(
      id: 'kroko_64l_it',
      type: ModelType.kroko,
      name: 'Kroko 64L Italian',
      languages: ['it'],
      sizeMB: 20,
      expectedWER: 5.2,
      expectedRTF: 0.08,
      encoderPath: 'models/it/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/it/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/it/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/it/kroko_64l/tokens.txt',
      priority: 1,
    ),
    SttModelConfig(
      id: 'parakeet_tdt_0.6b_v3',
      type: ModelType.parakeet,
      name: 'Parakeet TDT 0.6B v3',
      languages: [
        'en', 'es', 'it', 'fr', 'de', 'nl', 'ru', 'pl', 'uk', 'sk',
        'bg', 'fi', 'ro', 'hr', 'cs', 'sv', 'et', 'hu', 'lt', 'da',
        'mt', 'sl', 'lv', 'el', 'pt',
      ],
      sizeMB: 640,
      expectedWER: 6.1,
      expectedRTF: 0.03,
      encoderPath: 'models/parakeet/encoder.int8.onnx',
      decoderPath: 'models/parakeet/decoder.int8.onnx',
      joinerPath: 'models/parakeet/joiner.int8.onnx',
      tokensPath: 'models/parakeet/tokens.txt',
      priority: 2,
    ),
    SttModelConfig(
      id: 'whisper_tiny',
      type: ModelType.whisperTiny,
      name: 'Whisper Tiny',
      languages: [],
      sizeMB: 75,
      expectedWER: 11.0,
      expectedRTF: 0.8,
      encoderPath: 'models/whisper-tiny/encoder.int8.onnx',
      decoderPath: 'models/whisper-tiny/decoder.int8.onnx',
      tokensPath: 'models/whisper-tiny/tokens.txt',
      priority: 3,
    ),
    SttModelConfig(
      id: 'whisper_small',
      type: ModelType.whisperSmall,
      name: 'Whisper Small',
      languages: [],
      sizeMB: 250,
      expectedWER: 8.5,
      expectedRTF: 1.2,
      encoderPath: 'models/whisper-small/encoder.int8.onnx',
      decoderPath: 'models/whisper-small/decoder.int8.onnx',
      tokensPath: 'models/whisper-small/tokens.txt',
      priority: 4,
    ),
  ];

  static const Map<String, String> languageToModel = {
    'en': 'kroko_64l_en',
    'fr': 'kroko_64l_fr',
    'es': 'kroko_64l_es',
    'de': 'kroko_64l_de',
    'it': 'kroko_64l_it',
  };

  static SttModelConfig? getModelForLanguage(String languageCode) {
    final modelId = languageToModel[languageCode];
    if (modelId != null) {
      return models.firstWhere((m) => m.id == modelId);
    }
    return null;
  }

  static bool canDeviceHandleModel(SttModelConfig model, int availableMemoryMB) {
    final safeMemory = availableMemoryMB - 500;
    return model.sizeMB <= safeMemory;
  }

  static const Map<String, List<String>> supportedLanguagesByFamily = {
    'kroko': ['en', 'fr', 'es', 'de', 'it', 'pt', 'ru', 'nl', 'pl', 'uk'],
    'zipformer': ['en', 'zh', 'fr', 'es', 'de', 'it'],
    'parakeet': [
      'en', 'es', 'it', 'fr', 'de', 'nl', 'ru', 'pl', 'uk', 'sk',
      'bg', 'fi', 'ro', 'hr', 'cs', 'sv', 'et', 'hu', 'lt', 'da',
      'mt', 'sl', 'lv', 'el', 'pt',
    ],
    'whisper': [
      'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca',
      'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms',
      'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'lv',
      'et', 'ml', 'cy', 'sk', 'te', 'fa', 'sr', 'az', 'sl', 'kn', 'gu', 'am',
    ],
  };
}
