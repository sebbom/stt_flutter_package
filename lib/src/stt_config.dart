enum SttModelType { whisper, sherpa, voxtral }

class SttConfig {
  final SttModelType modelType;
  final String modelDir;
  final String? language;

  const SttConfig({
    required this.modelType,
    required this.modelDir,
    this.language,
  });
}
