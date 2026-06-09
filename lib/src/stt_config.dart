enum SttModelType {
  whisper,
  sherpa,
  nemo,
  canary,
  sensevoice,
  omnilingual,
  qwen3asr,
}

class SttConfig {
  final int sampleRate;
  final int chunkSize;
  final int numChannels;
  final int numThreads;
  final bool useParakeetForHighEnd;
  final bool useVAD;
  final int maxModelSizeMB;
  final Duration languageCacheDuration;
  final int maxLoadedModels;

  const SttConfig({
    this.sampleRate = 16000,
    this.chunkSize = 1600,
    this.numChannels = 1,
    this.numThreads = 4,
    this.useParakeetForHighEnd = true,
    this.useVAD = true,
    this.maxModelSizeMB = 500,
    this.languageCacheDuration = const Duration(minutes: 5),
    this.maxLoadedModels = 2,
  });
}
