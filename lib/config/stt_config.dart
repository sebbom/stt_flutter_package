import '../services/utils/device_utils.dart';

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

  static SttConfig fromDeviceCapabilities(DeviceCapabilities capabilities) {
    return SttConfig(
      numThreads: capabilities.cpuCores,
      useParakeetForHighEnd: capabilities.isHighEnd && capabilities.memoryMB >= 6000,
      maxModelSizeMB: capabilities.memoryMB > 4000 ? 640 : 200,
      maxLoadedModels: capabilities.memoryMB > 4000 ? 3 : 1,
    );
  }
}
