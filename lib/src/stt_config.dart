import 'dart:convert';

enum SttModelType {
  whisper,
  sherpa,
  nemo,
  canary,
  sensevoice,
  omnilingual,
  qwen3asr,
}

/// Extension to support JSON serialization for SttModelType
extension SttModelTypeJson on SttModelType {
  String toJsonValue() => name;

  static SttModelType fromJsonValue(String value) {
    return SttModelType.values.firstWhere(
      (e) => e.name == value,
      orElse: () => throw FormatException('Invalid SttModelType value: $value'),
    );
  }
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

  Map<String, dynamic> toJson() => {
        'sampleRate': sampleRate,
        'chunkSize': chunkSize,
        'numChannels': numChannels,
        'numThreads': numThreads,
        'useParakeetForHighEnd': useParakeetForHighEnd,
        'useVAD': useVAD,
        'maxModelSizeMB': maxModelSizeMB,
        'languageCacheDurationMs': languageCacheDuration.inMilliseconds,
        'maxLoadedModels': maxLoadedModels,
      };

  factory SttConfig.fromJson(Map<String, dynamic> json) => SttConfig(
        sampleRate: json['sampleRate'] as int? ?? 16000,
        chunkSize: json['chunkSize'] as int? ?? 1600,
        numChannels: json['numChannels'] as int? ?? 1,
        numThreads: json['numThreads'] as int? ?? 4,
        useParakeetForHighEnd: json['useParakeetForHighEnd'] as bool? ?? true,
        useVAD: json['useVAD'] as bool? ?? true,
        maxModelSizeMB: json['maxModelSizeMB'] as int? ?? 500,
        languageCacheDuration: Duration(
          milliseconds: json['languageCacheDurationMs'] as int? ?? 300000,
        ),
        maxLoadedModels: json['maxLoadedModels'] as int? ?? 2,
      );

  /// Serialize to JSON string
  String toJsonString() => json.encode(toJson());

  /// Deserialize from JSON string
  static SttConfig fromJsonString(String jsonString) {
    return SttConfig.fromJson(
      json.decode(jsonString) as Map<String, dynamic>,
    );
  }
}
