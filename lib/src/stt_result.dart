import 'dart:convert';

class SttResult {
  final String text;
  final double inferenceTimeMs;
  final String? lang;
  final double? confidence;
  final double? durationMs;
  final String? emotion;
  final List<String> events;

  const SttResult({
    required this.text,
    required this.inferenceTimeMs,
    this.lang,
    this.confidence,
    this.durationMs,
    this.emotion,
    this.events = const <String>[],
  });

  Map<String, dynamic> toJson() => {
        'text': text,
        'inferenceTimeMs': inferenceTimeMs,
        if (lang != null) 'lang': lang,
        if (confidence != null) 'confidence': confidence,
        if (durationMs != null) 'durationMs': durationMs,
        if (emotion != null) 'emotion': emotion,
        'events': events,
      };

  factory SttResult.fromJson(Map<String, dynamic> json) => SttResult(
        text: json['text'] as String,
        inferenceTimeMs: (json['inferenceTimeMs'] as num).toDouble(),
        lang: json['lang'] as String?,
        confidence: (json['confidence'] as num?)?.toDouble(),
        durationMs: (json['durationMs'] as num?)?.toDouble(),
        emotion: json['emotion'] as String?,
        events: (json['events'] as List<dynamic>?)?.cast<String>() ?? [],
      );

  /// Serialize to JSON string
  String toJsonString() => json.encode(toJson());

  /// Deserialize from JSON string
  static SttResult fromJsonString(String jsonString) {
    return SttResult.fromJson(
      json.decode(jsonString) as Map<String, dynamic>,
    );
  }
}
