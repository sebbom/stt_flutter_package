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
}
