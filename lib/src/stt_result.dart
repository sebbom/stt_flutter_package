class SttResult {
  final String text;
  final double inferenceTimeMs;
  final String? lang;
  final double? confidence;
  final double? durationMs;

  const SttResult({
    required this.text,
    required this.inferenceTimeMs,
    this.lang,
    this.confidence,
    this.durationMs,
  });
}
