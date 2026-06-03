class SttResult {
  final String text;
  final double inferenceTimeMs;
  final String? lang;

  const SttResult({required this.text, required this.inferenceTimeMs, this.lang});
}
