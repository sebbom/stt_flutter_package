class SttException implements Exception {
  final String message;
  final int? code;
  final dynamic cause;

  const SttException(this.message, [this.code, this.cause]);

  @override
  String toString() {
    final buf = StringBuffer('SttException');
    if (code != null) buf.write('($code)');
    buf.write(': $message');
    if (cause != null) buf.write(' ($cause)');
    return buf.toString();
  }

  factory SttException.notInitialized(String component) =>
      SttException('$component not initialized. Call initialize() first.', 1001);

  factory SttException.invalidArgument(String message) =>
      SttException(message, 1002);

  factory SttException.modelLoadFailed(String reason) =>
      SttException('Failed to load model: $reason', 2001);

  factory SttException.inferenceFailed(String reason) =>
      SttException('Inference failed: $reason', 3001);

  factory SttException.fileNotFound(String path) =>
      SttException('File not found: $path', 4001);

  factory SttException.downloadFailed(String reason) =>
      SttException('Download failed: $reason', 5001);
}

class OperationCancelledException implements Exception {
  @override
  String toString() => 'Operation was cancelled';
}
