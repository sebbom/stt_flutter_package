class VoiceActivityDetector {
  static const double speechThreshold = 0.01;
  static const double silenceThreshold = 0.001;
  static const int speechFramesRequired = 3;
  static const int silenceFramesRequired = 10;

  int _speechFrameCount = 0;
  int _silenceFrameCount = 0;
  bool _isSpeaking = false;

  bool processChunk(List<int> audioChunk) {
    final energy = _calculateEnergy(audioChunk);

    if (energy > speechThreshold) {
      _speechFrameCount++;
      _silenceFrameCount = 0;
      if (_speechFrameCount >= speechFramesRequired && !_isSpeaking) {
        _isSpeaking = true;
        _speechFrameCount = 0;
        return true;
      }
    } else {
      _silenceFrameCount++;
      _speechFrameCount = 0;
      if (_silenceFrameCount >= silenceFramesRequired && _isSpeaking) {
        _isSpeaking = false;
        _silenceFrameCount = 0;
        return false;
      }
    }

    return _isSpeaking;
  }

  double _calculateEnergy(List<int> audioChunk) {
    double sum = 0;
    for (final sample in audioChunk) {
      final normalized = sample / 32768.0;
      sum += normalized * normalized;
    }
    return sum / audioChunk.length;
  }

  bool get isSpeaking => _isSpeaking;

  void reset() {
    _speechFrameCount = 0;
    _silenceFrameCount = 0;
    _isSpeaking = false;
  }
}
