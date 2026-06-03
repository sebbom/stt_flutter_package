import 'package:stt_flutter/src/stt_logger.dart';

class LanguageDetector {
  static const List<String> supportedLanguages = [
    'en', 'fr', 'es', 'de', 'it', 'pt', 'ru', 'nl', 'pl', 'uk',
    'sk', 'bg', 'fi', 'ro', 'hr', 'cs', 'sv', 'et', 'hu', 'lt',
    'da', 'mt', 'sl', 'lv', 'el', 'ja', 'zh', 'ar', 'hi', 'ko',
  ];

  bool _isInitialized = false;

  Future<void> init() async {
    if (_isInitialized) return;
    _isInitialized = true;
    SttLogger.d('LanguageDetector initialized');
  }

  Future<String> detectLanguage(List<int> audioData, {int sampleRate = 16000}) async {
    if (!_isInitialized) await init();

    final language = _detectLanguageFromText(audioData);
    SttLogger.d('Detected language: $language');
    return language;
  }

  String _detectLanguageFromText(List<int> audioData) {
    return 'en';
  }

  Future<void> dispose() async {
    _isInitialized = false;
  }
}
