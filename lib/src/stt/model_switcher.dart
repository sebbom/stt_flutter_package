import 'package:stt_flutter/src/config/models.dart';
import 'package:stt_flutter/src/stt_logger.dart';
import 'package:stt_flutter/src/language_detector.dart';
import 'package:stt_flutter/src/models/model_manager.dart';

class ModelSwitcher {
  final ModelManager _modelManager;
  final LanguageDetector _languageDetector;

  final Map<String, String> _languageCache = {};
  final Map<String, DateTime> _cacheTimestamps = {};
  final Duration _cacheDuration = const Duration(minutes: 5);

  String? _currentModelId;
  dynamic _currentModelInstance;

  ModelSwitcher(this._modelManager, this._languageDetector);

  Future<String> getOptimalModelId({
    required List<int> audioData,
    String? forcedLanguage,
    bool useParakeetForHighEnd = true,
  }) async {
    String? language;

    if (forcedLanguage != null) {
      language = forcedLanguage;
    } else {
      language = _getCachedLanguage();
      if (language == null) {
        language = await _languageDetector.detectLanguage(audioData);
        _cacheLanguage(language);
      }
    }

    final languageModel = SttModelConfig.getModelForLanguage(language);
    if (languageModel != null &&
        SttModelConfig.canDeviceHandleModel(languageModel, _modelManager.availableMemoryMB)) {
      return languageModel.id;
    }

    if (useParakeetForHighEnd) {
      final parakeetModel = SttModelConfig.models.firstWhere(
        (m) => m.id == 'parakeet_tdt_0.6b_v3',
      );
      if (SttModelConfig.canDeviceHandleModel(parakeetModel, _modelManager.availableMemoryMB) &&
          parakeetModel.languages.contains(language)) {
        return parakeetModel.id;
      }
    }

    return 'whisper_tiny';
  }

  Future<dynamic> getModelForAudio({
    required List<int> audioData,
    String? forcedLanguage,
    bool autoSwitch = true,
  }) async {
    final optimalModelId = await getOptimalModelId(
      audioData: audioData,
      forcedLanguage: forcedLanguage,
    );

    if (autoSwitch && optimalModelId != _currentModelId) {
      await switchModel(optimalModelId);
    }

    return _currentModelInstance;
  }

  Future<void> switchModel(String modelId) async {
    if (_currentModelId != null && _currentModelId != modelId) {
      await _modelManager.unloadModel(_currentModelId!);
    }

    await _modelManager.ensureModelLoaded(modelId);
    _currentModelId = modelId;
    _currentModelInstance = _modelManager.getModelInstance(modelId);
    SttLogger.i('Switched to model: $modelId');
  }

  String? _getCachedLanguage() {
    final now = DateTime.now();
    for (final entry in _cacheTimestamps.entries) {
      if (now.difference(entry.value) < _cacheDuration) {
        return _languageCache[entry.key];
      }
    }
    return null;
  }

  void _cacheLanguage(String language) {
    final sessionId = DateTime.now().millisecondsSinceEpoch.toString();
    _languageCache[sessionId] = language;
    _cacheTimestamps[sessionId] = DateTime.now();
    _cleanCache();
  }

  void _cleanCache() {
    final now = DateTime.now();
    final oldKeys = _cacheTimestamps.entries
        .where((e) => now.difference(e.value) > _cacheDuration)
        .map((e) => e.key)
        .toList();
    for (final key in oldKeys) {
      _languageCache.remove(key);
      _cacheTimestamps.remove(key);
    }
  }

  String? get currentModelId => _currentModelId;
  dynamic get currentModelInstance => _currentModelInstance;

  Future<void> dispose() async {
    if (_currentModelId != null) {
      await _modelManager.unloadModel(_currentModelId!);
    }
    _currentModelId = null;
    _currentModelInstance = null;
  }
}
