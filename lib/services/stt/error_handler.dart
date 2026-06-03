import 'package:stt_flutter/config/models.dart';
import 'package:stt_flutter/src/stt_logger.dart';
import 'package:stt_flutter/services/models/model_manager.dart';
import 'package:stt_flutter/services/stt/model_switcher.dart';

class SttErrorHandler {
  final ModelManager _modelManager;
  final ModelSwitcher _modelSwitcher;

  SttErrorHandler(this._modelManager, this._modelSwitcher);

  Future<dynamic> handleError(
    Exception error,
    List<int> audioData,
    String? detectedLanguage,
  ) async {
    SttLogger.e('STT Error', error);

    if (error.toString().contains('out of memory') ||
        error.toString().contains('OOM')) {
      return _handleMemoryError(audioData, detectedLanguage);
    }

    if (error.toString().contains('model not found') ||
        error.toString().contains('FileNotFoundException')) {
      return _handleModelLoadError(audioData, detectedLanguage);
    }

    return _tryFallbackModel(audioData);
  }

  Future<dynamic> _handleMemoryError(
    List<int> audioData,
    String? detectedLanguage,
  ) async {
    await _modelManager.unloadAllModels();

    if (detectedLanguage != null) {
      final languageModel = SttModelConfig.getModelForLanguage(detectedLanguage);
      if (languageModel != null && languageModel.sizeMB < 100) {
        await _modelSwitcher.switchModel(languageModel.id);
        return _modelSwitcher.currentModelInstance;
      }
    }

    await _modelSwitcher.switchModel('whisper_tiny');
    return _modelSwitcher.currentModelInstance;
  }

  Future<dynamic> _handleModelLoadError(
    List<int> audioData,
    String? detectedLanguage,
  ) async {
    try {
      if (detectedLanguage != null) {
        final languageModel = SttModelConfig.getModelForLanguage(detectedLanguage);
        if (languageModel != null) {
          await _modelManager.ensureModelLoaded(languageModel.id);
          await _modelSwitcher.switchModel(languageModel.id);
          return _modelSwitcher.currentModelInstance;
        }
      }

      await _modelManager.ensureModelLoaded('whisper_tiny');
      await _modelSwitcher.switchModel('whisper_tiny');
      return _modelSwitcher.currentModelInstance;
    } catch (e) {
      return _tryFallbackModel(audioData);
    }
  }

  Future<dynamic> _tryFallbackModel(List<int> audioData) async {
    try {
      await _modelManager.ensureModelLoaded('whisper_tiny');
      await _modelSwitcher.switchModel('whisper_tiny');
      return _modelSwitcher.currentModelInstance;
    } catch (e) {
      throw Exception('All fallback models failed: $e');
    }
  }

  Future<String> fallbackToCloud(List<int> audioData) async {
    throw UnimplementedError('Cloud fallback not implemented');
  }
}
