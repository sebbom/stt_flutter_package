import 'package:stt_flutter/services/models/model_manager.dart';

class SttEngineService {
  final ModelManager _modelManager;

  SttEngineService(this._modelManager);

  Future<String> transcribeAudio(List<int> audioData, {String? modelId}) async {
    final targetModelId = modelId ?? 'whisper_tiny';
    await _modelManager.ensureModelLoaded(targetModelId);

    final modelInstance = _modelManager.getModelInstance(targetModelId);
    if (modelInstance is Map) {
      return '';
    }

    return '';
  }

  Future<void> dispose() async {
    await _modelManager.unloadAllModels();
  }
}
