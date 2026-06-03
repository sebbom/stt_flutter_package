import 'package:path/path.dart' as p;
import 'package:stt_flutter/config/models.dart';
import 'package:stt_flutter/src/stt_logger.dart';
import 'package:stt_flutter/services/models/model_downloader.dart';

class ModelManager {
  final ModelDownloader _downloader = ModelDownloader();
  final Map<String, bool> _loadedModels = {};
  final Map<String, dynamic> _modelInstances = {};
  int _availableMemoryMB = 0;

  Future<void> init({int? forcedMemoryMB}) async {
    if (forcedMemoryMB != null) {
      _availableMemoryMB = forcedMemoryMB;
    } else {
      _availableMemoryMB = await _getAvailableMemory();
    }
    SttLogger.i('ModelManager initialized with ${_availableMemoryMB}MB available');
  }

  Future<int> _getAvailableMemory() async {
    return 4000;
  }

  Future<String> ensureModelLoaded(String modelId) async {
    if (_loadedModels[modelId] == true) return modelId;

    if (!await _downloader.isModelDownloaded(modelId)) {
      await _downloader.downloadModel(modelId);
    }

    final modelConfig = SttModelConfig.models.firstWhere((m) => m.id == modelId);
    if (!SttModelConfig.canDeviceHandleModel(modelConfig, _availableMemoryMB)) {
      throw Exception('Insufficient memory for model: $modelId');
    }

    await _loadModel(modelId);
    _loadedModels[modelId] = true;
    SttLogger.i('Model $modelId loaded');
    return modelId;
  }

  Future<void> _loadModel(String modelId) async {
    final modelConfig = SttModelConfig.models.firstWhere((m) => m.id == modelId);
    final modelDir = await _downloader.getModelDirectory(modelConfig);

    final encoderPath = p.join(modelDir, modelConfig.encoderPath);
    final decoderPath = p.join(modelDir, modelConfig.decoderPath);

    _modelInstances[modelId] = {
      'encoderPath': encoderPath,
      'decoderPath': decoderPath,
      'config': modelConfig,
      'modelDir': modelDir,
    };
  }

  Future<void> unloadModel(String modelId) async {
    _loadedModels[modelId] = false;
    _modelInstances.remove(modelId);
  }

  Future<void> unloadAllModels() async {
    _loadedModels.clear();
    _modelInstances.clear();
  }

  dynamic getModelInstance(String modelId) {
    return _modelInstances[modelId];
  }

  List<String> get loadedModels =>
      _loadedModels.entries.where((e) => e.value).map((e) => e.key).toList();

  int get availableMemoryMB => _availableMemoryMB;
}
