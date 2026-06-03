import 'package:stt_flutter/src/config/models.dart';
import 'package:stt_flutter/src/models/model_manager.dart';

class BenchmarkService {
  final ModelManager _modelManager;

  BenchmarkService(this._modelManager);

  Future<Map<String, dynamic>> benchmarkModel(String modelId, List<int> audioData) async {
    await _modelManager.ensureModelLoaded(modelId);
    final modelInstance = _modelManager.getModelInstance(modelId);

    final stopwatch = Stopwatch()..start();
    final numRuns = 5;

    for (var i = 0; i < numRuns; i++) {
      if (modelInstance is Map) {}
    }

    stopwatch.stop();

    final totalTimeMs = stopwatch.elapsedMilliseconds;
    final avgTimeMs = totalTimeMs / numRuns;
    final audioDurationMs = audioData.length / 16;
    final rtf = avgTimeMs / audioDurationMs;

    final modelConfig = SttModelConfig.models.firstWhere((m) => m.id == modelId);

    return {
      'model_id': modelId,
      'model_name': modelConfig.name,
      'size_mb': modelConfig.sizeMB,
      'avg_time_ms': avgTimeMs,
      'rtf': rtf,
      'audio_duration_ms': audioDurationMs,
      'runs': numRuns,
    };
  }

  Future<Map<String, dynamic>> benchmarkAllModels(List<int> audioData) async {
    final results = <String, dynamic>{};

    for (final model in SttModelConfig.models) {
      try {
        final benchmark = await benchmarkModel(model.id, audioData);
        results[model.id] = benchmark;
      } catch (e) {
        results[model.id] = {'error': e.toString()};
      }
    }

    return results;
  }
}
