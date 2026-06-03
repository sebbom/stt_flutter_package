import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:stt_flutter/config/models.dart';
import 'package:stt_flutter/src/stt_logger.dart';

class ModelDownloader {
  static const String _baseUrl = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';

  static final Map<String, String> _modelUrls = {
    'kroko_64l_en': '$_baseUrl/runanywhere-models-v1/sherpa-onnx-whisper-tiny.en.tar.gz',
    'kroko_64l_fr': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-fr/resolve/main/sherpa-onnx-kroko-64l-fr.tar.gz',
    'kroko_64l_es': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-es/resolve/main/sherpa-onnx-kroko-64l-es.tar.gz',
    'kroko_64l_de': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-de/resolve/main/sherpa-onnx-kroko-64l-de.tar.gz',
    'kroko_64l_it': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-it/resolve/main/sherpa-onnx-kroko-64l-it.tar.gz',
    'parakeet_tdt_0.6b_v3': '$_baseUrl/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2',
    'whisper_tiny': '$_baseUrl/asr-models/sherpa-onnx-whisper-tiny.tar.bz2',
    'whisper_small': '$_baseUrl/asr-models/sherpa-onnx-whisper-small.tar.bz2',
    'whisper_tiny_lid': '$_baseUrl/asr-models/sherpa-onnx-whisper-tiny.tar.bz2',
  };

  Future<String> downloadModel(String modelId) async {
    final url = _modelUrls[modelId];
    if (url == null) throw Exception('Model $modelId URL not found');

    final modelConfig = SttModelConfig.models.firstWhere(
      (m) => m.id == modelId,
      orElse: () => throw Exception('Model config not found for $modelId'),
    );

    final dir = await getModelDirectory(modelConfig);
    final archivePath = p.join(dir, '$modelId.tar.bz2');

    if (!await File(archivePath).exists()) {
      SttLogger.i('Downloading model $modelId...');
      await _downloadFile(url, archivePath);
      await _extractArchive(archivePath, dir);
      await File(archivePath).delete();
      SttLogger.i('Model $modelId downloaded and extracted');
    }

    return dir;
  }

  Future<String> getModelDirectory(SttModelConfig model) async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelDir = p.join(appDir.path, 'models', model.id);
    final dir = Directory(modelDir);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return modelDir;
  }

  Future<void> _downloadFile(String url, String savePath) async {
    final response = await http.get(Uri.parse(url));
    if (response.statusCode != 200) {
      throw Exception('Failed to download: HTTP ${response.statusCode}');
    }
    await File(savePath).writeAsBytes(response.bodyBytes);
  }

  Future<void> _extractArchive(String archivePath, String extractTo) async {
    SttLogger.d('Extraction not implemented - manual extraction required');
  }

  Future<bool> isModelDownloaded(String modelId) async {
    try {
      final modelConfig = SttModelConfig.models.firstWhere((m) => m.id == modelId);
      final dir = await getModelDirectory(modelConfig);
      final encoderFile = File(p.join(dir, modelConfig.encoderPath));
      return await encoderFile.exists();
    } catch (_) {
      return false;
    }
  }

  Future<void> deleteModel(String modelId) async {
    final modelConfig = SttModelConfig.models.firstWhere(
      (m) => m.id == modelId,
      orElse: () => throw Exception('Model $modelId not found'),
    );
    final dir = await getModelDirectory(modelConfig);
    await Directory(dir).delete(recursive: true);
  }
}
