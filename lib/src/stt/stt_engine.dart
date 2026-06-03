import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import '../stt_flutter_impl.dart';
import '../stt_result.dart';
import '../model_registry.dart';
import '../model_downloader.dart';

class SttEngine {
  static bool _initialized = false;

  SttFlutter? _stt;
  bool _cancelRequested = false;

  SttEngine._();

  static final SttEngine instance = SttEngine._();

  bool get isReady => _stt != null && !_cancelRequested;

  Future<String> loadModel(ModelDescriptor model, {String? modelDir}) async {
    try {
      if (!_initialized) {
        initBindings();
        _initialized = true;
      }
      _cancelRequested = false;
      await _stt?.dispose();
      _stt = SttFlutter();
      if (modelDir == null) {
        modelDir = await ModelDownloader.defaultStoragePath(model);
        if (!await ModelDownloader.isDownloaded(model, storagePath: modelDir)) {
          await ModelDownloader.download(model, storagePath: modelDir);
        }
      }
      final stt = _stt!;
      await stt.initialize(model: model, modelDir: modelDir);
      return 'Success';
    } catch (e) {
      await _stt?.dispose();
      _stt = null;
      return e.toString();
    }
  }

  Future<SttResult> transcribeFile(String path, {String? language}) async {
    return _stt!.transcribeFile(path, language: language);
  }

  Future<SttResult> transcribeBuffer(Float32List samples, int sampleRate, {String? language}) async {
    return _stt!.transcribeBuffer(samples, sampleRate, language: language);
  }

  void cancel() {
    _cancelRequested = true;
    _stt?.cancel();
  }

  void destroy() {
    _stt?.dispose();
    _stt = null;
    _cancelRequested = false;
  }
}
