import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import 'model_registry.dart';
import 'model_downloader.dart';
import 'stt_result.dart';
import 'audio/audio_buffer.dart';
import 'audio/audio_processor.dart';
import 'engines/inference_engine.dart';
import 'engines/engine_factory.dart';

class SttFlutter {
  ort.OnnxRuntime? _ort;
  InferenceEngine? _engine;
  bool _initialized = false;

  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,
    String? language,
  }) async {
    final dir = modelDir ?? await ModelDownloader.defaultStoragePath(model);
    final modelFiles = <String, String>{};
    for (final f in model.files) {
      final path = '$dir/${f.filename}';
      final file = File(path);
      if (!await file.exists()) {
        throw FileSystemException('Model file not found: ${f.filename}', path);
      }
      modelFiles[f.filename] = path;
    }

    _ort = ort.OnnxRuntime();
    _engine = createEngine(model.type, _ort!);
    await _engine!.load(modelFiles);
    _initialized = true;
  }

  Future<SttResult> transcribeFile(String path) async {
    if (!_initialized) throw StateError('SttFlutter not initialized');
    final audio = await AudioProcessor.loadWav(path);
    return _transcribe(audio);
  }

  Future<SttResult> transcribeBuffer(Float32List samples, int sampleRate) async {
    if (!_initialized) throw StateError('SttFlutter not initialized');
    final audio = AudioBuffer(samples: samples, sampleRate: sampleRate);
    return _transcribe(audio);
  }

  Future<SttResult> _transcribe(AudioBuffer audio) async {
    // Offload audio preprocessing to a background isolate
    final resampled = await Isolate.run(() => AudioProcessor.resampleSync(audio));
    return _engine!.transcribe(resampled);
  }

  Future<void> dispose() async {
    if (!_initialized) return;
    await _engine?.dispose();
    _initialized = false;
  }
}
