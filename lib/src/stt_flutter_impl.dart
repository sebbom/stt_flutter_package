import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import 'model_registry.dart';
import 'model_downloader.dart';
import 'stt_result.dart';
import 'stt_exception.dart';
import 'cancellation_token.dart';
import 'audio/audio_buffer.dart';
import 'audio/audio_processor.dart';
import 'engines/inference_engine.dart';
import 'engines/engine_factory.dart';

class SttFlutter {
  ort.OnnxRuntime? _ort;
  InferenceEngine? _engine;
  bool _initialized = false;
  String? _defaultLanguage;
  CancellationToken? _currentToken;

  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,
    String? language,
  }) async {
    if (_initialized) throw SttException.notInitialized('SttFlutter');
    if (model.id.isEmpty) throw SttException.invalidArgument('model.id must not be empty');
    _defaultLanguage = language;

    final dir = modelDir ?? await ModelDownloader.defaultStoragePath(model);
    final modelFiles = <String, String>{};

    for (final f in model.files) {
      final path = '$dir/${f.filename}';
      final file = File(path);
      if (await file.exists()) {
        modelFiles[f.filename] = path;
      }
    }

    final modelDir_ = Directory(dir);
    if (await modelDir_.exists()) {
      await for (final entry in modelDir_.list()) {
        if (entry is File) {
          final name = entry.uri.pathSegments.last;
          if (name.endsWith('.onnx') || name.endsWith('.onnx_data') || name.endsWith('.txt') || name.endsWith('.json')) {
            modelFiles[name] = entry.path;
          }
        }
      }
    }

    if (modelFiles.isEmpty) {
      throw SttException.modelLoadFailed('No model files found in $dir');
    }

    try {
      _ort = ort.OnnxRuntime();
      _engine = createEngine(model.type, _ort!);
      await _engine!.load(modelFiles);
      _initialized = true;
    } catch (e) {
      await _cleanup();
      rethrow;
    }
  }

  Future<SttResult> transcribeFile(String path, {String? language, CancellationToken? token}) async {
    if (!_initialized) throw SttException.notInitialized('SttFlutter');
    final file = File(path);
    if (!await file.exists()) throw SttException.fileNotFound(path);
    final audio = await AudioProcessor.loadWav(path);
    if (audio.length == 0) throw SttException.invalidArgument('Audio file contains no samples');
    return _transcribe(audio, language: language, token: token);
  }

  Future<SttResult> transcribeBuffer(Float32List samples, int sampleRate, {String? language, CancellationToken? token}) async {
    if (!_initialized) throw SttException.notInitialized('SttFlutter');
    if (sampleRate <= 0 || sampleRate > 192000) {
      throw SttException.invalidArgument('sampleRate must be between 1 and 192000');
    }
    if (samples.length == 0) {
      throw SttException.invalidArgument('samples buffer must not be empty');
    }
    final audio = AudioBuffer(samples: samples, sampleRate: sampleRate);
    return _transcribe(audio, language: language, token: token);
  }

  Future<SttResult> _transcribe(AudioBuffer audio, {String? language, CancellationToken? token}) async {
    _currentToken = token;
    token?.throwIfCancelled();
    final resampled = await Isolate.run(() => AudioProcessor.resampleSync(audio));
    token?.throwIfCancelled();
    return _engine!.transcribe(resampled, language: language ?? _defaultLanguage, token: token);
  }

  void cancel() {
    _currentToken?.cancel();
  }

  Future<void> _cleanup() async {
    await _engine?.dispose();
    _engine = null;
    _ort = null;
    _initialized = false;
  }

  Future<void> dispose() async {
    cancel();
    if (!_initialized) return;
    await _engine?.dispose();
    _engine = null;
    _ort = null;
    _initialized = false;
  }
}
