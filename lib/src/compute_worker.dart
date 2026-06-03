import 'dart:isolate';
import 'dart:typed_data';
import 'audio/audio_buffer.dart';
import 'audio/audio_processor.dart';
import 'audio/mel_spectrogram.dart';
import 'audio/fbank.dart';
import 'utils/math_utils.dart' as math;
import 'stt_logger.dart';

enum ComputeTask {
  resample,
  melSpectrogram,
  fbank,
  transposeMel,
}

class ComputeRequest {
  final ComputeTask task;
  final dynamic data;
  final SendPort replyPort;

  ComputeRequest({required this.task, required this.data, required this.replyPort});
}

class ComputeResponse {
  final ComputeTask task;
  final dynamic result;
  final String? error;

  ComputeResponse({required this.task, this.result, this.error});
}

class ComputeWorker {
  static ComputeWorker? _instance;
  Isolate? _isolate;
  SendPort? _sendPort;
  bool _initialized = false;

  ComputeWorker._();

  static ComputeWorker get instance {
    _instance ??= ComputeWorker._();
    return _instance!;
  }

  Future<void> initialize() async {
    if (_initialized) return;
    final receivePort = ReceivePort();
    _isolate = await Isolate.spawn(_entry, receivePort.sendPort);
    _sendPort = await receivePort.first as SendPort;
    _initialized = true;
    SttLogger.d('ComputeWorker initialized');
  }

  static void _entry(SendPort sendPort) {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);

    receivePort.listen((message) {
      if (message is ComputeRequest) {
        try {
          dynamic result;
          switch (message.task) {
            case ComputeTask.resample:
              final data = message.data as Map;
              result = AudioProcessor.resampleSync(
                data['audio'] as AudioBuffer,
                targetRate: data['targetRate'] as int? ?? AudioProcessor.targetSampleRate,
              );
              break;
            case ComputeTask.melSpectrogram:
              final data = message.data as Map;
              final ms = MelSpectrogram(nMels: data['nMels'] as int? ?? 80);
              result = ms.compute(data['samples'] as Float32List);
              break;
            case ComputeTask.fbank:
              final data = message.data as Map;
              final fb = Fbank(nMels: data['nMels'] as int? ?? 80);
              result = fb.compute(data['samples'] as Float32List);
              break;
            case ComputeTask.transposeMel:
              final data = message.data as Map;
              result = math.transposeMel(
                data['mel'] as Float64List,
                data['nMels'] as int,
                data['totalFrames'] as int,
                data['offset'] as int,
                data['chunkSize'] as int,
              );
              break;
          }
          message.replyPort.send(ComputeResponse(task: message.task, result: result));
        } catch (e) {
          message.replyPort.send(ComputeResponse(task: message.task, error: e.toString()));
        }
      }
    });
  }

  Future<T> _execute<T>(ComputeTask task, dynamic data) async {
    if (!_initialized) throw StateError('ComputeWorker not initialized');
    final receivePort = ReceivePort();
    _sendPort!.send(ComputeRequest(task: task, data: data, replyPort: receivePort.sendPort));
    final response = await receivePort.first as ComputeResponse;
    if (response.error != null) throw Exception(response.error);
    return response.result as T;
  }

  Future<AudioBuffer> resample(AudioBuffer audio, {int targetRate = 16000}) async {
    return _execute(ComputeTask.resample, {
      'audio': audio,
      'targetRate': targetRate,
    });
  }

  Future<Float64List> melSpectrogram(Float32List samples, {int nMels = 80}) async {
    return _execute(ComputeTask.melSpectrogram, {
      'samples': samples,
      'nMels': nMels,
    });
  }

  Future<Float64List> fbank(Float32List samples, {int nMels = 80}) async {
    return _execute(ComputeTask.fbank, {
      'samples': samples,
      'nMels': nMels,
    });
  }

  Future<Float32List> transposeMel(Float64List mel, int nMels, int totalFrames, int offset, int chunkSize) async {
    return _execute(ComputeTask.transposeMel, {
      'mel': mel,
      'nMels': nMels,
      'totalFrames': totalFrames,
      'offset': offset,
      'chunkSize': chunkSize,
    });
  }

  Future<void> dispose() async {
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _sendPort = null;
    _initialized = false;
    _instance = null;
    SttLogger.d('ComputeWorker disposed');
  }
}
