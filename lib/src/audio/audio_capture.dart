import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:stt_flutter/src/stt_logger.dart';

class AudioCaptureService {
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _streamSubscription;
  StreamController<Float32List>? _chunkController;
  bool _isRecording = false;

  static const int sampleRate = 16000;
  static const int numChannels = 1;

  /// Stream of mono 16 kHz Float32 audio chunks.
  Stream<Float32List> get audioStream =>
      _chunkController?.stream ?? const Stream.empty();

  Future<bool> hasPermission({bool request = true}) =>
      _recorder.hasPermission(request: request);

  Future<void> startRecording() async {
    if (_isRecording) return;
    await _chunkController?.close();
    _chunkController = StreamController<Float32List>();

    final stream = await _recorder.startStream(const RecordConfig(
      sampleRate: sampleRate,
      numChannels: numChannels,
      encoder: AudioEncoder.pcm16bits,
    ));

    _isRecording = true;
    SttLogger.d('AudioCapture started');

    _streamSubscription = stream.listen((audioData) {
      final floats = bytesToFloat32(audioData);
      if (!_chunkController!.isClosed) {
        _chunkController!.add(floats);
      }
    });
  }

  Future<void> stopRecording() async {
    if (!_isRecording) return;
    await _streamSubscription?.cancel();
    _streamSubscription = null;
    await _recorder.stop();
    _isRecording = false;
    await _chunkController?.close();
    _chunkController = null;
    SttLogger.d('AudioCapture stopped');
  }

  Future<void> pauseRecording() async {
    if (!_isRecording) return;
    await _recorder.pause();
  }

  Future<void> resumeRecording() async {
    if (_isRecording) return;
    await _recorder.resume();
  }

  Future<void> dispose() async {
    await _streamSubscription?.cancel();
    if (_isRecording) await _recorder.stop();
    await _chunkController?.close();
    _chunkController = null;
    await _recorder.dispose();
  }

  /// Convert raw PCM 16-bit little-endian bytes to a normalized Float32List.
  /// Visible for testing.
  static Float32List bytesToFloat32(Uint8List bytes) {
    final n = bytes.length ~/ 2;
    final out = Float32List(n);
    for (var i = 0; i < n; i++) {
      final lo = bytes[i * 2];
      final hi = bytes[i * 2 + 1];
      final s = (lo | (hi << 8)).toSigned(16);
      out[i] = s / 32768.0;
    }
    return out;
  }

  bool get isRecording => _isRecording;
}
