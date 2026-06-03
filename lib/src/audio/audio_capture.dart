import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:stt_flutter/src/stt_logger.dart';

class AudioCaptureService {
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _streamSubscription;
  bool _isRecording = false;

  static const int sampleRate = 16000;
  static const int numChannels = 1;

  Future<void> startRecording(Function(List<int>) onAudioData) async {
    if (_isRecording) return;

    final stream = await _recorder.startStream(const RecordConfig(
      sampleRate: sampleRate,
      numChannels: numChannels,
      encoder: AudioEncoder.pcm16bits,
    ));

    _isRecording = true;
    SttLogger.d('AudioCapture started');

    _streamSubscription = stream.listen((audioData) {
      onAudioData(_bytesToInt16List(audioData));
    });
  }

  Future<void> stopRecording() async {
    if (!_isRecording) return;
    await _streamSubscription?.cancel();
    _streamSubscription = null;
    await _recorder.stop();
    _isRecording = false;
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
    await _recorder.dispose();
  }

  List<int> _bytesToInt16List(Uint8List bytes) {
    final values = <int>[];
    for (var i = 0; i < bytes.length; i += 2) {
      if (i + 1 < bytes.length) {
        final sample = bytes[i] | (bytes[i + 1] << 8);
        values.add(sample);
      }
    }
    return values;
  }

  bool get isRecording => _isRecording;
}
