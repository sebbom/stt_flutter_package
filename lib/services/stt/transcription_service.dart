import 'dart:io';
import 'package:stt_flutter/config/models.dart';
import 'package:stt_flutter/src/stt_logger.dart';
import 'package:stt_flutter/services/audio/audio_capture.dart';
import 'package:stt_flutter/services/lid/language_detector.dart';
import 'package:stt_flutter/services/stt/model_switcher.dart';

class TranscriptionService {
  final ModelSwitcher _modelSwitcher;
  final LanguageDetector _languageDetector;
  final AudioCaptureService _audioCapture = AudioCaptureService();

  static const int sampleRate = 16000;
  static const int chunkSize = 1600;

  bool _isRecording = false;
  bool _isProcessing = false;
  final List<int> _audioBuffer = [];
  final List<String> _transcriptionResults = [];

  Function(String)? onPartialTranscription;
  Function(String)? onFinalTranscription;
  Function(Exception)? onError;
  Function(String)? onLanguageDetected;

  TranscriptionService({
    required ModelSwitcher modelSwitcher,
    required LanguageDetector languageDetector,
  })  : _modelSwitcher = modelSwitcher,
        _languageDetector = languageDetector;

  Future<void> startRecording() async {
    if (_isRecording) return;

    try {
      _isRecording = true;
      _audioBuffer.clear();
      _transcriptionResults.clear();

      await _audioCapture.startRecording((audioData) {
        _processAudioChunk(audioData);
      });

      SttLogger.i('Recording started');
    } catch (e) {
      _isRecording = false;
      onError?.call(Exception('Failed to start recording: $e'));
    }
  }

  Future<void> stopRecording() async {
    if (!_isRecording) return;

    await _audioCapture.stopRecording();
    _isRecording = false;

    if (_audioBuffer.isNotEmpty) {
      await _processBuffer();
    }

    if (_transcriptionResults.isNotEmpty) {
      final fullText = _transcriptionResults.join(' ');
      onFinalTranscription?.call(fullText);
    }

    SttLogger.i('Recording stopped');
  }

  void _processAudioChunk(List<int> audioData) {
    if (_isProcessing) return;

    _audioBuffer.addAll(audioData);

    while (_audioBuffer.length >= chunkSize) {
      final chunk = _audioBuffer.sublist(0, chunkSize);
      _audioBuffer.removeRange(0, chunkSize);
      _processChunk(chunk);
    }
  }

  Future<void> _processChunk(List<int> chunk) async {
    if (_isProcessing) return;
    _isProcessing = true;

    try {
      final modelInstance = await _modelSwitcher.getModelForAudio(
        audioData: chunk,
        autoSwitch: true,
      );

      if (modelInstance is Map) {
        final text = _transcribeWithModel(modelInstance, chunk);
        if (text.isNotEmpty) {
          onPartialTranscription?.call(text);
          _transcriptionResults.add(text);
        }
      }

      if (_modelSwitcher.currentModelId != 'whisper_tiny' &&
          _modelSwitcher.currentModelId != 'parakeet_tdt_0.6b_v3') {
        return;
      }

      final detectedLang = await _languageDetector.detectLanguage(chunk);
      onLanguageDetected?.call(detectedLang);

      final languageModel = SttModelConfig.getModelForLanguage(detectedLang);
      if (languageModel != null) {
        await _modelSwitcher.switchModel(languageModel.id);
      }
    } catch (e) {
      onError?.call(Exception('Processing error: $e'));
    } finally {
      _isProcessing = false;

      if (_audioBuffer.isNotEmpty) {
        final nextSize = _audioBuffer.length > chunkSize ? chunkSize : _audioBuffer.length;
        _processChunk(_audioBuffer.sublist(0, nextSize));
      }
    }
  }

  Future<void> _processBuffer() async {
    while (_audioBuffer.isNotEmpty) {
      final currentSize = _audioBuffer.length > chunkSize ? chunkSize : _audioBuffer.length;
      final chunk = _audioBuffer.sublist(0, currentSize);
      _audioBuffer.removeRange(0, currentSize);
      await _processChunk(chunk);
    }
  }

  String _transcribeWithModel(dynamic modelInstance, List<int> audioData) {
    return '';
  }

  Future<String> transcribeFile(String filePath) async {
    final audioData = await File(filePath).readAsBytes();

    final language = await _languageDetector.detectLanguage(audioData);
    onLanguageDetected?.call(language);

    final modelInstance = await _modelSwitcher.getModelForAudio(
      audioData: audioData,
      forcedLanguage: language,
    );

    final result = _transcribeWithModel(modelInstance, audioData);
    return result;
  }

  Future<void> dispose() async {
    await _audioCapture.dispose();
    await _modelSwitcher.dispose();
  }

  bool get isRecording => _isRecording;
  bool get isProcessing => _isProcessing;
}
