import 'dart:async';
import 'package:flutter/material.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

class TranscriptionScreen extends StatefulWidget {
  final ModelDescriptor model;

  const TranscriptionScreen({super.key, required this.model});

  @override
  State<TranscriptionScreen> createState() => _TranscriptionScreenState();
}

class _TranscriptionScreenState extends State<TranscriptionScreen> {
  SttFlutter? _stt;
  bool _initialized = false;
  bool _transcribing = false;
  String _transcription = '';
  double _downloadProgress = 0;
  bool _downloading = false;
  String? _status;

  final _recorder = AudioRecorder();

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    setState(() => _status = 'Checking model...');

    final dir = await ModelDownloader.defaultStoragePath(widget.model);
    final downloaded = await ModelDownloader.isDownloaded(widget.model);

    if (!downloaded) {
      setState(() => _status = 'Downloading model...');
      await ModelDownloader.download(
        widget.model,
        onProgress: (received, total) {
          setState(() => _downloadProgress = total > 0 ? received / total : 0);
        },
      );
    }

    setState(() => _status = 'Loading model...');

    _stt = SttFlutter();
    await _stt!.initialize(model: widget.model, modelDir: dir);

    setState(() {
      _initialized = true;
      _status = null;
    });
  }

  Future<void> _recordAndTranscribe() async {
    if (_transcribing || !_initialized) return;

    setState(() => _transcribing = true);

    try {
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/stt_recording.wav';

      await _recorder.start(
        const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000),
        path: path,
      );

      await Future.delayed(const Duration(seconds: 3));
      await _recorder.stop();

      final result = await _stt!.transcribeFile(path);
      setState(() => _transcription = result.text);
    } catch (e) {
      setState(() => _transcription = 'Error: $e');
    } finally {
      setState(() => _transcribing = false);
    }
  }

  @override
  void dispose() {
    _stt?.dispose();
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.model.name)),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_status != null) ...[
              if (_downloading)
                LinearProgressIndicator(value: _downloadProgress),
              const SizedBox(height: 8),
              Text(_status!),
            ],
            const Spacer(),
            if (_transcription.isNotEmpty)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(_transcription, style: const TextStyle(fontSize: 18)),
                ),
              ),
            const Spacer(),
            FilledButton.icon(
              onPressed: _initialized && !_transcribing ? _recordAndTranscribe : null,
              icon: Icon(_transcribing ? Icons.hourglass_top : Icons.mic),
              label: Text(_transcribing ? 'Transcribing...' : 'Record & Transcribe'),
            ),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }
}
