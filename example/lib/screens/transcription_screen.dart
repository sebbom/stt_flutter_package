import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:stt_flutter/stt_flutter.dart';

class TranscriptionScreen extends StatefulWidget {
  final ModelDescriptor model;
  const TranscriptionScreen({super.key, required this.model});

  @override
  State<TranscriptionScreen> createState() => _TranscriptionScreenState();
}

class _TranscriptionScreenState extends State<TranscriptionScreen> {
  AudioCaptureService? _capture;
  VadEngine? _vad;

  bool _loading = true;
  String? _error;
  bool _transcribing = false;
  String _liveText = '';
  String? _detectedLang;
  final List<String> _history = [];
  String _language = 'en';

  final List<int> _speechBuffer = [];

  static const _vadUrl = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx';
  static const _vadModel = 'silero_vad.onnx';

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<String> _vadModelPath() async {
    final dir = await getApplicationDocumentsDirectory();
    final path = '${dir.path}/stt_models/silero_vad/$_vadModel';
    final file = File(path);
    if (!await file.exists()) {
      await file.create(recursive: true);
      await ModelDownloader.downloadFile(url: _vadUrl, destPath: path);
    }
    return path;
  }

  Future<void> _init() async {
    try {
      final dir = await ModelDownloader.defaultStoragePath(widget.model);
      final result = await SttEngine.instance
          .loadModel(widget.model, modelDir: dir)
          .timeout(const Duration(seconds: 120));
      if (result != 'Success') {
        if (mounted) setState(() { _loading = false; _error = result; });
        return;
      }
      final vadPath = await _vadModelPath();
      _capture = AudioCaptureService();
      _vad = SherpaOnnxVadEngine(modelPath: vadPath);
      if (mounted) setState(() => _loading = false);
    } on TimeoutException {
      if (mounted) setState(() { _loading = false; _error = 'Model load timed out'; });
    } catch (e) {
      if (mounted) setState(() { _loading = false; _error = e.toString(); });
    }
  }

  @override
  void dispose() {
    _capture?.dispose();
    if (_vad is SherpaOnnxVadEngine) {
      (_vad as SherpaOnnxVadEngine).free();
    }
    SttEngine.instance.destroy();
    super.dispose();
  }

  // ---------- Recording ----------

  void _toggleRecording() {
    if (_transcribing) {
      _stopRecording();
    } else {
      _startRecording();
    }
  }

  bool _stopping = false;

  Future<void> _stopRecording() async {
    _stopping = true;
    if (mounted) setState(() => _liveText = 'Stopping…');

    try {
      await _capture!.stopRecording().timeout(const Duration(seconds: 3));
    } catch (_) {}

    if (_speechBuffer.length >= 16000 * 0.3) {
      final buffer = List<int>.from(_speechBuffer);
      _speechBuffer.clear();
      try {
        await _runTranscribe(buffer).timeout(const Duration(seconds: 30));
      } on TimeoutException {
        SttLogger.d('stopRecording transcribe timed out');
      }
    }
    _speechBuffer.clear();
    if (mounted) setState(() {
      _transcribing = false;
      if (_liveText == 'Stopping…') {
        _liveText = '';
      }
    });
  }

  Future<void> _startRecording() async {
    if (!SttEngine.instance.isReady) return;
    if (!await _capture!.hasPermission()) {
      if (mounted) setState(() => _error = 'Microphone permission denied');
      return;
    }
    _vad!.reset();
    _speechBuffer.clear();
    _stopping = false;

    await _capture!.startRecording((chunk) => _onAudioChunk(chunk));
    if (mounted) setState(() => _transcribing = true);
  }

  void _onAudioChunk(List<int> chunk) {
    if (_stopping) return;
    _speechBuffer.addAll(chunk);
    if (_vad!.processChunk(chunk)) {
      if (mounted) setState(() => _liveText = 'Recording…');
    }
  }

  Future<void> _runTranscribe(List<int> chunk) async {
    final int16Samples = Int16List.fromList(chunk);
    final float32Samples = Float32List(int16Samples.length);
    for (int i = 0; i < int16Samples.length; i++) {
      float32Samples[i] = int16Samples[i] / 32768.0;
    }
    SttLogger.d('_runTranscribe: ${float32Samples.length} samples');
    try {
      final result = await SttEngine.instance.transcribeBuffer(
        float32Samples,
        16000,
        language: _language,
      );
      SttLogger.d('_runTranscribe result: "${result.text}" lang=${result.lang}');
      if (mounted && result.text.isNotEmpty) {
        setState(() {
          _liveText = result.text;
          _detectedLang = result.lang;
          if (!_history.contains(result.text)) _history.insert(0, result.text);
        });
      }
    } catch (e) {
      SttLogger.d('_runTranscribe error: $e');
      if (mounted) setState(() => _error = 'Transcribe error: $e');
    }
  }

  // ---------- Sample audio ----------

  Future<void> _testWithSample() async {
    if (_transcribing) return;
    setState(() => _transcribing = true);
    try {
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/sample_test.wav';
      final data = await rootBundle.load('assets/hello_en.wav');
      await File(path).writeAsBytes(data.buffer.asUint8List());
      final result = await SttEngine.instance.transcribeFile(path, language: _language);
      if (mounted) {
        setState(() {
          _liveText = result.text;
          _detectedLang = result.lang;
          _history.insert(0, '${result.text}  (${result.inferenceTimeMs.toStringAsFixed(1)}ms)');
          _transcribing = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() { _error = e.toString(); _transcribing = false; });
    }
  }

  // ---------- UI ----------

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: Text(widget.model.name)),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.error_outline, size: 48, color: theme.colorScheme.error),
                        const SizedBox(height: 12),
                        Text(_error!, textAlign: TextAlign.center),
                        const SizedBox(height: 16),
                        FilledButton(
                          onPressed: () { setState(() { _error = null; _loading = true; }); _init(); },
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  ),
                )
              : Column(
                  children: [
                    const Spacer(),
                    if (_liveText.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Card(
                          color: theme.colorScheme.primaryContainer,
                          child: Padding(
                            padding: const EdgeInsets.all(20),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                SelectableText(
                                  _liveText,
                                  style: theme.textTheme.titleMedium!.copyWith(
                                    color: theme.colorScheme.onPrimaryContainer,
                                  ),
                                ),
                                if (_detectedLang != null) ...[
                                  const SizedBox(height: 6),
                                  Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                    decoration: BoxDecoration(
                                      color: theme.colorScheme.primary.withOpacity(0.15),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(
                                      _detectedLang!,
                                      style: theme.textTheme.labelSmall!.copyWith(
                                        color: theme.colorScheme.onPrimaryContainer,
                                      ),
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          ),
                        ),
                      ),
                    const Spacer(),
                    SizedBox(
                      width: 80,
                      height: 80,
                      child: Material(
                        shape: const CircleBorder(),
                        color: _transcribing ? Colors.red : theme.colorScheme.primary,
                        child: InkWell(
                          customBorder: const CircleBorder(),
                          onTap: _toggleRecording,
                          child: Icon(
                            _transcribing ? Icons.stop : Icons.mic,
                            color: Colors.white,
                            size: 36,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _transcribing
                          ? 'Recording (VAD) — tap to stop'
                          : 'Tap to record',
                      style: theme.textTheme.bodySmall,
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text('Language: ', style: theme.textTheme.bodySmall),
                        DropdownButton<String>(
                          value: _language,
                          underline: const SizedBox(),
                          items: const [
                            DropdownMenuItem(value: 'en', child: Text('English')),
                            DropdownMenuItem(value: 'de', child: Text('German')),
                            DropdownMenuItem(value: 'fr', child: Text('French')),
                            DropdownMenuItem(value: 'es', child: Text('Spanish')),
                            DropdownMenuItem(value: 'it', child: Text('Italian')),
                            DropdownMenuItem(value: 'pt', child: Text('Portuguese')),
                          ],
                          onChanged: _transcribing ? null : (v) {
                            if (v != null) setState(() => _language = v);
                          },
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    OutlinedButton.icon(
                      onPressed: _transcribing ? null : _testWithSample,
                      icon: const Icon(Icons.audiotrack, size: 18),
                      label: const Text('Test with sample audio'),
                    ),
                    const SizedBox(height: 24),
                    if (_history.isNotEmpty) ...[
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Row(
                          children: [
                            Text('History', style: theme.textTheme.titleSmall),
                            const Spacer(),
                            TextButton(
                              onPressed: () => setState(() => _history.clear()),
                              child: const Text('Clear'),
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: ListView.builder(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          itemCount: _history.length,
                          itemBuilder: (_, i) => Card(
                            child: ListTile(
                              leading: const Icon(Icons.mic, size: 20),
                              title: Text(_history[i], maxLines: 2, overflow: TextOverflow.ellipsis),
                              subtitle: Text('Attempt ${_history.length - i}'),
                            ),
                          ),
                        ),
                      ),
                    ],
                    const SizedBox(height: 16),
                  ],
                ),
    );
  }
}
