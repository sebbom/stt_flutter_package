import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'package:path_provider/path_provider.dart';

class ModelSelectionScreen extends StatefulWidget {
  const ModelSelectionScreen({super.key});

  @override
  State<ModelSelectionScreen> createState() => _ModelSelectionScreenState();
}

class _ModelSelectionScreenState extends State<ModelSelectionScreen> {
  final Set<String> _downloaded = {};
  late List<ModelDescriptor> _models;

  @override
  void initState() {
    super.initState();
    _models = ModelRegistry.available();
    _checkDownloaded();
  }

  Future<void> _checkDownloaded() async {
    for (final m in _models) {
      if (await ModelDownloader.isDownloaded(m)) {
        if (mounted) setState(() => _downloaded.add(m.id));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Local STT'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _checkDownloaded,
          ),
        ],
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(12),
        itemCount: _models.length,
        itemBuilder: (_, i) => _ModelCard(
          model: _models[i],
          downloaded: _downloaded.contains(_models[i].id),
          onTap: () => _openModel(context, _models[i]),
        ),
      ),
    );
  }

  Future<void> _openModel(BuildContext context, ModelDescriptor model) async {
    final dir = await ModelDownloader.defaultStoragePath(model);
    if (!_downloaded.contains(model.id)) {
      if (!context.mounted) return;
      final ok = await showDialog<bool>(
        context: context,
        builder: (_) => _DownloadDialog(model: model, destDir: dir),
      );
      if (ok != true) return;
      if (context.mounted) {
        setState(() => _downloaded.add(model.id));
      }
    }
    if (!context.mounted) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => _TranscribeScreen(model: model, modelDir: dir),
      ),
    );
  }
}

// --- Model card ---

class _ModelCard extends StatelessWidget {
  final ModelDescriptor model;
  final bool downloaded;
  final VoidCallback onTap;

  const _ModelCard({required this.model, required this.downloaded, required this.onTap});

  IconData _icon() {
    switch (model.type) {
      case SttModelType.whisper:
        return Icons.mic;
      case SttModelType.sherpa:
        return Icons.hearing;
      case SttModelType.voxtral:
        return Icons.smart_toy;
    }
  }

  String _sizeStr(int mb) {
    if (mb >= 1000) return '${(mb / 1000).toStringAsFixed(1)} GB';
    return '$mb MB';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: CircleAvatar(child: Icon(_icon())),
        title: Text(model.name, maxLines: 1, overflow: TextOverflow.ellipsis),
        subtitle: Text(
          '${model.languages.join(', ')} • ${_sizeStr(model.sizeMb)}',
        ),
        trailing: downloaded
            ? Icon(Icons.check_circle, color: theme.colorScheme.primary)
            : const Icon(Icons.download),
        onTap: onTap,
      ),
    );
  }
}

// --- Download dialog ---

class _DownloadDialog extends StatefulWidget {
  final ModelDescriptor model;
  final String destDir;
  const _DownloadDialog({required this.model, required this.destDir});

  @override
  State<_DownloadDialog> createState() => _DownloadDialogState();
}

class _DownloadDialogState extends State<_DownloadDialog> {
  double _progress = 0;
  String _file = '';
  bool _done = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    try {
      await ModelDownloader.download(
        widget.model,
        storagePath: widget.destDir,
        onProgress: (rcv, total) {
          if (mounted) setState(() => _progress = total > 0 ? rcv / total : 0);
        },
        onFileProgress: (f, rcv, total) {
          if (mounted) setState(() => _file = '$f (${rcv ~/ 1024}/${total ~/ 1024} KB)');
        },
      );
      if (mounted) setState(() => _done = true);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: _done || _error != null,
      child: AlertDialog(
        title: Text(_done ? 'Ready!' : 'Downloading…'),
        content: SizedBox(
          width: 300,
          child: _error != null
              ? Text('Error: $_error')
              : Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (!_done) ...[
                      LinearProgressIndicator(value: _progress),
                      const SizedBox(height: 8),
                      Text(_file, style: Theme.of(context).textTheme.bodySmall),
                      Text('${(_progress * 100).toStringAsFixed(0)}%'),
                    ] else
                      const Icon(Icons.check_circle, size: 64, color: Colors.green),
                  ],
                ),
        ),
        actions: [
          if (_done)
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Continue'),
            ),
          if (_error != null)
            FilledButton(
              onPressed: () {
                setState(() {
                  _error = null;
                  _progress = 0;
                });
                _start();
              },
              child: const Text('Retry'),
            ),
          if (_error != null)
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel'),
            ),
        ],
      ),
    );
  }
}

// --- Transcription screen ---

class _TranscribeScreen extends StatefulWidget {
  final ModelDescriptor model;
  final String modelDir;
  const _TranscribeScreen({required this.model, required this.modelDir});

  @override
  State<_TranscribeScreen> createState() => _TranscribeScreenState();
}

class _TranscribeScreenState extends State<_TranscribeScreen> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  SttFlutter? _stt;
  String? _error;
  bool _loading = true;
  String _loadingMessage = 'Loading model...';
  bool _recording = false;
  bool _transcribing = false;
  String _text = '';
  String _language = 'en';
  final List<String> _history = [];

  static const _languages = {
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'ru': 'Russian',
    'it': 'Italian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'ar': 'Arabic',
  };

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    _loadingMessage = 'Loading model files...';
    try {
      final stt = SttFlutter();
      await stt
          .initialize(model: widget.model, modelDir: widget.modelDir)
          .timeout(const Duration(seconds: 120));
      await _recorder.openRecorder();
      if (mounted) {
        setState(() {
          _stt = stt;
          _loading = false;
        });
      }
    } on TimeoutException {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = 'Model loading timed out (120s). The model may be too large for this device.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = e.toString();
        });
      }
    }
  }

  @override
  void dispose() {
    _stt?.dispose();
    _recorder.closeRecorder();
    super.dispose();
  }

  Future<void> _testWithSample() async {
    if (_transcribing) return;
    setState(() {
      _transcribing = true;
      _error = null;
    });
    try {
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/sample_test.wav';
      final data = await rootBundle.load('assets/hello_en.wav');
      await File(path).writeAsBytes(data.buffer.asUint8List());
      final result = await _stt!.transcribeFile(path, language: _language);
      if (mounted) {
        setState(() {
          _text = result.text;
          _history.insert(0, result.text);
          _transcribing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _transcribing = false;
        });
      }
    }
  }

  Future<void> _record() async {
    if (_recording || _transcribing) return;
    setState(() {
      _recording = true;
      _error = null;
    });

    try {
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/stt.wav';

      // Request microphone permission explicitly
      final micStatus = await Permission.microphone.request();
      if (!micStatus.isGranted) {
        throw Exception(
          'Microphone permission denied. '
          'Grant it in Settings > Apps > stt_flutter_example > Permissions.',
        );
      }

      if (!await _recorder.isEncoderSupported(Codec.pcm16WAV)) {
        throw Exception('PCM16WAV codec not supported on this device');
      }
      await _recorder.startRecorder(
        toFile: path,
        codec: Codec.pcm16WAV,
        sampleRate: 16000,
        numChannels: 1,
        audioSource: AudioSource.defaultSource,
      );

      // Record for 5 seconds
      await Future.delayed(const Duration(seconds: 5));
      final savedPath = await _recorder.stopRecorder();

      if (savedPath != null) {
        final file = File(savedPath);
        if (!await file.exists() || await file.length() < 44) {
          throw Exception(
            'Recording produced an empty file. '
            'Check that microphone permission is granted in Settings.',
          );
        }
        setState(() {
          _recording = false;
          _transcribing = true;
        });
        final result = await _stt!.transcribeFile(savedPath, language: _language);
        if (mounted) {
          setState(() {
            _text = result.text;
            _history.insert(0, result.text);
            _transcribing = false;
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _recording = false;
          _transcribing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.model.name),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            _stt?.dispose();
            Navigator.pop(context);
          },
        ),
      ),
      body: _loading
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(_loadingMessage,
                      style: theme.textTheme.bodyMedium),
                  const SizedBox(height: 4),
                  Text('This may take a while for large models',
                      style: theme.textTheme.bodySmall),
                ],
              ),
            )
          : _error != null && _stt == null
              ? Center(child: Text('Failed to load: $_error'))
              : Column(
                  children: [
                    const Spacer(),
                    // transcription result
                    if (_text.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Card(
                          color: theme.colorScheme.primaryContainer,
                          child: Padding(
                            padding: const EdgeInsets.all(20),
                            child: SelectableText(
                              _text,
                              style: theme.textTheme.titleMedium!.copyWith(
                                color: theme.colorScheme.onPrimaryContainer,
                              ),
                            ),
                          ),
                        ),
                      ),
                    if (_error != null)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Card(
                          color: theme.colorScheme.errorContainer,
                          child: Padding(
                            padding: const EdgeInsets.all(12),
                            child: Text(
                              _error!,
                              style: TextStyle(color: theme.colorScheme.onErrorContainer),
                            ),
                          ),
                        ),
                      ),
                    const Spacer(),
                    // record button
                    GestureDetector(
                      onTapDown: _transcribing ? null : (_) => _record(),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 150),
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _recording
                              ? Colors.red
                              : _transcribing
                                  ? Colors.grey
                                  : theme.colorScheme.primary,
                        ),
                        child: Icon(
                          _recording ? Icons.stop : Icons.mic,
                          color: Colors.white,
                          size: 36,
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _recording
                          ? 'Recording (auto-stops after 5s)'
                          : _transcribing
                              ? 'Transcribing…'
                              : 'Tap to record (5s)',
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
                          items: _languages.entries
                              .map((e) => DropdownMenuItem(
                                    value: e.key,
                                    child: Text(e.value, style: const TextStyle(fontSize: 14)),
                                  ))
                              .toList(),
                          onChanged: _transcribing
                              ? null
                              : (v) {
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
                    // history
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
