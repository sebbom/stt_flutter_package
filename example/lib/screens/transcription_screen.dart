import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show defaultTargetPlatform, kIsWeb, TargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:stt_flutter/stt_flutter.dart';
import '../utils/audio_diagnostics.dart';

enum LangMode { auto, modelDefault, force }

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
  bool _busyFile = false;
  String _liveText = '';
  SttResult? _lastResult;
  final List<_ResultEntry> _history = [];

  LangMode _langMode = LangMode.auto;
  String _forceLang = '';
  bool _useLanguageDetector = false;
  String? _detectorStatus;

  NormalizeMode _normalize = NormalizeMode.none;
  bool _highPass = false;
  double _preprocessGain = 1.0;
  bool _noiseSuppression = false;

  DenoiserType _denoiserType = DenoiserType.none;
  String _denoiserModelDir = '';
  String _hotwordsText = '';

  final List<int> _speechBuffer = [];
  StreamSubscription<Float32List>? _captureSub;

  static const _vadUrl =
      'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx';
  static const _vadModel = 'silero_vad.onnx';

  static const Map<String, String> _langNames = {
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'es': 'Spanish',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'ru': 'Russian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'ko': 'Korean',
    'hi': 'Hindi',
    'cs': 'Czech',
    'da': 'Danish',
    'el': 'Greek',
    'fi': 'Finnish',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'sv': 'Swedish',
    'uk': 'Ukrainian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'et': 'Estonian',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mt': 'Maltese',
    'sk': 'Slovak',
    'sl': 'Slovenian',
  };

  static String _prettyLang(String code) =>
      _langNames[code] ?? code.toUpperCase();

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

  Future<String?> _sliModelPath(String filename) async {
    final dir = await ModelDownloader.defaultStoragePath(
      ModelDescriptor(
        id: 'whisper-tiny',
        name: 'Whisper Tiny',
        type: SttModelType.whisper,
        languages: const ['en'],
        sizeMb: 0,
        files: const [],
      ),
    );
    final path = '$dir/$filename';
    return await File(path).exists() ? path : null;
  }

  Future<void> _init() async {
    try {
      final dir = await ModelDownloader.defaultStoragePath(widget.model);
      final error = await SttEngine.instance
          .loadModel(widget.model, modelDir: dir)
          .timeout(const Duration(seconds: 120));
      if (error != null) {
        if (mounted) {
          setState(() {
            _loading = false;
            _error = error.toString();
          });
        }
        return;
      }
      final vadPath = await _vadModelPath();
      _capture = AudioCaptureService();
      _vad = SherpaOnnxVadEngine(modelPath: vadPath);
      if (mounted) setState(() => _loading = false);
    } on TimeoutException {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = 'Model load timed out';
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
    _captureSub?.cancel();
    _capture?.dispose();
    if (_vad is SherpaOnnxVadEngine) {
      (_vad as SherpaOnnxVadEngine).free();
    }
    SttEngine.instance.destroy();
    super.dispose();
  }

  String? _effectiveLanguage() {
    switch (_langMode) {
      case LangMode.auto:
        return null;
      case LangMode.modelDefault:
        final dl = SttEngine.instance.currentDefaultLanguage;
        return (dl != null && dl.isNotEmpty) ? dl : null;
      case LangMode.force:
        return _forceLang.isEmpty ? null : _forceLang;
    }
  }

  String _modeLabel() {
    switch (_langMode) {
      case LangMode.auto:
        return 'Auto-detect (engine decides)';
      case LangMode.modelDefault:
        final dl = SttEngine.instance.currentDefaultLanguage;
        return dl == null
            ? 'Default (unset — falls back to auto)'
            : 'Default from model: ${_prettyLang(dl)}';
      case LangMode.force:
        return _forceLang.isEmpty
            ? 'Force (no language picked)'
            : 'Force: ${_prettyLang(_forceLang)}';
    }
  }

  // ---------- Recording ----------

  bool _stopping = false;

  void _toggleRecording() {
    if (_transcribing) {
      _stopRecording();
    } else {
      _startRecording();
    }
  }

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
        await _runTranscribeBytes(buffer, source: 'mic').timeout(
              const Duration(seconds: 30),
            );
      } on TimeoutException {
        SttLogger.d('stopRecording transcribe timed out');
      }
    }
    _speechBuffer.clear();
    if (mounted) {
      setState(() {
        _transcribing = false;
        if (_liveText == 'Stopping…') {
          _liveText = '';
        }
      });
    }
  }

  Future<void> _startRecording() async {
    if (!SttEngine.instance.isReady) return;
    final granted = await _ensureMicPermission();
    if (!granted) {
      if (mounted) setState(() => _error = 'Microphone permission denied');
      return;
    }
    _vad!.reset();
    _speechBuffer.clear();
    _stopping = false;

    try {
      await _capture!.startRecording();
    } catch (e) {
      String msg = 'Failed to start recording: $e';
      if (Platform.isLinux) {
        final diag = await LinuxAudioDiagnostics.diagnose(rawError: e);
        msg = diag.preciseMessage;
      }
      if (mounted) {
        setState(() {
          _transcribing = false;
          _error = msg;
        });
      }
      return;
    }
    _captureSub = _capture!.audioStream.listen(_onAudioChunk);
    if (mounted) setState(() => _transcribing = true);
  }

  Future<bool> _ensureMicPermission() async {
    if (!_permissionHandlerSupported) {
      return _capture!.hasPermission();
    }
    try {
      final status = await Permission.microphone.status;
      if (status.isGranted) return true;
      if (status.isPermanentlyDenied) {
        if (mounted) {
          await showDialog<void>(
            context: context,
            builder: (_) => AlertDialog(
              title: const Text('Microphone permission required'),
              content: const Text(
                  'STT recording needs microphone access. Please enable it in app settings.'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: () {
                    Navigator.pop(context);
                    openAppSettings();
                  },
                  child: const Text('Open settings'),
                ),
              ],
            ),
          );
        }
        return false;
      }
      final result = await Permission.microphone.request();
      return result.isGranted;
    } catch (_) {
      return _capture!.hasPermission();
    }
  }

  static bool get _permissionHandlerSupported {
    if (kIsWeb) return false;
    return defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;
  }

  void _onAudioChunk(Float32List chunk) {
    if (_stopping) return;
    final int16 = Int16List(chunk.length);
    for (var i = 0; i < chunk.length; i++) {
      int16[i] = (chunk[i] * 32768.0).clamp(-32768, 32767).toInt();
    }
    _speechBuffer.addAll(int16);
    if (_vad!.processChunk(int16)) {
      if (mounted) setState(() => _liveText = 'Recording…');
    }
  }

  Future<void> _runTranscribeBytes(List<int> chunk,
      {required String source}) async {
    final int16Samples = Int16List.fromList(chunk);
    final float32Samples = Float32List(int16Samples.length);
    for (int i = 0; i < int16Samples.length; i++) {
      float32Samples[i] = int16Samples[i] / 32768.0;
    }
    SttLogger.d(
        'transcribe($source): ${float32Samples.length} samples, mode=${_langMode.name}');
    final result = await SttEngine.instance.transcribeBuffer(
      float32Samples,
      16000,
      language: _effectiveLanguage(),
    );
    if (mounted) {
      setState(() {
        _liveText = result.text;
        _lastResult = result;
        if (result.text.isNotEmpty) {
          _history.insert(
            0,
            _ResultEntry(
              text: result.text,
              lang: result.lang,
              confidence: result.confidence,
              durationMs: result.durationMs,
              inferenceTimeMs: result.inferenceTimeMs,
              source: source,
              langMode: _langMode,
            ),
          );
        }
      });
    }
  }

  // ---------- File inputs ----------

  Future<void> _testWithSample() async {
    if (_busyFile || _transcribing) return;
    setState(() => _busyFile = true);
    try {
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/sample_test.wav';
      final data = await rootBundle.load('assets/hello_en.wav');
      await File(path).writeAsBytes(data.buffer.asUint8List());
      await _transcribeFile(path, source: 'sample');
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
        });
      }
    } finally {
      if (mounted) setState(() => _busyFile = false);
    }
  }

  Future<void> _pickAndTranscribeFile() async {
    if (_busyFile || _transcribing) return;
    setState(() => _busyFile = true);
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'opus'],
        allowMultiple: false,
        withData: true,
      );
      if (result == null || result.files.isEmpty) return;
      final picked = result.files.single;
      final path = picked.path;
      if (path == null) {
        if (mounted) {
          setState(() => _error = 'Picked file has no local path');
        }
        return;
      }
      final bytes = await File(path).readAsBytes();
      await _transcribeBytes(
        Uint8List.fromList(bytes),
        source: 'file:${picked.name}',
        ext: _extOf(picked.name),
      );
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busyFile = false);
    }
  }

  String _extOf(String name) {
    final i = name.lastIndexOf('.');
    return i < 0 ? '' : name.substring(i + 1).toLowerCase();
  }

  PreprocessConfig get _preprocess => PreprocessConfig(
        gain: _preprocessGain,
        normalize: _normalize,
        highPass: _highPass,
        noiseSuppression: _noiseSuppression,
        denoiserModelDir:
            _denoiserModelDir.isEmpty ? null : _denoiserModelDir,
        denoiserType: _denoiserType,
      );

  bool get _isZipformer => widget.model.type == SttModelType.sherpa;
  bool get _isSenseVoice => widget.model.type == SttModelType.sensevoice;
  bool get _supportsHotwords => _isZipformer;

  Future<void> _applyHotwords() async {
    if (!_isZipformer) return;
    if (mounted) setState(() => _loading = true);
    final err = await SttEngine.instance.setHotwords(_hotwordsText);
    if (mounted) {
      setState(() {
        _loading = false;
        if (err != null) _error = err.toString();
      });
    }
  }

  Future<void> _transcribeFile(String path,
      {required String source}) async {
    final result = await SttEngine.instance.transcribeFile(
      path,
      language: _effectiveLanguage(),
      preprocess: _preprocess,
    );
    if (mounted) {
      setState(() {
        _liveText = result.text;
        _lastResult = result;
        if (result.text.isNotEmpty) {
          _history.insert(
            0,
            _ResultEntry(
              text: result.text,
              lang: result.lang,
              confidence: result.confidence,
              durationMs: result.durationMs,
              inferenceTimeMs: result.inferenceTimeMs,
              source: source,
              langMode: _langMode,
            ),
          );
        }
      });
    }
  }

  Future<void> _transcribeBytes(Uint8List bytes,
      {required String source, required String ext}) async {
    final dir = await getTemporaryDirectory();
    final path =
        '${dir.path}/picked_${DateTime.now().millisecondsSinceEpoch}.$ext';
    await File(path).writeAsBytes(bytes);
    final result = await SttEngine.instance.transcribeFile(
      path,
      language: _effectiveLanguage(),
      preprocess: _preprocess,
    );
    if (mounted) {
      setState(() {
        _liveText = result.text;
        _lastResult = result;
        if (result.text.isNotEmpty) {
          _history.insert(
            0,
            _ResultEntry(
              text: result.text,
              lang: result.lang,
              confidence: result.confidence,
              durationMs: result.durationMs,
              inferenceTimeMs: result.inferenceTimeMs,
              source: source,
              langMode: _langMode,
            ),
          );
        }
      });
    }
  }

  // ---------- Language detector ----------

  Future<void> _toggleLanguageDetector(bool enabled) async {
    if (!enabled) {
      setState(() {
        _useLanguageDetector = false;
        _detectorStatus = 'Disabled';
      });
      return;
    }
    final encoder = await _sliModelPath('tiny-encoder.onnx');
    final decoder = await _sliModelPath('tiny-decoder.onnx');
    if (encoder == null || decoder == null) {
      if (mounted) {
        setState(() {
          _useLanguageDetector = false;
          _detectorStatus =
              'Whisper-tiny not downloaded — pick a Whisper model first';
        });
      }
      return;
    }
    try {
      SttEngine.instance.configureLanguageDetector(
        encoderPath: encoder,
        decoderPath: decoder,
      );
      if (mounted) {
        setState(() {
          _useLanguageDetector = true;
          _detectorStatus = 'Active (Whisper-tiny SLI)';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _useLanguageDetector = false;
          _detectorStatus = 'Failed: $e';
        });
      }
    }
  }

  // ---------- Default-language reload ----------

  Future<void> _setDefaultLanguage() async {
    final selected = await showDialog<String?>(
      context: context,
      builder: (_) => _LangPickerDialog(
        languages: widget.model.languages,
        current: SttEngine.instance.currentDefaultLanguage,
        title: 'Set default language for this model',
      ),
    );
    if (selected == null || !mounted) return;
    if (mounted) setState(() => _loading = true);
    try {
      final dir = await ModelDownloader.defaultStoragePath(widget.model);
      final error = await SttEngine.instance
          .loadModel(
            widget.model,
            modelDir: dir,
            defaultLanguage: selected.isEmpty ? null : selected,
          )
          .timeout(const Duration(seconds: 60));
      if (mounted) {
        setState(() {
          _loading = false;
          if (error != null) _error = error.toString();
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

  // ---------- UI ----------

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.model.name),
        actions: [
          if (SttEngine.instance.isReady)
            IconButton(
              icon: const Icon(Icons.tune),
              tooltip: 'Set default language',
              onPressed: _setDefaultLanguage,
            ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _errorView(theme)
              : _mainView(theme),
    );
  }

  Widget _errorView(ThemeData theme) {
    return Center(
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
              onPressed: () {
                setState(() {
                  _error = null;
                  _loading = true;
                });
                _init();
              },
              child: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _mainView(ThemeData theme) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 24),
      children: [
        _modelInfoCard(theme),
        const SizedBox(height: 12),
        _languageModeCard(theme),
        const SizedBox(height: 12),
        if (_supportsHotwords) _hotwordsCard(theme),
        if (_supportsHotwords) const SizedBox(height: 12),
        _preprocessCard(theme),
        const SizedBox(height: 12),
        _actionsCard(theme),
        const SizedBox(height: 12),
        _resultCard(theme),
        const SizedBox(height: 12),
        if (_history.isNotEmpty) _historyCard(theme),
      ],
    );
  }

  Widget _modelInfoCard(ThemeData theme) {
    final m = widget.model;
    final mono = m.languages.length == 1;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(_engineIcon(m.type),
                    color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text(_engineTypeLabel(m.type),
                    style: theme.textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: [
                _chip(theme, mono ? 'Monolingual' : 'Multilingual',
                    mono ? Icons.lock : Icons.public),
                _chip(
                  theme,
                  mono
                      ? 'Force language = warning'
                      : 'Supports explicit language',
                  mono ? Icons.warning_amber : Icons.check,
                ),
                if (mono)
                  _chip(theme, 'Engine ignores "force" code', Icons.info_outline),
              ],
            ),
            if (!mono) ...[
              const SizedBox(height: 8),
              Text('Languages: ${m.languages.join(", ")}',
                  style: theme.textTheme.bodySmall),
            ],
          ],
        ),
      ),
    );
  }

  IconData _engineIcon(SttModelType t) {
    switch (t) {
      case SttModelType.whisper:
        return Icons.mic;
      case SttModelType.sherpa:
        return Icons.hearing;
      case SttModelType.nemo:
        return Icons.language;
      case SttModelType.canary:
        return Icons.record_voice_over;
      case SttModelType.sensevoice:
        return Icons.psychology;
      case SttModelType.omnilingual:
        return Icons.public;
      case SttModelType.qwen3asr:
        return Icons.auto_awesome;
    }
  }

  String _engineTypeLabel(SttModelType t) {
    switch (t) {
      case SttModelType.whisper:
        return 'Whisper (long-form chunked)';
      case SttModelType.sherpa:
        return 'Sherpa Zipformer';
      case SttModelType.nemo:
        return 'NeMo Parakeet';
      case SttModelType.canary:
        return 'Canary';
      case SttModelType.sensevoice:
        return 'SenseVoice (emotion + events)';
      case SttModelType.omnilingual:
        return 'Omnilingual CTC (1600 languages)';
      case SttModelType.qwen3asr:
        return 'Qwen3-ASR (autoregressive LLM)';
    }
  }

  Widget _chip(ThemeData theme, String label, IconData icon) {
    return Chip(
      avatar: Icon(icon, size: 16),
      label: Text(label, style: theme.textTheme.labelSmall),
      visualDensity: VisualDensity.compact,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    );
  }

  Widget _languageModeCard(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Language mode', style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            SegmentedButton<LangMode>(
              segments: const [
                ButtonSegment(
                    value: LangMode.auto,
                    label: Text('Auto'),
                    icon: Icon(Icons.auto_awesome)),
                ButtonSegment(
                    value: LangMode.modelDefault,
                    label: Text('Default'),
                    icon: Icon(Icons.settings)),
                ButtonSegment(
                    value: LangMode.force,
                    label: Text('Force'),
                    icon: Icon(Icons.bolt)),
              ],
              selected: {_langMode},
              onSelectionChanged: (s) =>
                  setState(() => _langMode = s.first),
            ),
            const SizedBox(height: 8),
            Text(_modeLabel(),
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                )),
            const SizedBox(height: 8),
            if (_langMode == LangMode.force) _forceLangPicker(theme),
            if (_langMode == LangMode.modelDefault) _defaultLangRow(theme),
            const SizedBox(height: 4),
            SwitchListTile.adaptive(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Language detector fallback (SLI)'),
              subtitle: Text(
                _detectorStatus ??
                    'Fills result.lang when the engine returns nothing.',
                style: theme.textTheme.bodySmall,
              ),
              value: _useLanguageDetector,
              onChanged: _toggleLanguageDetector,
            ),
          ],
        ),
      ),
    );
  }

  Widget _forceLangPicker(ThemeData theme) {
    return Row(
      children: [
        const Text('Pick: '),
        Expanded(
          child: DropdownButton<String>(
            isExpanded: true,
            value: _forceLang.isEmpty
                ? (widget.model.languages.isNotEmpty
                    ? widget.model.languages.first
                    : null)
                : _forceLang,
            items: widget.model.languages
                .map((l) => DropdownMenuItem(
                      value: l,
                      child: Text(_prettyLang(l)),
                    ))
                .toList(),
            onChanged: (v) {
              if (v != null) setState(() => _forceLang = v);
            },
          ),
        ),
      ],
    );
  }

  Widget _preprocessCard(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.graphic_eq, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text('Preprocessing (files only)',
                    style: theme.textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'Applies to Pick file / Sample. Use for low-volume recordings '
              '— STT engines decode blanks when peaks are very small.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            SegmentedButton<NormalizeMode>(
              segments: const [
                ButtonSegment(
                    value: NormalizeMode.none,
                    label: Text('Off'),
                    icon: Icon(Icons.block)),
                ButtonSegment(
                    value: NormalizeMode.peak,
                    label: Text('Peak'),
                    icon: Icon(Icons.trending_up)),
                ButtonSegment(
                    value: NormalizeMode.rms,
                    label: Text('RMS'),
                    icon: Icon(Icons.equalizer)),
              ],
              selected: {_normalize},
              onSelectionChanged: (s) =>
                  setState(() => _normalize = s.first),
            ),
            SwitchListTile.adaptive(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('High-pass filter (80 Hz)'),
              subtitle: Text(
                'Removes DC offset and low-frequency rumble.',
                style: theme.textTheme.bodySmall,
              ),
              value: _highPass,
              onChanged: (v) => setState(() => _highPass = v),
            ),
            Row(
              children: [
                const Text('Gain'),
                Expanded(
                  child: Slider(
                    value: _preprocessGain,
                    min: 0.5,
                    max: 5.0,
                    divisions: 18,
                    label: '${_preprocessGain.toStringAsFixed(2)}×',
                    onChanged: (v) => setState(() => _preprocessGain = v),
                  ),
                ),
                SizedBox(
                  width: 56,
                  child: Text(
                    '${_preprocessGain.toStringAsFixed(2)}×',
                    textAlign: TextAlign.end,
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              ],
            ),
            if (_preprocessGain > 1.0 && _normalize == NormalizeMode.none)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  '⚠ Gain > 1× without normalization can clip and hurt accuracy.',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                ),
              ),
            const Divider(height: 24),
            Row(
              children: [
                Icon(Icons.noise_aware, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text('Denoiser (sherpa-onnx)',
                    style: theme.textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'Neural speech enhancement (GTCRN or DPDFNet). Requires the '
              'sherpa-onnx denoiser model file in a local directory.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            SegmentedButton<DenoiserType>(
              segments: const [
                ButtonSegment(
                    value: DenoiserType.none,
                    label: Text('Off'),
                    icon: Icon(Icons.block)),
                ButtonSegment(
                    value: DenoiserType.gtcrn,
                    label: Text('GTCRN'),
                    icon: Icon(Icons.graphic_eq)),
                ButtonSegment(
                    value: DenoiserType.dpdfnet,
                    label: Text('DPDFNet'),
                    icon: Icon(Icons.equalizer)),
              ],
              selected: {_denoiserType},
              onSelectionChanged: (s) =>
                  setState(() => _denoiserType = s.first),
            ),
            const SizedBox(height: 8),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Denoiser model directory',
                helperText: 'Path containing model.onnx (GTCRN) or '
                    'model.onnx + model_post.onnx (DPDFNet)',
                isDense: true,
                border: OutlineInputBorder(),
              ),
              controller: TextEditingController(text: _denoiserModelDir)
                ..selection = TextSelection.collapsed(
                    offset: _denoiserModelDir.length),
              onChanged: (v) => _denoiserModelDir = v,
            ),
            const Divider(height: 24),
            SwitchListTile.adaptive(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Live mic noise suppression (flag)'),
              subtitle: Text(
                'UI hook — wire a platform plugin (e.g. noise_suppression) '
                'to actually filter the mic stream.',
                style: theme.textTheme.bodySmall,
              ),
              value: _noiseSuppression,
              onChanged: (v) => setState(() => _noiseSuppression = v),
            ),
          ],
        ),
      ),
    );
  }

  Widget _hotwordsCard(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.bolt, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text('Hotwords (Zipformer)', style: theme.textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'One entry per line, formatted as "word score". Boosts the '
              'likelihood of these words in the output.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              maxLines: 4,
              minLines: 2,
              decoration: const InputDecoration(
                labelText: 'hotwords.txt contents',
                hintText: 'kubernetes 2.0\nkubeflow 1.8',
                isDense: true,
                border: OutlineInputBorder(),
              ),
              controller: TextEditingController(text: _hotwordsText)
                ..selection =
                    TextSelection.collapsed(offset: _hotwordsText.length),
              onChanged: (v) => _hotwordsText = v,
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.tonalIcon(
                onPressed: _loading ? null : _applyHotwords,
                icon: const Icon(Icons.save, size: 18),
                label: const Text('Apply hotwords'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _defaultLangRow(ThemeData theme) {
    final dl = SttEngine.instance.currentDefaultLanguage;
    return Row(
      children: [
        Expanded(
          child: Text(
            dl == null
                ? 'No default set — falls back to auto'
                : 'Default: ${_prettyLang(dl)}',
            style: theme.textTheme.bodySmall,
          ),
        ),
        TextButton.icon(
          icon: const Icon(Icons.edit, size: 16),
          label: const Text('Set'),
          onPressed: _setDefaultLanguage,
        ),
      ],
    );
  }

  Widget _actionsCard(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Inputs', style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed:
                        (_transcribing || _busyFile) ? null : _testWithSample,
                    icon: const Icon(Icons.audiotrack, size: 18),
                    label: const Text('Sample'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: (_transcribing || _busyFile)
                        ? null
                        : _pickAndTranscribeFile,
                    icon: const Icon(Icons.folder_open, size: 18),
                    label: const Text('Pick file'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: FilledButton.icon(
                    style: _transcribing
                        ? FilledButton.styleFrom(
                            backgroundColor: Colors.red,
                            foregroundColor: Colors.white,
                          )
                        : null,
                    onPressed: _busyFile ? null : _toggleRecording,
                    icon: Icon(
                      _transcribing ? Icons.stop : Icons.mic,
                      size: 18,
                    ),
                    label: Text(_transcribing ? 'Stop' : 'Mic'),
                  ),
                ),
              ],
            ),
            if (_transcribing) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  const SizedBox(
                    width: 10,
                    height: 10,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: Colors.red,
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text('Recording — speak now',
                      style: theme.textTheme.labelSmall
                          ?.copyWith(color: Colors.red)),
                ],
              ),
            ],
            if (_busyFile) ...[
              const SizedBox(height: 8),
              const LinearProgressIndicator(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _resultCard(ThemeData theme) {
    if (_lastResult == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Center(
            child: Text(
              'No transcription yet',
              style: theme.textTheme.bodyMedium
                  ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            ),
          ),
        ),
      );
    }
    final r = _lastResult!;
    return Card(
      color: theme.colorScheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SelectableText(
              r.text.isEmpty ? '(empty)' : r.text,
              style: theme.textTheme.titleMedium
                  ?.copyWith(color: theme.colorScheme.onPrimaryContainer),
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: [
                if (r.lang != null && r.lang!.isNotEmpty)
                  _metricChip(theme, 'lang', r.lang!),
                if (r.confidence != null)
                  _metricChip(
                    theme,
                    'conf',
                    '${(r.confidence! * 100).toStringAsFixed(0)}%',
                  ),
                if (r.durationMs != null)
                  _metricChip(
                    theme,
                    'audio',
                    '${(r.durationMs! / 1000).toStringAsFixed(1)}s',
                  ),
                _metricChip(
                  theme,
                  'infer',
                  '${r.inferenceTimeMs.toStringAsFixed(0)}ms',
                ),
                _metricChip(theme, 'mode', _langMode.name),
                if (_isSenseVoice) ...[
                  if (r.emotion != null && r.emotion!.isNotEmpty)
                    _metricChip(theme, 'emotion', r.emotion!),
                  if (r.events.isNotEmpty)
                    _metricChip(theme, 'events', r.events.join(', ')),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _metricChip(ThemeData theme, String key, String value) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: theme.colorScheme.primary.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        '$key: $value',
        style: theme.textTheme.labelSmall
            ?.copyWith(color: theme.colorScheme.onPrimaryContainer),
      ),
    );
  }

  Widget _historyCard(ThemeData theme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text('History (${_history.length})',
                    style: theme.textTheme.titleSmall),
                const Spacer(),
                TextButton(
                  onPressed: () => setState(() => _history.clear()),
                  child: const Text('Clear'),
                ),
              ],
            ),
            const SizedBox(height: 4),
            ...List.generate(_history.length, (i) {
              final h = _history[i];
              return ListTile(
                contentPadding: EdgeInsets.zero,
                dense: true,
                leading: Icon(_sourceIcon(h.source), size: 20),
                title: Text(h.text,
                    maxLines: 2, overflow: TextOverflow.ellipsis),
                subtitle: Text(_historySubtitle(h)),
              );
            }),
          ],
        ),
      ),
    );
  }

  IconData _sourceIcon(String source) {
    if (source.startsWith('mic')) return Icons.mic;
    if (source.startsWith('sample')) return Icons.audiotrack;
    if (source.startsWith('file')) return Icons.folder_open;
    return Icons.text_fields;
  }

  String _historySubtitle(_ResultEntry h) {
    final parts = <String>[
      h.source,
      'mode=${h.langMode.name}',
      'infer=${h.inferenceTimeMs.toStringAsFixed(0)}ms',
    ];
    if (h.lang != null && h.lang!.isNotEmpty) parts.add('lang=${h.lang}');
    if (h.confidence != null) {
      parts.add('conf=${(h.confidence! * 100).toStringAsFixed(0)}%');
    }
    if (h.durationMs != null) {
      parts.add('audio=${(h.durationMs! / 1000).toStringAsFixed(1)}s');
    }
    return parts.join(' • ');
  }
}

class _ResultEntry {
  final String text;
  final String? lang;
  final double? confidence;
  final double? durationMs;
  final double inferenceTimeMs;
  final String source;
  final LangMode langMode;
  _ResultEntry({
    required this.text,
    required this.lang,
    required this.confidence,
    required this.durationMs,
    required this.inferenceTimeMs,
    required this.source,
    required this.langMode,
  });
}

class _LangPickerDialog extends StatefulWidget {
  final List<String> languages;
  final String? current;
  final String title;
  const _LangPickerDialog({
    required this.languages,
    required this.current,
    required this.title,
  });

  @override
  State<_LangPickerDialog> createState() => _LangPickerDialogState();
}

class _LangPickerDialogState extends State<_LangPickerDialog> {
  String? _selected;

  @override
  void initState() {
    super.initState();
    _selected = widget.current;
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(widget.title),
      content: SizedBox(
        width: 320,
        child: RadioGroup<String?>(
          groupValue: _selected,
          onChanged: (v) => setState(() => _selected = v),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              RadioListTile<String?>(
                title: const Text('(none — auto-detect)'),
                value: '',
                contentPadding: EdgeInsets.zero,
                dense: true,
              ),
              const Divider(),
              ...widget.languages.map(
                (l) => RadioListTile<String?>(
                  title: Text(
                      '${l.toUpperCase()} — ${_TranscriptionScreenState._langNames[l] ?? l}'),
                  value: l,
                  contentPadding: EdgeInsets.zero,
                  dense: true,
                ),
              ),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: () => Navigator.pop(context, _selected ?? ''),
          child: const Text('Apply'),
        ),
      ],
    );
  }
}
