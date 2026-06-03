import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'transcription_screen.dart';

class ModelSelectionScreen extends StatefulWidget {
  const ModelSelectionScreen({super.key});

  @override
  State<ModelSelectionScreen> createState() => _ModelSelectionScreenState();
}

class _ModelSelectionScreenState extends State<ModelSelectionScreen> {
  final Set<String> _downloadsInProgress = {};
  final Map<String, double> _downloadProgress = {};

  @override
  void initState() {
    super.initState();
    // Trigger default model registration
    ModelRegistry.available();
  }

  Future<void> _download(BuildContext context, ModelDescriptor model) async {
    if (_downloadsInProgress.contains(model.id)) return;

    setState(() {
      _downloadsInProgress.add(model.id);
      _downloadProgress[model.id] = 0;
    });

    try {
      final dir = await ModelDownloader.defaultStoragePath(model);
      await Directory(dir).create(recursive: true);

      await ModelDownloader.download(
        model,
        storagePath: dir,
        onProgress: (received, total) {
          final pct = total > 0 ? received / total : 0.0;
          if (mounted) setState(() => _downloadProgress[model.id] = pct);
        },
        onFileProgress: (file, received, total) {
          final pct = total > 0 ? received / total : 0.0;
          if (mounted) setState(() => _downloadProgress[model.id] = pct);
        },
      );

      if (mounted) {
        setState(() {
          _downloadsInProgress.remove(model.id);
          _downloadProgress.remove(model.id);
        });
        if (context.mounted) _openTranscription(context, model);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _downloadsInProgress.remove(model.id);
          _downloadProgress.remove(model.id);
        });
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Download failed: $e'), backgroundColor: Colors.red),
          );
        }
      }
    }
  }

  Future<void> _checkDownloadedAndRefresh() async {
    setState(() {});
  }

  void _openTranscription(BuildContext context, ModelDescriptor model) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => TranscriptionScreen(model: model)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final models = ModelRegistry.available();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Local STT'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _checkDownloadedAndRefresh,
          ),
        ],
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(12),
        itemCount: models.length,
        itemBuilder: (_, i) => _ModelCard(
          model: models[i],
          progress: _downloadProgress[models[i].id],
          downloading: _downloadsInProgress.contains(models[i].id),
          onDownload: () => _download(context, models[i]),
          onOpen: () => _openTranscription(context, models[i]),
        ),
      ),
    );
  }
}

class _ModelCard extends StatefulWidget {
  final ModelDescriptor model;
  final double? progress;
  final bool downloading;
  final VoidCallback onDownload;
  final VoidCallback onOpen;

  const _ModelCard({
    required this.model,
    this.progress,
    required this.downloading,
    required this.onDownload,
    required this.onOpen,
  });

  @override
  State<_ModelCard> createState() => _ModelCardState();
}

class _ModelCardState extends State<_ModelCard> {
  bool _downloaded = false;

  @override
  void initState() {
    super.initState();
    _check();
  }

  @override
  void didUpdateWidget(_ModelCard old) {
    super.didUpdateWidget(old);
    if (old.model.id != widget.model.id || old.downloading != widget.downloading) {
      _check();
    }
  }

  Future<void> _check() async {
    final dir = await ModelDownloader.defaultStoragePath(widget.model);
    final ok = await ModelDownloader.isDownloaded(widget.model, storagePath: dir);
    if (mounted && ok != _downloaded) setState(() => _downloaded = ok);
  }

  IconData _icon() {
    switch (widget.model.type) {
      case SttModelType.whisper:
        return Icons.mic;
      case SttModelType.sherpa:
        return Icons.hearing;
      case SttModelType.nemo:
        return Icons.language;
      case SttModelType.canary:
        return Icons.record_voice_over;
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
    final m = widget.model;
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: _downloaded ? widget.onOpen : null,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  CircleAvatar(child: Icon(_icon())),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(m.name, style: theme.textTheme.titleSmall),
                        const SizedBox(height: 2),
                        Text(
                          '${m.languages.length > 5 ? "${m.languages.length} langs" : m.languages.join(", ")} • ${_sizeStr(m.sizeMb)}',
                          style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              if (widget.downloading || !_downloaded) ...[
                const SizedBox(height: 8),
                if (widget.downloading)
                  LinearProgressIndicator(value: widget.progress)
                else
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      FilledButton.tonalIcon(
                        onPressed: widget.onDownload,
                        icon: const Icon(Icons.download, size: 18),
                        label: const Text('Download'),
                      ),
                    ],
                  ),
              ],
              if (_downloaded)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Row(
                    children: [
                      Icon(Icons.check_circle, size: 14, color: theme.colorScheme.primary),
                      const SizedBox(width: 4),
                      Text('Downloaded', style: theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.primary)),
                      const Spacer(),
                      Icon(Icons.arrow_forward_ios, size: 14, color: theme.colorScheme.onSurfaceVariant),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
