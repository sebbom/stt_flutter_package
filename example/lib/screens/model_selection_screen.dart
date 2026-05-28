import 'dart:async';
import 'package:flutter/material.dart';
import 'package:stt_flutter/stt_flutter.dart';
import 'transcription_screen.dart';

class ModelSelectionScreen extends StatefulWidget {
  const ModelSelectionScreen({super.key});

  @override
  State<ModelSelectionScreen> createState() => _ModelSelectionScreenState();
}

class _ModelSelectionScreenState extends State<ModelSelectionScreen> {
  List<ModelDescriptor> _models = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadModels();
  }

  void _loadModels() {
    setState(() {
      _models = ModelRegistry.available();
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Select STT Model')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView.builder(
              itemCount: _models.length,
              itemBuilder: (context, i) {
                final model = _models[i];
                return ListTile(
                  leading: Icon(_iconForType(model.type)),
                  title: Text(model.name),
                  subtitle: Text(
                    '${model.languages.join(', ')}  •  ${_sizeStr(model.sizeMb)}',
                  ),
                  onTap: () => _selectModel(context, model),
                );
              },
            ),
    );
  }

  IconData _iconForType(SttModelType type) {
    switch (type) {
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

  void _selectModel(BuildContext context, ModelDescriptor model) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => TranscriptionScreen(model: model),
      ),
    );
  }
}
