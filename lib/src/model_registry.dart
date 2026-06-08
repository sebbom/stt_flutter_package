import 'stt_config.dart';
import 'default_models/register_defaults.dart';

class ModelFile {
  final String url;
  final String filename;
  final String? sha256;
  final int? sizeBytes;

  /// Local file path to a hotwords text file consumed by Zipformer-based
  /// recognizers. One entry per line, formatted as `"word score"`.
  final String? hotwordsFile;

  /// Score applied to entries in [hotwordsFile]. Default is 1.5.
  final double? hotwordsScore;

  /// Raw hotwords string (used by Qwen3-ASR which accepts a string instead
  /// of a file). Format is the same as [hotwordsFile]: one entry per line,
  /// `"word score"`.
  final String? hotwordsString;

  const ModelFile({
    required this.url,
    required this.filename,
    this.sha256,
    this.sizeBytes,
    this.hotwordsFile,
    this.hotwordsScore,
    this.hotwordsString,
  });
}

class ModelDescriptor {
  final String id;
  final String name;
  final SttModelType type;
  final List<String> languages;
  final List<ModelFile> files;
  final int sizeMb;

  const ModelDescriptor({
    required this.id,
    required this.name,
    required this.type,
    required this.languages,
    required this.files,
    required this.sizeMb,
  });
}

class ModelRegistry {
  ModelRegistry._();

  static final Map<String, ModelDescriptor> _models = {};

  static bool _initialized = false;

  static void _ensureDefaults() {
    if (_initialized) return;
    _initialized = true;
    registerDefaultModels();
  }

  static void register(ModelDescriptor model) {
    _models[model.id] = model;
  }

  static List<ModelDescriptor> available({SttModelType? type}) {
    _ensureDefaults();
    if (type == null) return _models.values.toList();
    return _models.values.where((m) => m.type == type).toList();
  }

  static ModelDescriptor get(String id) {
    _ensureDefaults();
    final model = _models[id];
    if (model == null) throw ArgumentError('Unknown model id: $id');
    return model;
  }

  static bool isRegistered(String id) {
    _ensureDefaults();
    return _models.containsKey(id);
  }
}
