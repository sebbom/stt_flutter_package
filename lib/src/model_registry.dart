import 'stt_config.dart';
import 'default_models/register_defaults.dart';

/// Describes a single file that is part of a model.
///
/// Each model is composed of one or more files (e.g., encoder.onnx, decoder.onnx, tokens.txt)
/// that need to be downloaded and loaded together.
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

/// Describes a speech-to-text model that can be loaded and used for transcription.
///
/// Use [ModelRegistry.get] to retrieve a built-in model, or [ModelRegistry.register]
/// to add a custom model.
class ModelDescriptor {
  /// Unique identifier for this model (e.g., 'whisper-tiny')
  final String id;

  /// Human-readable name for this model
  final String name;

  /// The type of model, which determines which engine will be used
  final SttModelType type;

  /// List of supported language codes (ISO 639-1, e.g., 'en', 'de', 'fr')
  final List<String> languages;

  /// List of files that make up this model
  final List<ModelFile> files;

  /// Approximate size of the model in megabytes
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

/// Registry for speech-to-text models.
///
/// This class provides access to built-in models and allows registration of custom models.
/// Built-in models are registered lazily on first access.
///
/// Example:
/// ```dart
/// // Get a built-in model
/// final model = ModelRegistry.get('whisper-tiny');
///
/// // List all available models
/// final allModels = ModelRegistry.available();
///
/// // List models of a specific type
/// final whisperModels = ModelRegistry.available(type: SttModelType.whisper);
///
/// // Register a custom model
/// ModelRegistry.register(ModelDescriptor(
///   id: 'my-custom-model',
///   name: 'My Custom Model',
///   type: SttModelType.whisper,
///   languages: ['en'],
///   files: [ModelFile(url: '...', filename: 'encoder.onnx')],
///   sizeMb: 150,
/// ));
/// ```
class ModelRegistry {
  ModelRegistry._();

  static final Map<String, ModelDescriptor> _models = {};

  static bool _initialized = false;

  static void _ensureDefaults() {
    if (_initialized) return;
    _initialized = true;
    registerDefaultModels();
  }

  /// Registers a custom model with the registry.
  ///
  /// Use this to add models that are not included in the default set.
  /// The model can then be retrieved using [get] or [available].
  static void register(ModelDescriptor model) {
    _models[model.id] = model;
  }

  /// Returns a list of all available models.
  ///
  /// If [type] is provided, only models of that type are returned.
  /// Built-in models are registered lazily on first call.
  static List<ModelDescriptor> available({SttModelType? type}) {
    _ensureDefaults();
    if (type == null) return _models.values.toList();
    return _models.values.where((m) => m.type == type).toList();
  }

  /// Retrieves a model by its ID.
  ///
  /// Throws [ArgumentError] if no model with the given ID is registered.
  /// Built-in models are registered lazily on first call.
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
