import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';
import '../stt_flutter.dart';

/// Handles downloading and caching of model files.
///
/// Models are downloaded from their remote URLs (typically HuggingFace or GitHub),
/// verified with SHA256 checksums, and cached locally for offline use.
/// Tar.bz2 archives (used by Sherpa models) are automatically extracted.
///
/// Example:
/// ```dart
/// final model = ModelRegistry.get('whisper-tiny');
/// await ModelDownloader.download(model, onProgress: (received, total) {
///   print('Downloaded $received of $total bytes');
/// });
/// ```
class ModelDownloader {
  /// Returns the default storage path for a model.
  ///
  /// Models are stored in `{appDocumentsDirectory}/stt_models/{model.id}/`
  static Future<String> defaultStoragePath(ModelDescriptor model) async {
    final dir = await getApplicationDocumentsDirectory();
    return '${dir.path}/stt_models/${model.id}';
  }

  /// Downloads all files for a model and caches them locally.
  ///
  /// [model]: The model descriptor containing the files to download
  /// [storagePath]: Optional custom storage directory. Defaults to [defaultStoragePath]
  /// [client]: Optional HTTP client for testing. If not provided, a new client is created
  /// [onProgress]: Callback for overall download progress (bytes received, total bytes)
  /// [onFileProgress]: Callback for per-file download progress (filename, bytes received, total bytes)
  ///
  /// Files are verified with SHA256 checksums if provided in the model descriptor.
  /// Tar.bz2 archives are automatically extracted after download.
  ///
  /// Throws [SttException] if download fails or checksum verification fails
  static Future<void> download(
    ModelDescriptor model, {
    String? storagePath,
    http.Client? client,
    void Function(int received, int total)? onProgress,
    void Function(String file, int received, int total)? onFileProgress,
  }) async {
    final dir = storagePath ?? await defaultStoragePath(model);
    await Directory(dir).create(recursive: true);

    final effectiveClient = client ?? http.Client();
    try {
      for (final file in model.files) {
        final destPath = '$dir/${file.filename}';
        if (await _needsDownload(file, destPath)) {
          await downloadFile(
            client: effectiveClient,
            url: file.url,
            destPath: destPath,
            sha256: file.sha256,
            expectedSizeBytes: file.sizeBytes,
            onProgress: (received, total) {
              onFileProgress?.call(file.filename, received, total);
            },
          );
        }
      }

      for (final file in model.files) {
        if (file.filename.endsWith('.tar.bz2')) {
          await _extractTarBz2('$dir/${file.filename}', dir);
        }
      }
    } finally {
      if (client == null) effectiveClient.close();
    }
  }

  /// Checks if all files for a model are already downloaded and cached.
  ///
  /// [model]: The model descriptor to check
  /// [storagePath]: Optional custom storage directory. Defaults to [defaultStoragePath]
  ///
  /// Returns true if all files exist and have the correct size (if specified in the descriptor).
  /// Files with mismatched sizes are deleted to force a re-download.
  static Future<bool> isDownloaded(
    ModelDescriptor model, {
    String? storagePath,
  }) async {
    final dir = storagePath ?? await defaultStoragePath(model);
    for (final file in model.files) {
      if (await _needsDownload(file, '$dir/${file.filename}')) return false;
    }
    return true;
  }

  static Future<bool> _needsDownload(ModelFile file, String destPath) async {
    if (file.filename.endsWith('.tar.bz2')) return true;
    final f = File(destPath);
    if (!await f.exists()) return true;
    if (file.sizeBytes != null && (await f.length()) != file.sizeBytes) {
      await f.delete();
      return true;
    }
    return false;
  }

  static Future<void> downloadFile({
    required String url,
    required String destPath,
    String? sha256,
    int? expectedSizeBytes,
    http.Client? client,
    void Function(int received, int total)? onProgress,
  }) async {
    final effectiveClient = client ?? http.Client();
    try {
      final response =
          await effectiveClient.send(http.Request('GET', Uri.parse(url)));
      if (response.statusCode != 200) {
        throw SttException.downloadFailed('HTTP ${response.statusCode} for $url');
      }

      final total = expectedSizeBytes ?? response.contentLength ?? -1;
      final file = File(destPath);
      await file.create(recursive: true);
      final sink = file.openWrite();

      int received = 0;
      await for (final chunk in response.stream) {
        sink.add(chunk);
        received += chunk.length;
        onProgress?.call(received, total);
      }

      await sink.close();

      if (total > 0 && received != total) {
        await file.delete();
        throw SttException.downloadFailed(
          'Incomplete: $received of $total bytes. Connection may have dropped. Please retry.',
        );
      }

      if (sha256 != null) {
        final fileHash = await _computeSha256(destPath);
        if (fileHash != sha256) {
          await file.delete();
          throw SttException.downloadFailed(
            'SHA256 mismatch for ${file.path}. Expected $sha256, got $fileHash',
          );
        }
        SttLogger.d('SHA256 verified for ${file.path}');
      }
    } finally {
      if (client == null) effectiveClient.close();
    }
  }

  static Future<String> _computeSha256(String path) async {
    final file = File(path);
    final bytes = await file.readAsBytes();
    return sha256.convert(bytes).toString();
  }

  static Future<void> _extractTarBz2(String archivePath, String destDir) async {
    final bytes = await File(archivePath).readAsBytes();

    final bz2decoder = BZip2Decoder();
    final decodedBytes = bz2decoder.decodeBytes(bytes);

    final tarDecoder = TarDecoder();
    final archive = tarDecoder.decodeBytes(decodedBytes);

    for (final entry in archive) {
      if (entry.isFile) {
        final parts = entry.name.split('/');
        final name = parts.skip(1).join('/');
        if (name.isEmpty) continue;

        final destPath = '$destDir/$name';
        final destFile = File(destPath);
        await destFile.create(recursive: true);
        await destFile.writeAsBytes(entry.content as List<int>);
      }
    }
  }
}
