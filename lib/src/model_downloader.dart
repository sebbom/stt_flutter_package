import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';
import '../stt_flutter.dart';

class ModelDownloader {
  static Future<String> defaultStoragePath(ModelDescriptor model) async {
    final dir = await getApplicationDocumentsDirectory();
    return '${dir.path}/stt_models/${model.id}';
  }

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
