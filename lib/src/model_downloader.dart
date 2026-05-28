import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:archive/archive.dart';
import '../stt_flutter.dart';

class ModelDownloader {
  static Future<String> defaultStoragePath(ModelDescriptor model) async {
    final dir = await getApplicationDocumentsDirectory();
    return '${dir.path}/stt_models/${model.id}';
  }

  static Future<void> download(
    ModelDescriptor model, {
    String? storagePath,
    void Function(int received, int total)? onProgress,
    void Function(String file, int received, int total)? onFileProgress,
  }) async {
    final dir = storagePath ?? await defaultStoragePath(model);
    await Directory(dir).create(recursive: true);

    for (final file in model.files) {
      await _downloadFile(
        url: file.url,
        destPath: '$dir/${file.filename}',
        onProgress: (received, total) {
          onFileProgress?.call(file.filename, received, total);
        },
      );
    }

    // Extract .tar.bz2 archives (Sherpa models)
    for (final file in model.files) {
      if (file.filename.endsWith('.tar.bz2')) {
        await _extractTarBz2('$dir/${file.filename}', dir);
      }
    }
  }

  static Future<bool> isDownloaded(
    ModelDescriptor model, {
    String? storagePath,
  }) async {
    final dir = storagePath ?? await defaultStoragePath(model);
    for (final file in model.files) {
      final f = File('$dir/${file.filename}');
      if (!await f.exists()) return false;
    }
    return true;
  }

  static Future<void> _downloadFile({
    required String url,
    required String destPath,
    void Function(int received, int total)? onProgress,
  }) async {
    final response = await http.Client().send(http.Request('GET', Uri.parse(url)));
    if (response.statusCode != 200) {
      throw HttpException('Download failed: ${response.statusCode} $url');
    }

    final total = response.contentLength ?? -1;
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
  }

  static Future<void> _extractTarBz2(String archivePath, String destDir) async {
    final bytes = await File(archivePath).readAsBytes();

    // BZip2 decode
    final bz2decoder = BZip2Decoder();
    final decodedBytes = bz2decoder.decodeBytes(bytes);

    // Tar decode
    final tarDecoder = TarDecoder();
    final archive = tarDecoder.decodeBytes(decodedBytes);

    for (final entry in archive) {
      if (entry.isFile) {
        // Strip the top-level directory from the entry name
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
