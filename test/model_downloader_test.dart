import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:stt_flutter/stt_flutter.dart';

class _MockClient extends http.BaseClient {
  _MockClient(this._handler);
  final Future<http.StreamedResponse> Function(http.BaseRequest req) _handler;
  int closeCount = 0;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    return _handler(request);
  }

  @override
  void close() {
    closeCount++;
    super.close();
  }
}

Future<TempDir> _mkTemp() async {
  final d = await Directory.systemTemp.createTemp('stt_test_');
  return TempDir(d);
}

class TempDir {
  final Directory dir;
  TempDir(this.dir);
  String path() => dir.path;
  Future<void> cleanup() => dir.delete(recursive: true);
}

void main() {
  group('ModelDownloader.downloadFile', () {
    test('writes the body to disk and verifies SHA256', () async {
      final body = utf8.encode('hello world');
      final hash =
          'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9';
      final client = _MockClient((req) async {
        return http.StreamedResponse(
          Stream.value(body),
          200,
          contentLength: body.length,
        );
      });

      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/hello.txt');
      try {
        await ModelDownloader.downloadFile(
          client: client,
          url: 'https://example.com/hello.txt',
          destPath: dest.path,
          sha256: hash,
        );
        expect(await dest.exists(), true);
        expect(await dest.readAsString(), 'hello world');
        // User-supplied client is the caller's responsibility to close.
        expect(client.closeCount, 0);
      } finally {
        await tmp.cleanup();
      }
    });

    test('throws SttException on SHA256 mismatch and deletes the file',
        () async {
      final body = utf8.encode('hello world');
      final client = _MockClient((req) async {
        return http.StreamedResponse(
          Stream.value(body),
          200,
          contentLength: body.length,
        );
      });
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/hello.txt');
      try {
        await expectLater(
          ModelDownloader.downloadFile(
            client: client,
            url: 'https://example.com/hello.txt',
            destPath: dest.path,
            sha256: '0' * 64,
          ),
          throwsA(isA<SttException>()),
        );
        expect(await dest.exists(), false);
        expect(client.closeCount, 0);
      } finally {
        await tmp.cleanup();
      }
    });

    test('throws SttException on non-200 status', () async {
      final client = _MockClient((req) async {
        return http.StreamedResponse(
          const Stream<List<int>>.empty(),
          404,
        );
      });
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/missing.txt');
      try {
        await expectLater(
          ModelDownloader.downloadFile(
            client: client,
            url: 'https://example.com/missing',
            destPath: dest.path,
          ),
          throwsA(isA<SttException>()),
        );
        expect(client.closeCount, 0);
      } finally {
        await tmp.cleanup();
      }
    });

    test('user-supplied client is NOT auto-closed (caller owns it)', () async {
      final body = utf8.encode('x');
      final client = _MockClient((req) async {
        return http.StreamedResponse(
          Stream.value(body),
          200,
          contentLength: body.length,
        );
      });
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/owned.txt');
      try {
        await ModelDownloader.downloadFile(
          client: client,
          url: 'https://example.com/x',
          destPath: dest.path,
        );
        expect(client.closeCount, 0,
            reason: 'caller-supplied client must not be auto-closed');
        client.close();
        expect(client.closeCount, 1);
      } finally {
        await tmp.cleanup();
      }
    });

    test('throws and deletes the file when body is shorter than expectedSizeBytes',
        () async {
      final body = utf8.encode('short');
      final client = _MockClient((req) async {
        return http.StreamedResponse(
          Stream.value(body),
          200,
        );
      });
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/partial.bin');
      try {
        await expectLater(
          ModelDownloader.downloadFile(
            client: client,
            url: 'https://example.com/big.bin',
            destPath: dest.path,
            expectedSizeBytes: 1024,
          ),
          throwsA(predicate<SttException>(
            (e) => e.message.contains('Incomplete') &&
                e.message.contains('1024'),
            'expected Incomplete-size SttException',
          )),
        );
        expect(await dest.exists(), false,
            reason: 'partial file must be deleted on size mismatch');
      } finally {
        await tmp.cleanup();
      }
    });

    test('uses expectedSizeBytes for progress when server omits contentLength',
        () async {
      final body = utf8.encode('x' * 100);
      final client = _MockClient((req) async {
        return http.StreamedResponse(
          Stream.value(body),
          200,
        );
      });
      final totals = <int>[];
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/progress.bin');
      try {
        await ModelDownloader.downloadFile(
          client: client,
          url: 'https://example.com/p.bin',
          destPath: dest.path,
          expectedSizeBytes: 100,
          onProgress: (received, total) => totals.add(total),
        );
        expect(totals, isNotEmpty);
        expect(totals.last, 100,
            reason: 'progress total should reflect expectedSizeBytes');
      } finally {
        await tmp.cleanup();
      }
    });
  });

  group('ModelDownloader.isDownloaded (size verification)', () {
    ModelDescriptor fakeModel({
      required List<ModelFile> files,
      String id = 'fake-model',
    }) {
      return ModelDescriptor(
        id: id,
        name: 'fake',
        type: SttModelType.sherpa,
        languages: const ['en'],
        files: files,
        sizeMb: 1,
      );
    }

    test('returns false and deletes the partial file when size differs', () async {
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/encoder.int8.onnx');
      await dest.writeAsBytes(List<int>.filled(66 * 1024 * 1024, 0));
      final model = fakeModel(files: [
        ModelFile(
          url: 'https://example.com/encoder.int8.onnx',
          filename: 'encoder.int8.onnx',
          sizeBytes: 652184281,
        ),
      ]);
      try {
        final ok = await ModelDownloader.isDownloaded(
          model,
          storagePath: tmp.path(),
        );
        expect(ok, isFalse,
            reason: 'truncated file must be treated as not downloaded');
        expect(await dest.exists(), isFalse,
            reason: 'partial file must be removed to force a clean re-download');
      } finally {
        await tmp.cleanup();
      }
    });

    test('returns true when on-disk size matches declared sizeBytes', () async {
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/x.bin');
      await dest.writeAsBytes(List<int>.filled(100, 0));
      final model = fakeModel(files: [
        ModelFile(
          url: 'https://example.com/x.bin',
          filename: 'x.bin',
          sizeBytes: 100,
        ),
      ]);
      try {
        expect(
          await ModelDownloader.isDownloaded(model, storagePath: tmp.path()),
          isTrue,
        );
        expect(await dest.exists(), isTrue,
            reason: 'valid file must not be deleted');
      } finally {
        await tmp.cleanup();
      }
    });

    test('returns false when file is missing', () async {
      final tmp = await _mkTemp();
      final model = fakeModel(files: [
        ModelFile(
          url: 'https://example.com/x.bin',
          filename: 'x.bin',
          sizeBytes: 100,
        ),
      ]);
      try {
        expect(
          await ModelDownloader.isDownloaded(model, storagePath: tmp.path()),
          isFalse,
        );
      } finally {
        await tmp.cleanup();
      }
    });

    test('returns true when sizeBytes is null and file exists (back-compat)',
        () async {
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/x.bin');
      await dest.writeAsBytes(List<int>.filled(7, 0));
      final model = fakeModel(files: [
        ModelFile(url: 'https://example.com/x.bin', filename: 'x.bin'),
      ]);
      try {
        expect(
          await ModelDownloader.isDownloaded(model, storagePath: tmp.path()),
          isTrue,
        );
      } finally {
        await tmp.cleanup();
      }
    });

    test('download() re-fetches a truncated file end-to-end', () async {
      final tmp = await _mkTemp();
      final dest = File('${tmp.path()}/encoder.int8.onnx');
      await dest.writeAsBytes(List<int>.filled(1024, 0));

      final fullBody = utf8.encode('x' * 4096);
      var requests = 0;
      final client = _MockClient((req) async {
        requests++;
        return http.StreamedResponse(
          Stream.value(fullBody),
          200,
          contentLength: fullBody.length,
        );
      });

      final model = fakeModel(files: [
        ModelFile(
          url: 'https://example.com/encoder.int8.onnx',
          filename: 'encoder.int8.onnx',
          sizeBytes: 4096,
        ),
      ]);
      try {
        await ModelDownloader.download(
          model,
          storagePath: tmp.path(),
          client: client,
        );
        expect(requests, 1,
            reason: 'truncated file must trigger exactly one re-download');
        expect(await dest.length(), 4096);
        expect(await dest.readAsString(), 'x' * 4096);
      } finally {
        await tmp.cleanup();
      }
    });
  });
}
