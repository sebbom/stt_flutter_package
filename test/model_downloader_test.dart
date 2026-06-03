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
  });
}
