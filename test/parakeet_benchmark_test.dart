// ignore_for_file: avoid_print

@TestOn('linux')
library;

import 'dart:io';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import 'package:stt_flutter/stt_flutter.dart';

int _levenshtein(String a, String b) {
  final m = a.length, n = b.length;
  final dp = List.generate(m + 1, (_) => List.filled(n + 1, 0));
  for (int i = 0; i <= m; i++) {
    dp[i][0] = i;
  }
  for (int j = 0; j <= n; j++) {
    dp[0][j] = j;
  }
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      dp[i][j] = [
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1),
      ].reduce(math.min);
    }
  }
  return dp[m][n];
}

bool _wordsMatch(String a, String b) {
  if (a == b) return true;
  final dist = _levenshtein(a, b);
  final maxLen = math.max(a.length, b.length);
  return maxLen >= 3 && dist / maxLen <= 0.25;
}

double _wordErrorRate(String hypothesis, String reference) {
  final hyp = hypothesis
      .trim()
      .toLowerCase()
      .split(RegExp(r'\s+'))
      .where((w) => w.isNotEmpty)
      .toList();
  final ref = reference
      .trim()
      .toLowerCase()
      .split(RegExp(r'\s+'))
      .where((w) => w.isNotEmpty)
      .toList();
  final m = hyp.length, n = ref.length;
  if (n == 0) return m == 0 ? 0.0 : 1.0;
  if (m == 0) return 1.0;

  final dp = List.generate(m + 1, (_) => List.filled(n + 1, 1.0));
  for (int i = 0; i <= m; i++) {
    dp[i][0] = i.toDouble();
  }
  for (int j = 0; j <= n; j++) {
    dp[0][j] = j.toDouble();
  }
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      final cost = _wordsMatch(hyp[i - 1], ref[j - 1]) ? 0.0 : 1.0;
      dp[i][j] = [
        dp[i - 1][j] + 1.0,
        dp[i][j - 1] + 1.0,
        dp[i - 1][j - 1] + cost,
      ].reduce(math.min);
    }
  }
  return dp[m][n] / n;
}

void main() {
  final shouldRun = Platform.environment['PARAKEET_BENCHMARK'] == 'true';

  if (!shouldRun) {
    test('skip (set PARAKEET_BENCHMARK=true to run)', () {});
    return;
  }

  const modelId = 'parakeet-tdt-0.6b-multilingual';
  final home = Platform.environment['HOME'] ?? '/tmp';
  final modelDir = Platform.environment['PARAKEET_MODEL_DIR'] ??
      '$home/.cache/stt_benchmark/$modelId';

  final wavDir = Directory.current.path;
  final enWav = '$wavDir/example/assets/hello_en.wav';
  final enRef = '$wavDir/example/assets/hello_en_tr.md';
  final frWav = '$wavDir/example/assets/podcast_fr.wav';
  final frRef = '$wavDir/example/assets/podcast_fr.md';

  late SttFlutter stt;
  bool nativeAvailable = true;

  setUpAll(() async {
    if (!File(enWav).existsSync()) {
      throw Exception(
          'WAV file not found: $enWav\nRun from the package root directory.');
    }

    final model = ModelRegistry.get(modelId);
    final already = await ModelDownloader.isDownloaded(
      model,
      storagePath: modelDir,
    );

    if (!already) {
      print('Downloading $modelId (~640 MB) to: $modelDir');
      await ModelDownloader.download(model, storagePath: modelDir);
      print('Download complete.');
    } else {
      print('Model found at: $modelDir');
    }

    try {
      initBindings();
    } catch (e) {
      nativeAvailable = false;
      print('SKIPPING: sherpa_onnx native library not available — $e');
      return;
    }

    stt = SttFlutter();
    try {
      await stt.initialize(model: model, modelDir: modelDir);
    } catch (e) {
      nativeAvailable = false;
      print('SKIPPING: native FFI unavailable — $e');
    }
  });

  tearDownAll(() async {
    try {
      await stt.dispose();
    } catch (_) {}
  });

  group('Parakeet benchmark', () {
    test('hello_en.wav — English transcription & WER', () async {
      if (!nativeAvailable) return;

      final expected = await File(enRef).readAsString();
      final result =
          await stt.transcribeFile(enWav).timeout(const Duration(minutes: 30));

      final wer = _wordErrorRate(result.text, expected);

      print('');
      print('=== Parakeet Benchmark: hello_en.wav (English, 8kHz → 16kHz) ===');
      print('Inference time : ${result.inferenceTimeMs.toStringAsFixed(0)} ms');
      print('Confidence     : ${result.confidence?.toStringAsFixed(4) ?? "?"}');
      print(
          'Duration       : ${result.durationMs?.toStringAsFixed(0) ?? "?"} ms');
      print('Language       : ${result.lang ?? "?"}');
      print('WER            : ${(wer * 100).toStringAsFixed(2)}%');
      print('');

      final hypShort = result.text.length > 300
          ? '${result.text.substring(0, 300)}…'
          : result.text;
      final refShort =
          expected.length > 300 ? '${expected.substring(0, 300)}…' : expected;
      print('--- Hypothesis ---');
      print(hypShort);
      print('');
      print('--- Reference ---');
      print(refShort);
      print('');

      expect(result.text.isNotEmpty, isTrue);
      expect(wer, lessThan(1.0));
    }, timeout: const Timeout(Duration(minutes: 35)));

    test('podcast_fr.wav — French transcription & WER', () async {
      if (!nativeAvailable) return;

      final expected = await File(frRef).readAsString();
      final result = await stt
          .transcribeFile(frWav, language: 'fr')
          .timeout(const Duration(minutes: 60));

      final wer = _wordErrorRate(result.text, expected);

      print('');
      print('=== Parakeet Benchmark: podcast_fr.wav (French, 16kHz) ===');
      print('Inference time : ${result.inferenceTimeMs.toStringAsFixed(0)} ms');
      print('Confidence     : ${result.confidence?.toStringAsFixed(4) ?? "?"}');
      print(
          'Duration       : ${result.durationMs?.toStringAsFixed(0) ?? "?"} ms');
      print('Language       : ${result.lang ?? "?"}');
      print('WER            : ${(wer * 100).toStringAsFixed(2)}%');
      print('');

      final hypShort = result.text.length > 300
          ? '${result.text.substring(0, 300)}…'
          : result.text;
      final refShort =
          expected.length > 300 ? '${expected.substring(0, 300)}…' : expected;
      print('--- Hypothesis ---');
      print(hypShort);
      print('');
      print('--- Reference ---');
      print(refShort);
      print('');

      expect(result.text.isNotEmpty, isTrue);
      expect(wer, lessThan(1.0));
    }, timeout: const Timeout(Duration(minutes: 65)));
  });
}
