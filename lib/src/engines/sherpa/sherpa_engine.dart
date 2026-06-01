import 'dart:io';
import 'dart:isolate';
import 'dart:math' show min;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import '../../stt_result.dart';
import '../../audio/audio_buffer.dart';
import '../inference_engine.dart';
import 'fbank.dart';

class SherpaInferenceEngine implements InferenceEngine {
  final ort.OnnxRuntime _runtime;
  ort.OrtSession? _encoderSession;
  ort.OrtSession? _decoderSession;
  ort.OrtSession? _joinerSession;
  Map<int, String>? _tokens;

  String _encInput = 'x';
  String _encOutput = 'encoder_out';
  String _decInput = 'y';
  String _decOutput = 'decoder_out';
  String _joinInputEnc = 'encoder_out';
  String _joinInputDec = 'decoder_out';
  String _joinOutput = 'logits';

  static const int blank = 0;
  static const int contextSize = 2;

  SherpaInferenceEngine(this._runtime);

  String _findFile(Map<String, String> files, List<String> patterns) {
    // First try exact match
    for (final p in patterns) {
      if (files.containsKey(p)) return files[p]!;
    }
    // Then try substring match (for archive-extracted files with versioned names)
    for (final p in patterns) {
      for (final entry in files.entries) {
        if (entry.key.contains(p)) return entry.value;
      }
    }
    throw FileSystemException('Model file not found for patterns: $patterns');
  }

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    _encoderSession = await _runtime.createSession(
      _findFile(modelFiles, ['encoder.onnx', 'encoder']),
    );
    _decoderSession = await _runtime.createSession(
      _findFile(modelFiles, ['decoder.onnx', 'decoder']),
    );
    _joinerSession = await _runtime.createSession(
      _findFile(modelFiles, ['joiner.onnx', 'joiner']),
    );

    // Detect encoder input/output names
    final encInputs = await _encoderSession!.getInputInfo();
    debugPrint('=== Sherpa Encoder inputs ===');
    for (final info in encInputs) {
      debugPrint('  $info');
    }
    if (encInputs.isNotEmpty) {
      _encInput = (encInputs.first['name'] as String?) ?? _encInput;
    }
    final encOutputs = await _encoderSession!.getOutputInfo();
    debugPrint('=== Sherpa Encoder outputs ===');
    for (final info in encOutputs) {
      debugPrint('  $info');
    }
    if (encOutputs.isNotEmpty) {
      _encOutput = (encOutputs.first['name'] as String?) ?? _encOutput;
    }

    // Detect decoder input/output names
    final decInputs = await _decoderSession!.getInputInfo();
    debugPrint('=== Sherpa Decoder inputs ===');
    for (final info in decInputs) {
      debugPrint('  $info');
    }
    // Find the main input (y / input_ids / tokens)
    for (final info in decInputs) {
      final name = (info['name'] as String?) ?? '';
      final shape = List<int>.from(info['shape'] as List<dynamic>? ?? []);
      if (shape.length == 2 && !name.startsWith('init_state')) {
        _decInput = name;
        break;
      }
    }
    final decOutputs = await _decoderSession!.getOutputInfo();
    debugPrint('=== Sherpa Decoder outputs ===');
    for (final info in decOutputs) {
      debugPrint('  $info');
    }
    if (decOutputs.isNotEmpty) {
      _decOutput = (decOutputs.first['name'] as String?) ?? _decOutput;
    }

    // Detect joiner input/output names
    final joinInputs = await _joinerSession!.getInputInfo();
    debugPrint('=== Sherpa Joiner inputs ===');
    for (final info in joinInputs) {
      debugPrint('  $info');
    }
    final names = joinInputs.map((i) => (i['name'] as String?) ?? '').toList();
    if (names.length >= 2) {
      _joinInputEnc = names[0];
      _joinInputDec = names[1];
    }
    final joinOutputs = await _joinerSession!.getOutputInfo();
    if (joinOutputs.isNotEmpty) {
      _joinOutput = (joinOutputs.first['name'] as String?) ?? _joinOutput;
    }

    // Load tokens.txt
    final tokensPath = modelFiles.entries
        .firstWhere((e) => e.key.contains('tokens'),
            orElse: () => const MapEntry('', ''))
        .value;
    if (tokensPath.isNotEmpty) {
      try {
        final text = await File(tokensPath).readAsString();
        _tokens = {};
        for (final line in text.split('\n')) {
          final trimmed = line.trim();
          if (trimmed.isEmpty) continue;
          final parts = trimmed.split(' ');
          if (parts.length >= 2) {
            _tokens![int.parse(parts[0])] = parts.sublist(1).join(' ');
          }
        }
        debugPrint('  => tokens loaded: ${_tokens!.length} entries');
      } catch (e) {
        debugPrint('  => tokens load FAILED: $e');
      }
    }
  }

  String _decode(List<int> ids) {
    if (_tokens == null) {
      return ids.map((i) => i.toString()).join(' ');
    }
    return ids.map((i) => _tokens![i] ?? '').join('').trim();
  }

  @override
  Future<SttResult> transcribe(AudioBuffer audio, {String? language}) async {
    final stopwatch = Stopwatch()..start();

    // 1. Compute Fbank features in background isolate
    final fbank = Fbank(nMels: 80);
    final features = await Isolate.run(() {
      final fb = Fbank(nMels: 80);
      return fb.compute(audio.samples);
    });

    final nFrames = features.length ~/ fbank.nMels;
    debugPrint('=== Sherpa transcribe: frames=$nFrames, nMels=${fbank.nMels} ===');

    // 2. Run encoder
    final featureTensor = await ort.OrtValue.fromList(
      Float32List.fromList(features.map((e) => e.toDouble()).toList()),
      [1, nFrames, fbank.nMels],
    );
    final encoderOut = await _encoderSession!.run({_encInput: featureTensor});
    final rawEnc = await encoderOut[_encOutput]!.asFlattenedList();
    await featureTensor.dispose();
    for (final v in encoderOut.values) {
      await v.dispose();
    }

    // Detect encoder output dim
    final encDim = rawEnc.length ~/ nFrames;

    // 3. Greedy RNNT decoding
    final resultTokens = <int>[];
    int t = 0;
    final maxTokens = 1000;

    // Create decoder state tensors (zero-initialized) if the decoder expects them
    final decInputInfos = await _decoderSession!.getInputInfo();
    final stateInputs = <String, ort.OrtValue>{};
    final stateOutputNames = <String>{};

    for (final info in decInputInfos) {
      final name = (info['name'] as String?) ?? '';
      if (name.startsWith('init_state') || name.startsWith('state')) {
        final shape = List<int>.from(info['shape'] as List<dynamic>? ?? [1]);
        final fixedShape = shape.map((s) => s == -1 ? 1 : s).toList();
        final total = fixedShape.fold(1, (a, b) => a * b);
        final type = (info['type'] as String? ?? 'float32').toLowerCase();
        if (type.contains('float')) {
          stateInputs[name] = await ort.OrtValue.fromList(Float32List(total), fixedShape);
        } else {
          stateInputs[name] = await ort.OrtValue.fromList(Float64List(total), fixedShape);
        }
      }
    }

    final decOutputInfos = await _decoderSession!.getOutputInfo();
    for (final info in decOutputInfos) {
      final name = (info['name'] as String?) ?? '';
      if (name.startsWith('init_state') || name.startsWith('state')) {
        stateOutputNames.add(name);
      }
    }

    debugPrint('  encoder: frames=$nFrames, dim=$encDim');
    debugPrint('  decoder states: ${stateInputs.length} in, ${stateOutputNames.length} out');

    int prevToken = blank;
    while (t < nFrames && resultTokens.length < maxTokens) {
      // Decoder forward
      final decInputIds = Int64List.fromList([prevToken]);
      final decInMap = <String, ort.OrtValue>{
        _decInput: await ort.OrtValue.fromList(decInputIds, [1, 1]),
      };
      decInMap.addAll(stateInputs);

      final decoderOut = await _decoderSession!.run(decInMap);
      final rawDec = await decoderOut[_decOutput]!.asFlattenedList();

      // Update states
      for (final sName in stateOutputNames) {
        final existing = stateInputs[sName];
        if (existing != null) {
          await existing.dispose();
        }
        stateInputs[sName] = decoderOut[sName]!;
      }

      // Joiner: combine encoder frame t + decoder output
      final encFrameStart = t * encDim;
      final encFrame = rawEnc.sublist(encFrameStart, min(encFrameStart + encDim, rawEnc.length));
      final joinInputs = <String, ort.OrtValue>{
        _joinInputEnc: await ort.OrtValue.fromList(
          Float32List.fromList(encFrame.cast<num>().map((e) => e.toDouble()).toList()),
          [1, 1, encDim],
        ),
        _joinInputDec: await ort.OrtValue.fromList(
          Float32List.fromList(rawDec.cast<num>().map((e) => e.toDouble()).toList()),
          [1, 1, rawDec.length],
        ),
      };

      final joinerOut = await _joinerSession!.run(joinInputs);
      final rawLogits = await joinerOut[_joinOutput]!.asFlattenedList();

      // Cleanup
      await decInMap[_decInput]!.dispose();
      await joinInputs[_joinInputEnc]!.dispose();
      await joinInputs[_joinInputDec]!.dispose();
      for (final v in decoderOut.values) {
        await v.dispose();
      }
      for (final v in joinerOut.values) {
        await v.dispose();
      }

      // Argmax
      int best = 0;
      double bestVal = double.negativeInfinity;
      for (int i = 0; i < rawLogits.length; i++) {
        final v = (rawLogits[i] as num).toDouble();
        if (v > bestVal) {
          bestVal = v;
          best = i;
        }
      }

      if (best == blank) {
        // Advance to next encoder frame
        t++;
        if (resultTokens.isNotEmpty) {
          prevToken = resultTokens.last;
        }
      } else {
        resultTokens.add(best);
        prevToken = best;
        if (resultTokens.length < 20 || resultTokens.length % 50 == 0) {
          debugPrint('  sherpa step t=$t: token=$best, partial="${_decode(resultTokens)}"');
        }
      }
    }

    // Cleanup states
    for (final v in stateInputs.values) {
      await v.dispose();
    }

    final text = _decode(resultTokens);
    stopwatch.stop();
    debugPrint('=== sherpa result: "$text" (${resultTokens.length} tokens) in ${stopwatch.elapsedMilliseconds}ms ===');

    return SttResult(
      text: text,
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
    );
  }

  @override
  Future<void> dispose() async {
    await _encoderSession?.close();
    await _decoderSession?.close();
    await _joinerSession?.close();
  }
}
