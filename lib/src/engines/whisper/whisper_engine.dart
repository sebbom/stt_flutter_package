import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import '../../stt_result.dart';
import '../../audio/audio_buffer.dart';
import '../inference_engine.dart';
import 'mel_spectrogram.dart';

class WhisperInferenceEngine implements InferenceEngine {
  final ort.OnnxRuntime _runtime;
  ort.OrtSession? _encoderSession;
  ort.OrtSession? _decoderSession;
  Map<int, String>? _vocab;

  static const int nMels = 80;
  static const int maxFrames = 3000;
  static const int maxTokens = 448;
  static const int encoderFrames = 1500;

  static const int sot = 50258;
  static const int eot = 50257;
  static const int transcribeTok = 50359;
  static const int noTimestamps = 50363;

  static const int en = 50259;
  static const int de = 50261;
  static const int fr = 50263;
  static const int es = 50265;

  int _dModel = 384;
  String _encoderInputName = 'input_features';
  String _encoderOutputName = 'last_hidden_state';
  String _decoderInputIds = 'input_ids';
  String _decoderInputStates = 'encoder_hidden_states';
  String _decoderOutputLogits = 'logits';
  final List<Map<String, dynamic>> _decoderExtraInputs = [];

  WhisperInferenceEngine(this._runtime);

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    _encoderSession = await _runtime.createSession(modelFiles['encoder.onnx']!);
    _decoderSession = await _runtime.createSession(modelFiles['decoder.onnx']!);

    final encInputs = await _encoderSession!.getInputInfo();
    debugPrint('=== Encoder inputs ===');
    for (final info in encInputs) {
      debugPrint('  $info');
    }
    if (encInputs.isNotEmpty) {
      _encoderInputName = encInputs.first['name'] as String? ?? _encoderInputName;
    }
    final encOutputs = await _encoderSession!.getOutputInfo();
    debugPrint('=== Encoder outputs ===');
    for (final info in encOutputs) {
      debugPrint('  $info');
    }
    if (encOutputs.isNotEmpty) {
      _encoderOutputName = encOutputs.first['name'] as String? ?? _encoderOutputName;
      final shape = encOutputs.first['shape'] as List<dynamic>?;
      if (shape != null && shape.length >= 3) {
        _dModel = (shape[2] as num).toInt();
        debugPrint('  => d_model = $_dModel');
      }
    }

    final decInputs = await _decoderSession!.getInputInfo();
    debugPrint('=== Decoder inputs ===');
    for (final info in decInputs) {
      debugPrint('  $info');
    }
    for (final info in decInputs) {
      final name = info['name'] as String? ?? '';
      if (name.startsWith('past')) continue;
      if (name.contains('input_ids')) {
        _decoderInputIds = name;
      } else if (name.contains('encoder_hidden_states')) {
        _decoderInputStates = name;
      } else {
        _decoderExtraInputs.add(info);
      }
    }
    final decOutputs = await _decoderSession!.getOutputInfo();
    debugPrint('=== Decoder outputs ===');
    for (final info in decOutputs) {
      debugPrint('  $info');
    }
    if (decOutputs.isNotEmpty) {
      _decoderOutputLogits = decOutputs.first['name'] as String? ?? _decoderOutputLogits;
    }

    if (modelFiles.containsKey('vocab.json')) {
      try {
        final json = jsonDecode(await File(modelFiles['vocab.json']!).readAsString());
        _vocab = {};
        for (final entry in (json as Map<String, dynamic>).entries) {
          _vocab![(entry.value as num).toInt()] = entry.key;
        }
        debugPrint('  => vocab loaded: ${_vocab!.length} entries');
      } catch (e) {
        debugPrint('  => vocab load FAILED: $e');
      }
    } else {
      debugPrint('  => vocab.json NOT in modelFiles');
    }
  }

  static int _languageToken(String? language) {
    switch (language?.toLowerCase()) {
      case 'de': case 'german': return de;
      case 'fr': case 'french': return fr;
      case 'es': case 'spanish': return es;
      default: return en;
    }
  }

  String _decode(List<int> tokens) {
    if (_vocab == null) {
      return tokens.map((t) => t.toString()).join(' ');
    }
    final b = StringBuffer();
    for (final t in tokens) {
      if (t >= 50256) continue;
      final s = _vocab![t];
      if (s != null) b.write(s);
    }
    return b.toString().replaceAll('Ġ', ' ').trim();
  }

  static int _argmax(List<dynamic> logits) {
    int best = 0;
    double bestVal = double.negativeInfinity;
    for (int i = 0; i < logits.length; i++) {
      final v = (logits[i] as num).toDouble();
      if (v > bestVal) {
        bestVal = v;
        best = i;
      }
    }
    return best;
  }

  Float32List _transposeMel(Float64List mel, int totalFrames, int chunkOffset, int chunkSize) {
    final actual = (totalFrames - chunkOffset).clamp(0, chunkSize);
    final out = Float32List(nMels * chunkSize);
    for (int t = 0; t < actual; t++) {
      final srcBase = (chunkOffset + t) * nMels;
      for (int m = 0; m < nMels; m++) {
        out[m * chunkSize + t] = mel[srcBase + m];
      }
    }
    return out;
  }

  @override
  Future<SttResult> transcribe(AudioBuffer audio, {String? language}) async {
    final stopwatch = Stopwatch()..start();
    final mel = await Isolate.run(() => MelSpectrogram.compute(audio.samples));
    final totalFrames = mel.length ~/ nMels;
    final fullText = StringBuffer();
    final langToken = _languageToken(language);
    final prompt = [sot, langToken, transcribeTok, noTimestamps];

    debugPrint('=== transcribe: audio=${audio.length}samples, melFrames=$totalFrames, langToken=$langToken ===');

    for (int offset = 0; offset < totalFrames && offset < maxFrames * 100; offset += maxFrames) {
      final chunk = _transposeMel(mel, totalFrames, offset, maxFrames);

      final inputTensor = await ort.OrtValue.fromList(chunk, [1, nMels, maxFrames]);
      final encoderOut = await _encoderSession!.run({_encoderInputName: inputTensor});
      final rawStates = await encoderOut[_encoderOutputName]!.asFlattenedList();
      final encData = Float32List.fromList(rawStates.cast<num>().map((e) => e.toDouble()).toList());

      await inputTensor.dispose();
      for (final v in encoderOut.values) {
        await v.dispose();
      }

      debugPrint('  encoder: offset=$offset, output=${encData.length} floats');

      final encTensor = await ort.OrtValue.fromList(encData, [1, encoderFrames, _dModel]);
      final tokens = <int>[...prompt];

      for (int i = 0; i < maxTokens - prompt.length; i++) {
        final ids = Int64List.fromList(tokens.map((t) => t.toInt()).toList());
        final idTensor = await ort.OrtValue.fromList(ids, [1, tokens.length]);

        final decoderInputs = <String, ort.OrtValue>{
          _decoderInputIds: idTensor,
          _decoderInputStates: encTensor,
        };
        for (final extra in _decoderExtraInputs) {
          final eName = extra['name'] as String? ?? '';
          final extraShape = List<int>.from(extra['shape'] as List<dynamic>? ?? [1]);
          final extraType = (extra['type'] as String? ?? 'float32').toLowerCase();
          final fixedShape = extraShape.map((s) => s == -1 ? 1 : s).toList();
          final total = fixedShape.fold(1, (a, b) => a * b);

          if (extraType.contains('bool')) {
            decoderInputs[eName] = await ort.OrtValue.fromList(List<bool>.filled(total, false), fixedShape);
          } else if (extraType.contains('int64')) {
            final data = Int64List(total);
            decoderInputs[eName] = await ort.OrtValue.fromList(data, fixedShape);
          } else if (extraType.contains('int32')) {
            final data = Int32List(total);
            decoderInputs[eName] = await ort.OrtValue.fromList(data, fixedShape);
          } else {
            final data = Float32List(total);
            decoderInputs[eName] = await ort.OrtValue.fromList(data, fixedShape);
          }
        }

        final decoderOut = await _decoderSession!.run(decoderInputs);
        final rawLogits = await decoderOut[_decoderOutputLogits]!.asFlattenedList();

        final vocabSize = rawLogits.length ~/ tokens.length;
        final lastStart = (tokens.length - 1) * vocabSize;
        final lastLogits = lastStart + vocabSize <= rawLogits.length
            ? rawLogits.sublist(lastStart, lastStart + vocabSize)
            : rawLogits;

        final next = _argmax(lastLogits);

        await idTensor.dispose();
        for (final v in decoderOut.values) {
          await v.dispose();
        }

        if (next == eot) break;
        tokens.add(next);

        if (i < 20 || i % 50 == 0) {
          final partial = _decode(tokens.sublist(prompt.length));
          debugPrint('  decoder step $i: token=$next, partial="$partial"');
        }
      }

      await encTensor.dispose();

      final text = _decode(tokens.sublist(prompt.length));
      debugPrint('  chunk text: "$text" (${tokens.length - prompt.length} tokens)');
      if (text.isNotEmpty) {
        if (fullText.isNotEmpty) fullText.write(' ');
        fullText.write(text);
      }
    }

    stopwatch.stop();
    debugPrint('=== result: "${fullText.toString()}" in ${stopwatch.elapsedMilliseconds}ms ===');
    return SttResult(
      text: fullText.toString(),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
    );
  }

  @override
  Future<void> dispose() async {
    await _encoderSession?.close();
    await _decoderSession?.close();
  }
}
