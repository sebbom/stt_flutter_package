import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart' as ort;
import '../../stt_result.dart';
import '../../cancellation_token.dart';
import '../../stt_logger.dart';
import '../../audio/audio_buffer.dart';
import '../inference_engine.dart';
import '../../utils/math_utils.dart';
import '../../audio/mel_spectrogram.dart';

class WhisperInferenceEngine implements InferenceEngine {
  final ort.OnnxRuntime _runtime;
  ort.OrtSession? _encoderSession;
  ort.OrtSession? _decoderSession;
  Map<int, String>? _vocab;

  int nMels = 80;
  int maxFrames = 3000;
  static const int maxTokens = 448;
  int encoderFrames = 1500;

  static const int sot = 50258;
  static const int eot = 50257;
  static const int transcribeTok = 50359;
  static const int noTimestamps = 50363;

  static const int en = 50259;
  static const int de = 50261;
  static const int fr = 50263;
  static const int es = 50265;
  static const int pt = 50267;
  static const int ja = 50249;
  static const int zh = 50250;
  static const int ru = 50260;
  static const int it = 50264;
  static const int nl = 50262;
  static const int pl = 50268;
  static const int tr = 50272;
  static const int ar = 50270;

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
    SttLogger.d('=== Encoder inputs ===');
    for (final info in encInputs) {
      SttLogger.d('  $info');
    }
    if (encInputs.isNotEmpty) {
      _encoderInputName = encInputs.first['name'] as String? ?? _encoderInputName;
      final shape = encInputs.first['shape'] as List<dynamic>?;
      if (shape != null && shape.length >= 3) {
        nMels = (shape[1] as num).toInt();
        maxFrames = (shape[2] as num).toInt();
        encoderFrames = maxFrames ~/ 2;
        SttLogger.d('  => encoder input: nMels=$nMels, maxFrames=$maxFrames, encoderFrames=$encoderFrames');
      }
    }
    final encOutputs = await _encoderSession!.getOutputInfo();
    SttLogger.d('=== Encoder outputs ===');
    for (final info in encOutputs) {
      SttLogger.d('  $info');
    }
    if (encOutputs.isNotEmpty) {
      _encoderOutputName = encOutputs.first['name'] as String? ?? _encoderOutputName;
      final shape = encOutputs.first['shape'] as List<dynamic>?;
      if (shape != null && shape.length >= 3) {
        _dModel = (shape[2] as num).toInt();
        SttLogger.d('  => d_model = $_dModel');
      }
    }

    final decInputs = await _decoderSession!.getInputInfo();
    SttLogger.d('=== Decoder inputs ===');
    for (final info in decInputs) {
      SttLogger.d('  $info');
    }
    for (final info in decInputs) {
      final name = info['name'] as String? ?? '';
      if (name.contains('input_ids')) {
        _decoderInputIds = name;
      } else if (name.contains('encoder_hidden_states')) {
        _decoderInputStates = name;
      } else {
        _decoderExtraInputs.add(info);
      }
    }
    final decOutputs = await _decoderSession!.getOutputInfo();
    SttLogger.d('=== Decoder outputs ===');
    for (final info in decOutputs) {
      SttLogger.d('  $info');
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
        SttLogger.d('  => vocab loaded: ${_vocab!.length} entries');
      } catch (e) {
        SttLogger.d('  => vocab load FAILED: $e');
      }
    } else {
      SttLogger.d('  => vocab.json NOT in modelFiles');
    }
  }

  static int _languageToken(String? language) {
    switch (language?.toLowerCase()) {
      case 'de': case 'german': return de;
      case 'fr': case 'french': return fr;
      case 'es': case 'spanish': return es;
      case 'pt': case 'portuguese': return pt;
      case 'ja': case 'japanese': return ja;
      case 'zh': case 'chinese': return zh;
      case 'ru': case 'russian': return ru;
      case 'it': case 'italian': return it;
      case 'nl': case 'dutch': return nl;
      case 'pl': case 'polish': return pl;
      case 'tr': case 'turkish': return tr;
      case 'ar': case 'arabic': return ar;
      default: return en;
    }
  }

  static final Map<int, int> _unicodeToByte = _buildUtob();

  static Map<int, int> _buildUtob() {
    final btou = <int, int>{};
    for (int b = 33; b <= 126; b++) btou[b] = b;
    for (int b = 161; b <= 172; b++) btou[b] = b;
    for (int b = 174; b <= 255; b++) btou[b] = b;
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!btou.containsKey(b)) {
        btou[b] = 256 + n;
        n++;
      }
    }
    final utob = <int, int>{};
    for (final entry in btou.entries) {
      utob[entry.value] = entry.key;
    }
    return utob;
  }

  String _decode(List<int> tokens) {
    if (_vocab == null) {
      return tokens.map((t) => t.toString()).join(' ');
    }
    final bytes = <int>[];
    for (final t in tokens) {
      if (t >= 50256) continue;
      final s = _vocab![t];
      if (s == null) continue;
      for (int i = 0; i < s.length; i++) {
        final cp = s.codeUnitAt(i);
        bytes.add(_unicodeToByte[cp] ?? cp);
      }
    }
    return utf8.decode(bytes, allowMalformed: true).replaceAll('Ġ', ' ').trim();
  }

  @override
  Future<SttResult> transcribe(AudioBuffer audio, {String? language, CancellationToken? token}) async {
    final stopwatch = Stopwatch()..start();
    final nMels = this.nMels;
    final maxFrames = this.maxFrames;
    final mel = await Isolate.run(() {
      final ms = MelSpectrogram(nMels: nMels);
      return ms.compute(audio.samples);
    });
    final totalFrames = mel.length ~/ nMels;
    final fullText = StringBuffer();
    final langToken = _languageToken(language);
    final prompt = [sot, langToken, transcribeTok, noTimestamps];

    SttLogger.d('=== transcribe: audio=${audio.length}samples, melFrames=$totalFrames, langToken=$langToken ===');

    for (int offset = 0; offset < totalFrames && offset < maxFrames * 100; offset += maxFrames) {
      final chunk = transposeMel(mel, nMels, totalFrames, offset, maxFrames);

      final inputTensor = await ort.OrtValue.fromList(chunk, [1, nMels, maxFrames]);
      final encoderOut = await _encoderSession!.run({_encoderInputName: inputTensor});
      final rawStates = await encoderOut[_encoderOutputName]!.asFlattenedList();
      final encData = Float32List.fromList(rawStates.cast<num>().map((e) => e.toDouble()).toList());

      await inputTensor.dispose();
      for (final v in encoderOut.values) {
        await v.dispose();
      }

      token?.throwIfCancelled();
      SttLogger.d('  encoder: offset=$offset, output=${encData.length} floats');

      final encTensor = await ort.OrtValue.fromList(encData, [1, encoderFrames, _dModel]);
      final tokens = <int>[...prompt];

      for (int i = 0; i < maxTokens - prompt.length; i++) {
        token?.throwIfCancelled();
        final ids = Int64List.fromList(tokens.map((t) => t.toInt()).toList());
        final idTensor = await ort.OrtValue.fromList(ids, [1, tokens.length]);

        final decoderInputs = <String, ort.OrtValue>{
          _decoderInputIds: idTensor,
          _decoderInputStates: encTensor,
        };
        final extraTensors = <ort.OrtValue>[];
        Map<String, ort.OrtValue>? decoderOut;
        try {
          for (final extra in _decoderExtraInputs) {
            final eName = extra['name'] as String? ?? '';
            final extraShape = List<int>.from(extra['shape'] as List<dynamic>? ?? [1]);
            final extraType = (extra['type'] as String? ?? 'float32').toLowerCase();
            final fixedShape = extraShape.map((s) => s == -1 ? 1 : s).toList();
            final total = fixedShape.fold(1, (a, b) => a * b);

            ort.OrtValue tensor;
            if (extraType.contains('bool')) {
              tensor = await ort.OrtValue.fromList(List<bool>.filled(total, false), fixedShape);
            } else if (extraType.contains('int64')) {
              tensor = await ort.OrtValue.fromList(Int64List(total), fixedShape);
            } else if (extraType.contains('int32')) {
              tensor = await ort.OrtValue.fromList(Int32List(total), fixedShape);
            } else {
              tensor = await ort.OrtValue.fromList(Float32List(total), fixedShape);
            }
            extraTensors.add(tensor);
            decoderInputs[eName] = tensor;
          }

          decoderOut = await _decoderSession!.run(decoderInputs);
          final rawLogits = await decoderOut[_decoderOutputLogits]!.asFlattenedList();

          final vocabSize = rawLogits.length ~/ tokens.length;
          final lastStart = (tokens.length - 1) * vocabSize;
          final lastLogits = lastStart + vocabSize <= rawLogits.length
              ? rawLogits.sublist(lastStart, lastStart + vocabSize)
              : rawLogits;

          final next = argmax(lastLogits);

          if (next == eot) break;
          tokens.add(next);

          if (i < 20 || i % 50 == 0) {
            final partial = _decode(tokens.sublist(prompt.length));
            SttLogger.d('decoder step $i: token=$next, partial="$partial"');
          }
        } finally {
          await idTensor.dispose();
          if (decoderOut != null) {
            for (final v in decoderOut.values) {
              await v.dispose();
            }
          }
          for (final t in extraTensors) {
            await t.dispose();
          }
        }
      }

      await encTensor.dispose();

      final text = _decode(tokens.sublist(prompt.length));
      SttLogger.d('  chunk text: "$text" (${tokens.length - prompt.length} tokens)');
      if (text.isNotEmpty) {
        if (fullText.isNotEmpty) fullText.write(' ');
        fullText.write(text);
      }
    }

    stopwatch.stop();
    SttLogger.d('=== result: "${fullText.toString()}" in ${stopwatch.elapsedMilliseconds}ms ===');
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
