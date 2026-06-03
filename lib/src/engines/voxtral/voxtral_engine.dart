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

class VoxtralInferenceEngine implements InferenceEngine {
  final ort.OnnxRuntime _runtime;
  ort.OrtSession? _audioEncoderSession;
  ort.OrtSession? _decoderSession;
  Map<int, String>? _tokenizer;

  String _audioEncInput = 'input_features';
  String _audioEncOutput = 'last_hidden_state';
  String _decInputIds = 'input_ids';
  String _decInputStates = 'encoder_hidden_states';
  String _decOutputLogits = 'logits';
  final List<Map<String, dynamic>> _decoderExtraInputs = [];

  int _dModel = 1024;
  int _vocabSize = 40000;
  int _maxSourcePositions = 1500;
  int _maxTargetPositions = 512;
  int _numMelBins = 80;

  static const int bos = 1;
  static const int eos = 2;

  VoxtralInferenceEngine(this._runtime);

  String _findFile(Map<String, String> files, List<String> patterns) {
    for (final p in patterns) {
      for (final entry in files.entries) {
        if (entry.key.contains(p)) return entry.value;
      }
    }
    throw FileSystemException('Model file not found for patterns: $patterns');
  }

  @override
  Future<void> load(Map<String, String> modelFiles) async {
    // Load config
    final cfgPath = modelFiles.entries
        .firstWhere((e) => e.key == 'config.json',
            orElse: () => const MapEntry('', ''))
        .value;
    if (cfgPath.isNotEmpty) {
      try {
        final cfg = jsonDecode(await File(cfgPath).readAsString());
        _dModel = cfg['d_model'] as int? ?? _dModel;
        _vocabSize = cfg['vocab_size'] as int? ?? _vocabSize;
        _maxSourcePositions = cfg['max_source_positions'] as int? ?? _maxSourcePositions;
        _maxTargetPositions = cfg['max_target_positions'] as int? ?? _maxTargetPositions;
        _numMelBins = cfg['num_mel_bins'] as int? ?? _numMelBins;
        SttLogger.d('=> Voxtral config: d_model=$_dModel, vocab=$_vocabSize, '
            'max_src=$_maxSourcePositions, max_tgt=$_maxTargetPositions, mels=$_numMelBins');
      } catch (e) {
        SttLogger.d('=> config load FAILED: $e');
      }
    }

    // Create sessions (discover files by pattern for quantized variants)
    try {
      _audioEncoderSession = await _runtime.createSession(
        _findFile(modelFiles, ['audio_encoder']),
      );
      _decoderSession = await _runtime.createSession(
        _findFile(modelFiles, ['decoder_model_merged']),
      );
    } catch (e) {
      final msg = e.toString().toLowerCase();
      if (msg.contains('deserialize') || msg.contains('int4') || msg.contains('q4f')) {
        throw Exception(
          'Voxtral q4f16 uses INT4 quantization unsupported by mobile ONNX Runtime. '
          'FP16 model variants are required for on-device inference.',
        );
      }
      rethrow;
    }

    // Detect encoder input/output names
    final encInputs = await _audioEncoderSession!.getInputInfo();
    SttLogger.d('=== Voxtral AudioEncoder inputs ===');
    for (final info in encInputs) {
      SttLogger.d('  $info');
    }
    if (encInputs.isNotEmpty) {
      _audioEncInput = (encInputs.first['name'] as String?) ?? _audioEncInput;
    }
    final encOutputs = await _audioEncoderSession!.getOutputInfo();
    SttLogger.d('=== Voxtral AudioEncoder outputs ===');
    for (final info in encOutputs) {
      SttLogger.d('  $info');
    }
    if (encOutputs.isNotEmpty) {
      _audioEncOutput = (encOutputs.first['name'] as String?) ?? _audioEncOutput;
      final shape = encOutputs.first['shape'] as List<dynamic>?;
      if (shape != null && shape.length >= 3) {
        _dModel = (shape[2] as num).toInt();
        SttLogger.d('  => d_model = $_dModel');
      }
    }

    // Detect decoder input/output names
    final decInputs = await _decoderSession!.getInputInfo();
    SttLogger.d('=== Voxtral Decoder inputs ===');
    for (final info in decInputs) {
      SttLogger.d('  $info');
    }
    for (final info in decInputs) {
      final name = (info['name'] as String?) ?? '';
      if (name.contains('input_ids')) {
        _decInputIds = name;
      } else if (name.contains('encoder_hidden_states') || name.contains('encoder_attn')) {
        _decInputStates = name;
      } else {
        _decoderExtraInputs.add(info);
      }
    }
    final decOutputs = await _decoderSession!.getOutputInfo();
    SttLogger.d('=== Voxtral Decoder outputs ===');
    for (final info in decOutputs) {
      SttLogger.d('  $info');
    }
    if (decOutputs.isNotEmpty) {
      _decOutputLogits = (decOutputs.first['name'] as String?) ?? _decOutputLogits;
    }

    // Load tokenizer
    final tokPath = modelFiles.entries
        .firstWhere((e) => e.key.contains('tokenizer.json'),
            orElse: () => const MapEntry('', ''))
        .value;
    if (tokPath.isNotEmpty) {
      try {
        final raw = jsonDecode(await File(tokPath).readAsString());
        _tokenizer = {};
        // HuggingFace tokenizer.json has `added_tokens` and `model/vocab` sections
        final modelData = raw['model'] as Map<String, dynamic>?;
        if (modelData != null) {
          final vocab = modelData['vocab'] as Map<String, dynamic>?;
          if (vocab != null) {
            for (final entry in vocab.entries) {
              _tokenizer![(entry.value as num).toInt()] = entry.key;
            }
          }
        }
        // Also add added_tokens
        final addedTokens = raw['added_tokens'] as List<dynamic>?;
        if (addedTokens != null) {
          for (final t in addedTokens) {
            final id = (t['id'] as num).toInt();
            final content = t['content'] as String?;
            if (content != null) {
              _tokenizer![id] = content;
            }
          }
        }
        SttLogger.d('  => tokenizer loaded: ${_tokenizer!.length} entries');
      } catch (e) {
        SttLogger.d('  => tokenizer load FAILED: $e');
      }
    }
  }

  String _decode(List<int> ids) {
    if (_tokenizer == null) {
      return ids.map((i) => i.toString()).join(' ');
    }
    final texts = <String>[];
    for (final id in ids) {
      final s = _tokenizer![id];
      if (s != null && !s.startsWith('<')) {
        texts.add(s);
      }
    }
    return texts.join('').replaceAll('Ġ', ' ').trim();
  }

  @override
  Future<SttResult> transcribe(AudioBuffer audio, {String? language, CancellationToken? token}) async {
    final stopwatch = Stopwatch()..start();

    // 1. Compute mel spectrogram
    final mel = await Isolate.run(() {
      final ms = MelSpectrogram(nMels: _numMelBins);
      return ms.compute(audio.samples);
    });
    final totalFrames = mel.length ~/ _numMelBins;
    SttLogger.d('=== Voxtral transcribe: audio=${audio.length}samples, melFrames=$totalFrames ===');

    final fullText = StringBuffer();
    final fullMelFrames = totalFrames < _maxSourcePositions ? totalFrames : _maxSourcePositions;

    // 2. Transpose and pad to max frames
    final chunk = transposeMel(mel, _numMelBins, totalFrames, 0, fullMelFrames);
    final melPadded = Float32List(_numMelBins * _maxSourcePositions);
    melPadded.setRange(0, chunk.length, chunk);

    // 3. Run audio encoder
    final inputTensor = await ort.OrtValue.fromList(melPadded, [1, _numMelBins, _maxSourcePositions]);
    final encoderOut = await _audioEncoderSession!.run({_audioEncInput: inputTensor});
    final rawStates = await encoderOut[_audioEncOutput]!.asFlattenedList();
    await inputTensor.dispose();
    for (final v in encoderOut.values) {
      await v.dispose();
    }

    SttLogger.d('  encoder: output=${rawStates.length} floats');

    final encSeqLen = rawStates.length ~/ _dModel;
    final encData = Float32List.fromList(rawStates.cast<num>().map((e) => e.toDouble()).toList());
    final encTensor = await ort.OrtValue.fromList(encData, [1, encSeqLen, _dModel]);

    // 4. Autoregressive decoder loop
    final tokens = <int>[bos];
    final maxNewTokens = _maxTargetPositions < 200 ? _maxTargetPositions : 200;

    for (int i = 0; i < maxNewTokens; i++) {
      token?.throwIfCancelled();
      final ids = Int64List.fromList(tokens.map((t) => t.toInt()).toList());
      final idTensor = await ort.OrtValue.fromList(ids, [1, tokens.length]);

      final decoderInputs = <String, ort.OrtValue>{
        _decInputIds: idTensor,
        _decInputStates: encTensor,
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
          } else {
            tensor = await ort.OrtValue.fromList(Float32List(total), fixedShape);
          }
          extraTensors.add(tensor);
          decoderInputs[eName] = tensor;
        }

        decoderOut = await _decoderSession!.run(decoderInputs);
        final rawLogits = await decoderOut[_decOutputLogits]!.asFlattenedList();

        final vocabSize = rawLogits.length ~/ tokens.length;
        final lastStart = (tokens.length - 1) * vocabSize;
        final lastLogits = lastStart + vocabSize <= rawLogits.length
            ? rawLogits.sublist(lastStart, lastStart + vocabSize)
            : rawLogits;

        final best = argmax(lastLogits);

        if (best == eos) break;
        tokens.add(best);

        if (i < 20 || i % 50 == 0) {
          final partial = _decode(tokens.sublist(1));
          SttLogger.d('decoder step $i: token=$best, partial="$partial"');
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

    final text = _decode(tokens.sublist(1));
    if (text.isNotEmpty) fullText.write(text);
    stopwatch.stop();
    SttLogger.d('=== voxtral result: "$text" (${tokens.length - 1} tokens) in ${stopwatch.elapsedMilliseconds}ms ===');

    return SttResult(
      text: fullText.toString(),
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000,
    );
  }

  @override
  Future<void> dispose() async {
    await _audioEncoderSession?.close();
    await _decoderSession?.close();
  }
}
