import 'package:path/path.dart' as p;
import 'package:stt_flutter/src/config/models.dart';
import 'package:stt_flutter/src/stt_logger.dart';

class ModelLoader {
  Map<String, dynamic> loadTransducerModel(SttModelConfig config, String modelDir) {
    final encoderPath = p.join(modelDir, config.encoderPath);
    final decoderPath = p.join(modelDir, config.decoderPath);
    final tokensPath = p.join(modelDir, config.tokensPath);

    final modelData = <String, dynamic>{
      'type': 'transducer',
      'modelId': config.id,
      'encoderPath': encoderPath,
      'decoderPath': decoderPath,
      'tokensPath': tokensPath,
    };

    if (config.joinerPath != null) {
      final joinerPath = p.join(modelDir, config.joinerPath!);
      modelData['joinerPath'] = joinerPath;
    }

    SttLogger.d('Loaded transducer model: ${config.id}');
    return modelData;
  }

  Map<String, dynamic> loadWhisperModel(SttModelConfig config, String modelDir) {
    final encoderPath = p.join(modelDir, config.encoderPath);
    final decoderPath = p.join(modelDir, config.decoderPath);
    final tokensPath = p.join(modelDir, config.tokensPath);

    final modelData = <String, dynamic>{
      'type': 'whisper',
      'modelId': config.id,
      'encoderPath': encoderPath,
      'decoderPath': decoderPath,
      'tokensPath': tokensPath,
    };

    SttLogger.d('Loaded Whisper model: ${config.id}');
    return modelData;
  }
}
