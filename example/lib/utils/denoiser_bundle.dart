import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

/// Resolves bundled denoiser asset paths to real on-disk file paths.
///
/// sherpa-onnx's `OfflineSpeechDenoiser` reads model files directly from the
/// filesystem, but Flutter ships assets inside the APK / IPA. We extract the
/// asset bytes to a temp file the first time it's requested and cache the
/// resulting path so subsequent calls are cheap.
class DenoiserBundle {
  static const String _gtcrnAsset = 'assets/denoisers/gtcrn/model.onnx';
  static const String _dpdfnetAsset = 'assets/denoisers/dpdfnet/model.onnx';

  static String? _gtcrnPath;
  static String? _dpdfnetPath;

  /// Extracts the bundled GTCRN denoiser model to a temp file and returns the
  /// path. Subsequent calls return the cached path.
  static Future<String> gtcrnModelFile() async {
    if (_gtcrnPath != null && File(_gtcrnPath!).existsSync()) {
      return _gtcrnPath!;
    }
    _gtcrnPath = await _extract(_gtcrnAsset, 'denoisers/gtcrn/model.onnx');
    return _gtcrnPath!;
  }

  /// Extracts the bundled DPDFNet denoiser model to a temp file.
  static Future<String> dpdfnetModelFile() async {
    if (_dpdfnetPath != null && File(_dpdfnetPath!).existsSync()) {
      return _dpdfnetPath!;
    }
    _dpdfnetPath =
        await _extract(_dpdfnetAsset, 'denoisers/dpdfnet/model.onnx');
    return _dpdfnetPath!;
  }

  /// Returns the directory containing the model file for the given type.
  /// `null` if the asset hasn't been extracted yet — call [gtcrnModelFile] or
  /// [dpdfnetModelFile] first.
  static String? dirFor(String? modelFile) {
    if (modelFile == null) return null;
    final i = modelFile.lastIndexOf('/');
    return i < 0 ? modelFile : modelFile.substring(0, i);
  }

  static Future<String> _extract(String assetPath, String relativePath) async {
    final bytes = await rootBundle.load(assetPath);
    final tempRoot = await getTemporaryDirectory();
    final outDir = Directory('${tempRoot.path}/stt_bundle');
    if (!await outDir.exists()) {
      await outDir.create(recursive: true);
    }
    final outFile = File('${outDir.path}/$relativePath');
    if (!await outFile.parent.exists()) {
      await outFile.parent.create(recursive: true);
    }
    await outFile.writeAsBytes(
      bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes),
      flush: true,
    );
    return outFile.path;
  }
}
