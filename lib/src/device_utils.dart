import 'dart:io' show Platform;
import 'package:device_info_plus/device_info_plus.dart';

class DeviceCapabilities {
  final int memoryMB;
  final int cpuCores;
  final bool isHighEnd;
  final bool supportsNPU;

  const DeviceCapabilities({
    required this.memoryMB,
    required this.cpuCores,
    required this.isHighEnd,
    required this.supportsNPU,
  });
}

class DeviceUtils {
  static Future<DeviceCapabilities> detectCapabilities() async {
    final deviceInfo = DeviceInfoPlugin();

    if (Platform.isAndroid) {
      final androidInfo = await deviceInfo.androidInfo;
      return _detectAndroidCapabilities(androidInfo);
    } else if (Platform.isIOS) {
      final iosInfo = await deviceInfo.iosInfo;
      return _detectIOSCapabilities(iosInfo);
    }

    return const DeviceCapabilities(
      memoryMB: 4000,
      cpuCores: 4,
      isHighEnd: true,
      supportsNPU: false,
    );
  }

  static DeviceCapabilities _detectAndroidCapabilities(AndroidDeviceInfo info) {
    final model = info.model.toLowerCase();
    final sdkVersion = info.version.sdkInt;

    final highEndModels = [
      'samsung sm-g', 'pixel', 'oneplus', 'xiaomi 13', 'xiaomi 14',
      's23', 's24', 'note 20', 'note 23', 'fold', 'z fold',
    ];

    final isHighEnd = highEndModels.any((m) => model.contains(m)) ||
        sdkVersion >= 34;

    int memoryMB = 4000;
    if (model.contains('s23') || model.contains('s24')) {
      memoryMB = 8192;
    } else if (model.contains('pixel 8') || model.contains('oneplus 12')) {
      memoryMB = 12288;
    } else if (model.contains('pixel 7') || model.contains('s22')) {
      memoryMB = 6144;
    }

    final supportsNPU = sdkVersion >= 31 && (
      model.contains('samsung') ||
      model.contains('qualcomm') ||
      model.contains('mediatek')
    );

    return DeviceCapabilities(
      memoryMB: memoryMB,
      cpuCores: 8,
      isHighEnd: isHighEnd,
      supportsNPU: supportsNPU,
    );
  }

  static DeviceCapabilities _detectIOSCapabilities(IosDeviceInfo info) {
    final model = info.model;
    final systemVersion = info.systemVersion;
    final iosVersion = double.tryParse(systemVersion) ?? 15.0;

    final highEndModels = ['iPhone15', 'iPhone14', 'iPhone13', 'iPad Pro'];
    final isHighEnd = highEndModels.any((m) => model.contains(m)) ||
        iosVersion >= 16.0;

    int memoryMB = 4096;
    if (model.contains('iPhone15Pro') || model.contains('iPad Pro')) {
      memoryMB = iosVersion >= 17.0 ? 8192 : 6144;
    } else if (model.contains('iPhone15') || model.contains('iPhone14')) {
      memoryMB = 6144;
    } else if (model.contains('iPhone13')) {
      memoryMB = 4096;
    }

    return DeviceCapabilities(
      memoryMB: memoryMB,
      cpuCores: 6,
      isHighEnd: isHighEnd,
      supportsNPU: true,
    );
  }
}
