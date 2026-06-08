import 'package:flutter/foundation.dart' show debugPrint;

enum LogLevel { verbose, debug, info, warning, error, none }

class SttLogger {
  SttLogger._();

  static LogLevel _level = LogLevel.info;
  static String _tag = 'SttFlutter';
  static bool _showCallerInfo = false;

  static void setLevel(LogLevel level) => _level = level;

  static void setTag(String tag) => _tag = tag;

  static void setShowCallerInfo(bool show) => _showCallerInfo = show;

  static void v(String message) => _log(LogLevel.verbose, message);
  static void d(String message) => _log(LogLevel.debug, message);
  static void i(String message) => _log(LogLevel.info, message);
  static void w(String message) => _log(LogLevel.warning, message);
  static void e(String message, [dynamic error, StackTrace? stack]) =>
      _log(LogLevel.error, message, error, stack);

  static void _log(LogLevel msgLevel, String message,
      [dynamic error, StackTrace? stack]) {
    if (msgLevel.index < _level.index) return;

    final prefix = _formatLevel(msgLevel);
    final tag = _tag;
    final caller = _showCallerInfo ? ' [${StackTrace.current.toString().split('\n').elementAt(2).trim()}]' : '';
    final errorStr = error != null ? ' | $error' : '';

    debugPrint('$prefix $tag: $message$caller$errorStr');
  }

  static String _formatLevel(LogLevel level) {
    switch (level) {
      case LogLevel.verbose:
        return '[VERBOSE]';
      case LogLevel.debug:
        return '[DEBUG]';
      case LogLevel.info:
        return '[INFO]';
      case LogLevel.warning:
        return '[WARN]';
      case LogLevel.error:
        return '[ERROR]';
      case LogLevel.none:
        return '';
    }
  }
}
