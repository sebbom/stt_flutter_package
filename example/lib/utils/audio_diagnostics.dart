import 'dart:io';

class LinuxAudioSource {
  final int id;
  final String name;
  final String description;
  final bool isDefault;

  const LinuxAudioSource({
    required this.id,
    required this.name,
    required this.description,
    required this.isDefault,
  });

  @override
  String toString() =>
      '${isDefault ? '*' : ' '}$id $description${name.isNotEmpty && name != description ? ' ($name)' : ''}';
}

class LinuxAudioDiagnostic {
  final bool serverRunning;
  final List<LinuxAudioSource> sources;
  final LinuxAudioSource? defaultSource;
  final String? rawError;
  final String toolUsed;

  const LinuxAudioDiagnostic({
    required this.serverRunning,
    required this.sources,
    required this.defaultSource,
    required this.rawError,
    required this.toolUsed,
  });

  bool get hasSource => sources.isNotEmpty;
  bool get hasDefaultSource => defaultSource != null;

  String get setDefaultCommand {
    if (sources.isEmpty) return '';
    final src = defaultSource ?? sources.first;
    if (toolUsed == 'wpctl') return 'wpctl set-default ${src.id}';
    if (toolUsed == 'pactl') return 'pactl set-default-source ${src.name}';
    return '';
  }

  String get preciseMessage {
    final raw = rawError == null ? '' : '\nUnderlying error: $rawError';
    if (!serverRunning) {
      return 'No PipeWire or PulseAudio server is reachable on this session.\n'
          'Try: systemctl --user status pipewire pipewire-pulse wireplumber\n'
          'And: systemctl --user start pipewire pipewire-pulse wireplumber$raw';
    }
    if (!hasSource) {
      return 'Audio server is running, but no input source (microphone) is '
          'visible to it.\n'
          'Check that a mic is connected, unmuted, and exposed to the seat '
          '(wpctl status / pactl list sources short).$raw';
    }
    if (!hasDefaultSource) {
      final hint = setDefaultCommand.isEmpty ? '' : '\nFix: $setDefaultCommand';
      return 'Microphone detected but no default input source is configured.\n'
          'Detected: ${sources.join(', ')}$hint$raw';
    }
    return 'Audio server is up and a default source is configured '
        '(${defaultSource!.description}), but recording still failed.$raw';
  }
}

class LinuxAudioDiagnostics {
  static const Duration _timeout = Duration(seconds: 2);

  static Future<LinuxAudioDiagnostic> diagnose({Object? rawError}) async {
    final raw = rawError?.toString();
    if (!Platform.isLinux) {
      return LinuxAudioDiagnostic(
        serverRunning: false,
        sources: const [],
        defaultSource: null,
        rawError: raw,
        toolUsed: 'none',
      );
    }

    final wp = await _tryWpctl();
    if (wp != null) return _withError(wp, raw);
    final pa = await _tryPactl();
    if (pa != null) return _withError(pa, raw);

    return LinuxAudioDiagnostic(
      serverRunning: false,
      sources: const [],
      defaultSource: null,
      rawError: raw,
      toolUsed: 'none',
    );
  }

  static LinuxAudioDiagnostic _withError(
      LinuxAudioDiagnostic d, String? raw) {
    return LinuxAudioDiagnostic(
      serverRunning: d.serverRunning,
      sources: d.sources,
      defaultSource: d.defaultSource,
      rawError: raw,
      toolUsed: d.toolUsed,
    );
  }

  static Future<LinuxAudioDiagnostic?> _tryWpctl() async {
    try {
      final r = await Process.run('wpctl', ['status'])
          .timeout(_timeout);
      if (r.exitCode != 0) return null;
      final out = r.stdout is String ? r.stdout as String : '';
      if (out.isEmpty) return null;
      final sources = parseWpctlSources(out);
      return LinuxAudioDiagnostic(
        serverRunning: true,
        sources: sources,
        defaultSource: sources.firstWhere(
          (s) => s.isDefault,
          orElse: () => const LinuxAudioSource(
            id: -1,
            name: '',
            description: '',
            isDefault: false,
          ),
        ).id == -1
            ? null
            : sources.firstWhere((s) => s.isDefault),
        rawError: null,
        toolUsed: 'wpctl',
      );
    } catch (_) {
      return null;
    }
  }

  static List<LinuxAudioSource> parseWpctlSources(String status) {
    final lines = status.split('\n');
    final sources = <LinuxAudioSource>[];
    var inSources = false;
    final srcLineRe = RegExp(
      r'^\s*[\u2502|]\s*(\*?)\s*(\d+)\.\s+(.+?)(?:\s*\[vol:[^\]]*\])?\s*$',
    );
    for (final raw in lines) {
      final line = raw.replaceAll('└', ' ').replaceAll('├', ' ');
      if (line.contains('Sources:')) {
        inSources = true;
        continue;
      }
      if (inSources) {
        if (line.contains('Source endpoints:') ||
            line.contains('Video') ||
            line.contains('Settings') ||
            line.contains('Stream')) {
          inSources = false;
          continue;
        }
        final m = srcLineRe.firstMatch(raw);
        if (m != null) {
          final isDefault = m.group(1) == '*';
          final id = int.tryParse(m.group(2) ?? '') ?? -1;
          final desc = (m.group(3) ?? '').trim();
          if (id > 0 && desc.isNotEmpty) {
            sources.add(LinuxAudioSource(
              id: id,
              name: desc,
              description: desc,
              isDefault: isDefault,
            ));
          }
        }
      }
    }
    return sources;
  }

  static Future<LinuxAudioDiagnostic?> _tryPactl() async {
    try {
      final info = await Process.run('pactl', ['info'])
          .timeout(_timeout);
      if (info.exitCode != 0) return null;
      final out = info.stdout is String ? info.stdout as String : '';
      if (out.isEmpty || !out.contains('Server Name')) return null;

      final list = await Process.run('pactl', ['list', 'sources', 'short'])
          .timeout(_timeout);
      final def = await Process.run('pactl', ['get-default-source'])
          .timeout(_timeout);

      final sources = <LinuxAudioSource>[];
      if (list.exitCode == 0) {
        final lines = (list.stdout as String).split('\n');
        for (final l in lines) {
          final parts = l.split('\t');
          if (parts.length < 2) continue;
          final idx = int.tryParse(parts[0]);
          final name = parts[1];
          if (idx == null || name.isEmpty) continue;
          sources.add(LinuxAudioSource(
            id: idx,
            name: name,
            description: name,
            isDefault: false,
          ));
        }
      }

      final defaultName = def.exitCode == 0
          ? (def.stdout as String).trim()
          : '';
      LinuxAudioSource? defaultSource;
      if (defaultName.isNotEmpty) {
        for (final s in sources) {
          if (s.name == defaultName) {
            defaultSource = LinuxAudioSource(
              id: s.id,
              name: s.name,
              description: s.description,
              isDefault: true,
            );
            sources[sources.indexOf(s)] = defaultSource;
            break;
          }
        }
      }

      return LinuxAudioDiagnostic(
        serverRunning: true,
        sources: sources,
        defaultSource: defaultSource,
        rawError: null,
        toolUsed: 'pactl',
      );
    } catch (_) {
      return null;
    }
  }
}
