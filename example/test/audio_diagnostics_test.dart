import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter_example/utils/audio_diagnostics.dart';

void main() {
  group('parseWpctlSources', () {
    test('parses a Sources section with one default source', () {
      const status = '''
Audio
 ├─ Sinks:
 │  *   57. Family 17h/19h HD Audio Controller Stéréo analogique [vol: 0.94]
 │
 ├─ Sources:
 │  *   58. Family 17h/19h HD Audio Controller Stéréo analogique [vol: 0.27]
 │
 ├─ Source endpoints:
 │
 └─ Streams:
''';
      final sources = LinuxAudioDiagnostics.parseWpctlSources(status);
      expect(sources, hasLength(1));
      expect(sources.first.id, 58);
      expect(sources.first.isDefault, isTrue);
      expect(sources.first.description,
          'Family 17h/19h HD Audio Controller Stéréo analogique');
    });

    test('parses multiple sources and only marks the default one', () {
      const status = '''
 ├─ Sources:
 │      42. USB Microphone                          [vol: 0.50]
 │  *   58. Internal Mic                            [vol: 0.27]
 │
 ├─ Source endpoints:
''';
      final sources = LinuxAudioDiagnostics.parseWpctlSources(status);
      expect(sources, hasLength(2));
      expect(sources[0].id, 42);
      expect(sources[0].isDefault, isFalse);
      expect(sources[1].id, 58);
      expect(sources[1].isDefault, isTrue);
    });

    test('returns empty list when there is no Sources section', () {
      const status = '''
Audio
 ├─ Sinks:
 │  *   57. Some Sink [vol: 0.94]
''';
      expect(LinuxAudioDiagnostics.parseWpctlSources(status), isEmpty);
    });

    test('accepts ASCII pipe fallback (|) instead of Unicode │', () {
      const status = '''
 | Sources:
 |  *   58. ASCII-pipe source [vol: 0.27]
''';
      final sources = LinuxAudioDiagnostics.parseWpctlSources(status);
      expect(sources, hasLength(1));
      expect(sources.first.id, 58);
      expect(sources.first.isDefault, isTrue);
      expect(sources.first.description, 'ASCII-pipe source');
    });
  });

  group('LinuxAudioDiagnostic.preciseMessage', () {
    test('says server is unreachable when nothing is running', () {
      const d = LinuxAudioDiagnostic(
        serverRunning: false,
        sources: [],
        defaultSource: null,
        rawError: 'spawn failed',
        toolUsed: 'none',
      );
      expect(d.preciseMessage, contains('No PipeWire or PulseAudio server'));
      expect(d.preciseMessage, contains('spawn failed'));
    });

    test('flags missing input source when only sinks exist', () {
      const d = LinuxAudioDiagnostic(
        serverRunning: true,
        sources: [],
        defaultSource: null,
        rawError: null,
        toolUsed: 'wpctl',
      );
      expect(d.preciseMessage, contains('no input source'));
    });

    test('suggests a set-default command when source present but no default',
        () {
      const sources = [
        LinuxAudioSource(
          id: 58,
          name: 'alsa_input.pci-0000_07_00.6.analog-stereo',
          description: 'Internal Mic',
          isDefault: false,
        ),
      ];
      const d = LinuxAudioDiagnostic(
        serverRunning: true,
        sources: sources,
        defaultSource: null,
        rawError: null,
        toolUsed: 'wpctl',
      );
      expect(d.preciseMessage, contains('no default input source'));
      expect(d.preciseMessage, contains('wpctl set-default 58'));
    });

    test('reports the underlying error when default source is set', () {
      const src = LinuxAudioSource(
        id: 58,
        name: 'Internal Mic',
        description: 'Internal Mic',
        isDefault: true,
      );
      const d = LinuxAudioDiagnostic(
        serverRunning: true,
        sources: [src],
        defaultSource: src,
        rawError: 'BUSY',
        toolUsed: 'wpctl',
      );
      expect(d.preciseMessage, contains('Internal Mic'));
      expect(d.preciseMessage, contains('BUSY'));
    });

    test('uses pactl syntax for set-default-source under pactl', () {
      const sources = [
        LinuxAudioSource(
          id: 1,
          name: 'alsa_input.pci-0000_07_00.6.analog-stereo',
          description: 'alsa_input.pci-0000_07_00.6.analog-stereo',
          isDefault: false,
        ),
      ];
      const d = LinuxAudioDiagnostic(
        serverRunning: true,
        sources: sources,
        defaultSource: null,
        rawError: null,
        toolUsed: 'pactl',
      );
      expect(d.preciseMessage,
          contains('pactl set-default-source alsa_input.pci-0000_07_00.6.analog-stereo'));
    });
  });
}
