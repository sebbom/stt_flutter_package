import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:stt_flutter/src/audio/audio_buffer.dart';
import 'package:stt_flutter/src/audio/audio_chunker.dart';

AudioBuffer _buffer(int seconds, {int sampleRate = 16000}) {
  return AudioBuffer(
    samples: Float32List(seconds * sampleRate),
    sampleRate: sampleRate,
  );
}

void main() {
  group('chunkBuffer', () {
    test('returns single chunk for short audio', () {
      final chunks = chunkBuffer(_buffer(5));
      expect(chunks.length, 1);
      expect(chunks.first.samples.length, 5 * 16000);
    });

    test('returns single chunk when audio fits exactly in window', () {
      final chunks = chunkBuffer(_buffer(30));
      expect(chunks.length, 1);
    });

    test('chunks long audio into 30s/2s windows', () {
      // 65 seconds → chunk1[0..30], chunk2[28..58], chunk3[56..65].
      final chunks = chunkBuffer(_buffer(65));
      expect(chunks.length, 3);
      expect(chunks[0].samples.length, 30 * 16000);
      expect(chunks[1].samples.length, 30 * 16000);
      expect(chunks[2].samples.length, 9 * 16000);
    });

    test('chunks with no overlap when overlap=0', () {
      final chunks =
          chunkBuffer(_buffer(75), config: const ChunkingConfig(
        windowSeconds: 30,
        overlapSeconds: 0,
      ));
      expect(chunks.length, 3);
      expect(chunks[0].samples.length, 30 * 16000);
      expect(chunks[1].samples.length, 30 * 16000);
      expect(chunks[2].samples.length, 15 * 16000);
    });

    test('throws on overlap >= window', () {
      expect(
        () => chunkBuffer(
          _buffer(60),
          config: const ChunkingConfig(
            windowSeconds: 30,
            overlapSeconds: 30,
          ),
        ),
        throwsArgumentError,
      );
    });

    test('returns empty list for empty audio', () {
      final chunks = chunkBuffer(AudioBuffer(
        samples: Float32List(0),
        sampleRate: 16000,
      ));
      expect(chunks, isEmpty);
    });

    test('preserves sample rate', () {
      final chunks = chunkBuffer(AudioBuffer(
        samples: Float32List(8000),
        sampleRate: 8000,
      ));
      expect(chunks.first.sampleRate, 8000);
    });
  });

  group('dedupJoinedText', () {
    test('joins single part', () {
      expect(dedupJoinedText(['hello world']), 'hello world');
    });

    test('joins two non-overlapping parts with a space', () {
      expect(dedupJoinedText(['hello', 'world']), 'hello world');
    });

    test('strips common suffix/prefix across chunks', () {
      expect(
        dedupJoinedText([
          'the quick brown fox jumps',
          'fox jumps over the lazy dog',
        ]),
        'the quick brown fox jumps over the lazy dog',
      );
    });

    test('fuzzy match (case-insensitive + punctuation)', () {
      expect(
        dedupJoinedText([
          'Hello, World.',
          'world. How are you?',
        ]),
        'Hello, World. How are you?',
      );
    });

    test('handles empty parts in the list', () {
      expect(
        dedupJoinedText(['', 'hello', '', 'world', '']),
        'hello world',
      );
    });
  });
}
