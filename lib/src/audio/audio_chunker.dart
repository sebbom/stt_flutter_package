import 'dart:typed_data';

import 'audio_buffer.dart';

/// Configuration for sliding-window chunking of long audio.
///
/// Each engine decides what to do with the [windowSeconds] / [overlapSeconds]
/// — see the per-engine defaults below.
class ChunkingConfig {
  /// Length of each chunk in seconds. Audio shorter than this is not chunked.
  final int windowSeconds;

  /// Overlap between adjacent chunks in seconds. Used to provide cross-chunk
  /// context for non-streaming models. Must be in `[0, windowSeconds)`.
  final int overlapSeconds;

  const ChunkingConfig({
    this.windowSeconds = 30,
    this.overlapSeconds = 2,
  });

  static const none = ChunkingConfig(
    windowSeconds: 30,
    overlapSeconds: 0,
  );

  static const defaultForTransducer = ChunkingConfig(
    windowSeconds: 30,
    overlapSeconds: 2,
  );

  static const whisper = ChunkingConfig(
    windowSeconds: 30,
    overlapSeconds: 5,
  );
}

/// Splits [audio] into overlapping windows. Returns a list with a single
/// element when the audio already fits in one window.
List<AudioBuffer> chunkBuffer(
  AudioBuffer audio, {
  ChunkingConfig config = ChunkingConfig.defaultForTransducer,
}) {
  if (audio.samples.isEmpty) return const [];
  if (config.overlapSeconds >= config.windowSeconds) {
    throw ArgumentError(
      'overlapSeconds (${config.overlapSeconds}) must be < windowSeconds '
      '(${config.windowSeconds})',
    );
  }
  final window = config.windowSeconds * audio.sampleRate;
  final overlap = config.overlapSeconds * audio.sampleRate;
  final step = window - overlap;

  if (audio.samples.length <= window) {
    return [audio];
  }

  final chunks = <AudioBuffer>[];
  for (var start = 0; start < audio.samples.length; start += step) {
    final end = (start + window).clamp(0, audio.samples.length);
    final slice = Float32List.sublistView(audio.samples, start, end);
    chunks.add(AudioBuffer(samples: slice, sampleRate: audio.sampleRate));
    if (end >= audio.samples.length) break;
  }
  return chunks;
}

/// Joins per-chunk transcriptions and removes the overlap-region duplicates
/// that occur at chunk boundaries.
///
/// The algorithm walks adjacent pairs, looks for the longest fuzzy suffix /
/// prefix match (up to [_overlapSearchWords] words), and strips the matching
/// prefix from the next chunk.
String dedupJoinedText(List<String> parts) {
  if (parts.length <= 1) return parts.join(' ').trim();
  final out = StringBuffer();
  String prevTail = '';
  for (var i = 0; i < parts.length; i++) {
    final p = parts[i].trim();
    if (p.isEmpty) continue;
    if (i == 0) {
      out.write(p);
    } else {
      final deduped = _stripOverlapPrefix(p, prevTail);
      if (deduped.isNotEmpty) {
        if (out.isNotEmpty) out.write(' ');
        out.write(deduped);
      }
    }
    prevTail = _tailWords(p, 8);
  }
  return out.toString();
}

const int _overlapSearchWords = 12;

String _stripOverlapPrefix(String current, String previousTail) {
  if (previousTail.isEmpty) return current;
  final prevWords = previousTail.split(RegExp(r'\s+'));
  final curWords = current.split(RegExp(r'\s+'));
  final maxN = prevWords.length < curWords.length
      ? prevWords.length
      : curWords.length;
  if (maxN > _overlapSearchWords) {
    // Limit search to the tail of prev / head of current to keep this O(N).
    final tail = prevWords.sublist(prevWords.length - _overlapSearchWords);
    final head = curWords.sublist(0, _overlapSearchWords);
    return _findAndStrip(tail, head, current);
  }
  return _findAndStrip(prevWords, curWords, current);
}

String _findAndStrip(
  List<String> prevWords,
  List<String> curWords,
  String current,
) {
  final maxN = prevWords.length < curWords.length
      ? prevWords.length
      : curWords.length;
  for (var n = maxN; n > 0; n--) {
    final tail = prevWords.sublist(prevWords.length - n).join(' ');
    final head = curWords.sublist(0, n).join(' ');
    if (_fuzzyEqual(tail, head)) {
      return curWords.sublist(n).join(' ');
    }
  }
  return current;
}

String _tailWords(String s, int n) {
  final words = s.split(RegExp(r'\s+'));
  if (words.length <= n) return s;
  return words.sublist(words.length - n).join(' ');
}

bool _fuzzyEqual(String a, String b) {
  if (a == b) return true;
  final la = a.toLowerCase();
  final lb = b.toLowerCase();
  if (la == lb) return true;
  final strippedA = la.replaceAll(RegExp(r'[^\w\s]'), '');
  final strippedB = lb.replaceAll(RegExp(r'[^\w\s]'), '');
  if (strippedA == strippedB) return true;
  if (strippedA.length > 4 && strippedB.length > 4) {
    return strippedA.substring(strippedA.length - 4) ==
        strippedB.substring(strippedB.length - 4);
  }
  return false;
}
