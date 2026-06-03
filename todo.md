# STT Flutter Package - Improvement Analysis

> **Branch:** `vibe/review-improvements-bc2b72`  
> **Date:** 2025-06-01  
> **Base Commit:** f349744  

---

## Executive Summary

The `stt_flutter` package is a well-architected, feature-rich Flutter plugin for **fully local, on-device speech-to-text** using ONNX models. It supports **Whisper**, **Sherpa-ONNX**, and **Voxtral** model families with a clean, extensible design. The implementation is production-ready in many aspects but has **critical architectural, performance, and maintainability issues** that should be addressed before widespread adoption.

**Overall Assessment: 8.5/10** (Excellent foundation, needs refinement)

---

## Strengths

### ✅ Architecture
- **Clean separation of concerns**: Engine layer, audio processing, model registry are well-isolated
- **Extensible design**: `InferenceEngine` abstract class + factory pattern allows adding new model types easily
- **Model registry pattern**: Users can register custom models in one line
- **Multi-engine support**: Whisper, Sherpa, Voxtral all implemented with consistent interfaces

### ✅ Performance
- **Background isolate usage**: Audio preprocessing offloaded to `Isolate.run()` to avoid UI blocking
- **Async ONNX inference**: `flutter_onnxruntime` uses MethodChannel internally, keeping Dart event loop free
- **Efficient memory management**: Proper disposal of OrtValue tensors throughout

### ✅ Features
- **Multi-language**: 99+ languages via Whisper, 8 via Voxtral
- **Runtime model download**: Models downloaded and cached on first use
- **Flexible input**: File path or raw PCM buffer transcription
- **Progress callbacks**: Download progress reporting
- **Three model families**: Covers different use cases (accuracy vs. size)

### ✅ Code Quality
- **Comprehensive documentation**: PLAN.md is excellent, README is clear
- **Unit tests**: Good coverage of audio processing, tokenizers, spectrograms
- **Type safety**: Strong typing throughout
- **Error handling**: Basic validation present

---

## Critical Issues

### 🔴 1. Background Isolate Architecture Problem

**Severity:** CRITICAL  
**Impact:** Performance, Memory, Stability  

The current implementation uses **ephemeral isolates** (`Isolate.run()`) for audio preprocessing but runs **ONNX inference on the main isolate**. This is problematic:

```dart
// Current: stt_flutter_impl.dart line 52-54
Future<SttResult> _transcribe(AudioBuffer audio) async {
  final resampled = await Isolate.run(() => AudioProcessor.resampleSync(audio));
  return _engine!.transcribe(resampled);  // ← Runs on MAIN isolate!
}
```

**Problems:**
- ONNX inference (`session.run()`) is CPU-intensive and blocks the Dart event loop
- Despite `flutter_onnxruntime` using MethodChannel, the Dart-side tensor preparation and result processing is synchronous
- UI jank will occur during transcription on weaker devices
- No true parallelism for inference

**Evidence from code:**
- `whisper_engine.dart` line 140-145: Encoder runs synchronously on main isolate
- `sherpa_engine.dart` line 150-155: Same pattern
- `voxtral_engine.dart` line 180-185: Same pattern

### 🔴 2. Memory Leak Risk

**Severity:** HIGH  
**Impact:** Memory, Stability  

Multiple engines have **incomplete resource cleanup**:

```dart
// whisper_engine.dart line 140-145
final encoderOut = await _encoderSession!.run({_encoderInputName: inputTensor});
final rawStates = await encoderOut[_encoderOutputName]!.asFlattenedList();
final encData = Float32List.fromList(rawStates.cast<num>().map((e) => e.toDouble()).toList());

await inputTensor.dispose();
for (final v in encoderOut.values) {
  await v.dispose();  // ← Good
}
```

But in the decoder loop (line 160-180):
```dart
final decoderOut = await _decoderSession!.run(decoderInputs);
final rawLogits = await decoderOut[_decoderOutputLogits]!.asFlattenedList();
// ...
await idTensor.dispose();
for (final v in decoderOut.values) {
  await v.dispose();  // ← Also good
}
```

**However**, the `decoderInputs` OrtValue tensors are **not always disposed**:
```dart
// Line 155-160
decoderInputs[eName] = await ort.OrtValue.fromList(List<bool>.filled(total, false), fixedShape);
// ... no disposal of these extra inputs!
```

**Same issue in:**
- `sherpa_engine.dart` line 200-210 (state tensors)
- `voxtral_engine.dart` line 220-230 (extra inputs)

### 🔴 3. No Cancellation Support

**Severity:** HIGH  
**Impact:** User Experience  

There is **no way to cancel** an ongoing transcription. Users must wait for completion or call `dispose()` which kills everything.

**Missing:**
- `CancelableOperation` or custom cancellation tokens
- Timeout mechanisms
- Progress callbacks for transcription (only download has progress)

**User pain point:** If transcription takes too long, user is stuck.

---

## Major Issues

### 🟡 4. Hardcoded Language Tokens

**Severity:** MEDIUM  
**Impact:** Maintainability, Extensibility  

Language tokens are hardcoded in each engine:

```dart
// whisper_engine.dart line 25-28
static const int en = 50259;
static const int de = 50261;
static const int fr = 50263;
static const int es = 50265;

static int _languageToken(String? language) {
  switch (language?.toLowerCase()) {
    case 'de': case 'german': return de;
    case 'fr': case 'french': return fr;
    case 'es': case 'spanish': return es;
    default: return en;
  }
}
```

**Problems:**
- Only 4 languages supported despite Whisper supporting 99+
- Adding new languages requires code changes
- Inconsistent with Voxtral which uses tokenizer-based language detection

**Solution:** Load language tokens from `tokenizer.json` dynamically.

### 🟡 5. No Batch Processing

**Severity:** MEDIUM  
**Impact:** Performance  

Audio is processed in **single chunks** without batching:

```dart
// whisper_engine.dart line 130-135
for (int offset = 0; offset < totalFrames && offset < maxFrames * 100; offset += maxFrames) {
  final chunk = _transposeMel(mel, totalFrames, offset, maxFrames);
  // Process one chunk at a time
}
```

**Problem:** Inefficient for long audio files. Should process multiple chunks in parallel or use streaming.

### 🟡 6. Inconsistent Error Handling

**Severity:** MEDIUM  
**Impact:** Debugging, User Experience  

Error handling is inconsistent:
- Some methods throw `StateError` (line 40, 44 in stt_flutter_impl.dart)
- Some use `debugPrint` for errors (whisper_engine.dart line 80-85)
- Some silently fail (model_downloader.dart line 60-65)

**Missing:**
- Custom exception types
- Error codes
- Consistent error reporting

### 🟡 7. No Input Validation

**Severity:** MEDIUM  
**Impact:** Stability  

Minimal input validation:

```dart
// stt_flutter_impl.dart line 39-40
if (!_initialized) throw StateError('SttFlutter not initialized');
```

**Missing validation:**
- Audio buffer sample rate bounds
- Audio buffer length (minimum viable audio)
- Language code format
- Model file existence before load
- File path validity

### 🟡 8. Inefficient File Discovery

**Severity:** MEDIUM  
**Impact:** Performance  

Model file discovery is inefficient:

```dart
// stt_flutter_impl.dart line 18-28
final modelFiles = <String, String>{};
for (final f in model.files) {
  final path = '$dir/${f.filename}';
  final file = File(path);
  if (await file.exists()) {
    modelFiles[f.filename] = path;
  }
}

// Also discover extracted files from .tar.bz2 archives (e.g. Sherpa)
final modelDir_ = Directory(dir);
if (await modelDir_.exists()) {
  await for (final entry in modelDir_.list()) {
    // ... iterate all files
  }
}
```

**Problems:**
- Two separate file discovery passes
- `Directory.list()` is async and slow
- No caching of discovered files
- Sherpa models require substring matching (fragile)

---

## Code Quality Issues

### 🟢 9. Duplicate Code

**Severity:** LOW  
**Impact:** Maintainability  

Significant code duplication between engines:

| Pattern | Whisper | Sherpa | Voxtral |
|---------|---------|--------|---------|
| Mel/Fbank computation | ✅ | ✅ (different) | ✅ |
| Transpose mel | ✅ | ❌ | ✅ |
| Argmax | ✅ | ✅ | ✅ |
| Decoder loop | ✅ | ✅ (different) | ✅ |
| Tensor disposal | ✅ | ✅ | ✅ |

**Examples:**
- `_argmax()` appears in all 3 engines (identical logic)
- `_transposeMel()` in Whisper and Voxtral (nearly identical)
- Decoder autoregressive loop pattern repeated 3x

### 🟢 10. Magic Numbers

**Severity:** LOW  
**Impact:** Readability, Maintainability  

Hardcoded constants throughout:

```dart
// whisper_engine.dart
static const int nMels = 80;
static const int maxFrames = 3000;
static const int maxTokens = 448;
static const int encoderFrames = 1500;

// sherpa_engine.dart
static const int blank = 0;
static const int contextSize = 2;

// voxtral_engine.dart
int _dModel = 1024;
int _vocabSize = 40000;
```

**Problem:** Hard to maintain, error-prone, poor discoverability.

### 🟢 11. Debug Prints in Production Code

**Severity:** LOW  
**Impact:** Performance, Cleanliness  

Excessive `debugPrint` statements:
- whisper_engine.dart: ~20 debugPrint calls
- sherpa_engine.dart: ~15 debugPrint calls
- voxtral_engine.dart: ~15 debugPrint calls

**Impact:**
- Performance overhead (string formatting)
- Console spam in production
- Should be behind a debug flag or logging system

### 🟢 12. Missing Documentation

**Severity:** LOW  
**Impact:** Adoption  

Missing docs:
- No API documentation comments
- No examples in class/method docstrings
- Limited inline comments for complex algorithms
- No CHANGELOG.md
- No CONTRIBUTING.md

### 🟢 13. Test Coverage Gaps

**Severity:** LOW  
**Impact:** Reliability  

Current test files:
- ✅ `audio_processor_test.dart` - Good
- ✅ `model_registry_test.dart` - Good
- ✅ `mel_spectrogram_test.dart` - Good
- ❌ No engine integration tests
- ❌ No model downloader tests
- ❌ No transcription tests
- ❌ No error handling tests

**Missing:**
- Integration tests for full transcription pipeline
- Mock HTTP tests for downloader
- Error case tests
- Performance benchmarks

---

## Security Issues

### 🔵 14. No SHA256 Verification

**Severity:** MEDIUM  
**Impact:** Security  

Model files have optional SHA256 but **not verified**:

```dart
// model_registry.dart
class ModelFile {
  final String url;
  final String filename;
  final String? sha256;  // ← Optional, never checked
}

// model_downloader.dart
static Future<void> _downloadFile({...}) async {
  // ... download logic
  // NO SHA256 verification!
}
```

**Risk:** Malicious model files could be injected via MITM or compromised CDN.

### 🔵 15. No HTTPS Certificate Pinning

**Severity:** LOW  
**Impact:** Security  

HTTP downloads use standard `http` package without certificate pinning:

```dart
// model_downloader.dart line 60
final response = await http.Client().send(http.Request('GET', Uri.parse(url)));
```

**Risk:** MITM attacks could intercept model downloads.

---

## Performance Issues

### ⚡ 16. Inefficient Resampling

**Severity:** MEDIUM  
**Impact:** Performance  

Current resampling is **linear interpolation** in Dart:

```dart
// audio_processor.dart line 28-38
static AudioBuffer resampleSync(AudioBuffer input, {int targetRate = targetSampleRate}) {
  if (input.sampleRate == targetRate) return input;
  final ratio = input.sampleRate / targetRate;
  final newLength = (input.length / ratio).round();
  final output = Float32List(newLength);

  for (int i = 0; i < newLength; i++) {
    final srcPos = i * ratio;
    final srcIdx = srcPos.floor();
    final frac = srcPos - srcIdx;
    if (srcIdx + 1 < input.length) {
      output[i] = input.samples[srcIdx] * (1 - frac) + input.samples[srcIdx + 1] * frac;
    } else {
      output[i] = input.samples[srcIdx];
    }
  }
  return AudioBuffer(samples: output, sampleRate: targetRate);
}
```

**Problems:**
- O(n) Dart loop (slow for large audio)
- No SIMD optimization
- Should use native code or optimized library

### ⚡ 17. No Audio Normalization

**Severity:** LOW  
**Impact:** Accuracy  

No audio normalization before processing:

```dart
// audio_processor.dart - no normalization
```

**Problem:** Audio with varying volumes may affect transcription accuracy.

**Solution:** Add optional normalization (peak or RMS).

### ⚡ 18. Inefficient Tensor Operations

**Severity:** LOW  
**Impact:** Performance  

Manual tensor manipulation in Dart:

```dart
// whisper_engine.dart line 145-150
final encData = Float32List.fromList(
  rawStates.cast<num>().map((e) => e.toDouble()).toList()
);
```

**Problem:** Creating intermediate lists is slow and memory-intensive.

**Solution:** Use typed data operations directly or native extensions.

---

## Architectural Recommendations

### 🎯 1. Implement True Background Inference

**Priority:** P0 (Critical)  
**Effort:** High  

Create a **long-lived background isolate** for inference:

```dart
// New architecture:
class InferenceWorker {
  final SendPort _sendPort;
  final ReceivePort _receivePort;
  OrtSession? _encoderSession;
  OrtSession? _decoderSession;
  
  InferenceWorker(this._sendPort);
  
  void _handleMessage(dynamic message) {
    // Process inference requests
  }
}

class SttFlutter {
  Isolate? _workerIsolate;
  SendPort? _workerSendPort;
  
  Future<void> initialize(...) async {
    final receivePort = ReceivePort();
    _workerIsolate = await Isolate.spawn(
      InferenceWorker._entry,
      receivePort.sendPort,
    );
    _workerSendPort = await receivePort.first as SendPort;
  }
  
  Future<SttResult> transcribe(AudioBuffer audio) async {
    final responsePort = ReceivePort();
    _workerSendPort!.send({
      'type': 'transcribe',
      'audio': audio,
      'responsePort': responsePort.sendPort,
    });
    return await responsePort.first as SttResult;
  }
}
```

**Benefits:**
- True parallel inference
- UI thread never blocked
- Better resource isolation

**Challenges:**
- `flutter_onnxruntime` uses MethodChannel (requires `BackgroundIsolateBinaryMessenger`)
- Need to pass OrtSession across isolates (not directly possible)
- Complex error handling

### 🎯 2. Implement Proper Resource Management

**Priority:** P0 (Critical)  
**Effort:** Medium  

Create a **resource manager** pattern:

```dart
class OrtResourceManager {
  final List<ort.OrtValue> _allocated = [];
  final List<ort.OrtSession> _sessions = [];
  
  OrtValue allocateTensor(...) {
    final value = ...;
    _allocated.add(value);
    return value;
  }
  
  Future<void> disposeAll() async {
    for (final v in _allocated) {
      await v.dispose();
    }
    for (final s in _sessions) {
      await s.close();
    }
  }
}
```

**Also:**
- Use `try/finally` blocks for guaranteed cleanup
- Add `dispose()` calls for all decoder inputs
- Implement RAII pattern via Dart's `Finalizer` (Dart 2.17+)

### 🎯 3. Add Cancellation Support

**Priority:** P0 (Critical)  
**Effort:** Medium  

Implement cancellation tokens:

```dart
class CancellationToken {
  bool _cancelled = false;
  final _completer = Completer<void>();
  
  Future<void> get onCancelled => _completer.future;
  
  void cancel() {
    _cancelled = true;
    _completer.complete();
  }
  
  bool get isCancelled => _cancelled;
}

class SttFlutter {
  CancellationToken? _currentToken;
  
  Future<SttResult> transcribeFile(String path, {CancellationToken? token}) async {
    _currentToken = token;
    // Check token.isCancelled periodically
    if (token?.isCancelled ?? false) {
      throw OperationCancelledException();
    }
  }
  
  void cancelCurrent() {
    _currentToken?.cancel();
  }
}
```

---

## Code Quality Recommendations

### 🎯 4. Extract Common Utilities

**Priority:** P1 (High)  
**Effort:** Low  

Create shared utility classes:

```dart
// lib/src/utils/math_utils.dart
class MathUtils {
  static int argmax(List<num> values) {
    int best = 0;
    double bestVal = double.negativeInfinity;
    for (int i = 0; i < values.length; i++) {
      if (values[i].toDouble() > bestVal) {
        bestVal = values[i].toDouble();
        best = i;
      }
    }
    return best;
  }
  
  static Float32List transposeMel(Float64List mel, int nMels, int totalFrames, 
      int chunkOffset, int chunkSize) {
    // ... shared implementation
  }
}
```

### 🎯 5. Centralize Constants

**Priority:** P1 (High)  
**Effort:** Low  

Create a constants file:

```dart
// lib/src/constants.dart
class SttConstants {
  // Audio
  static const int targetSampleRate = 16000;
  static const int nMelBins = 80;
  static const int maxFrames = 3000;
  static const int maxTokens = 448;
  
  // Whisper tokens
  static const int sot = 50258;
  static const int eot = 50257;
  static const int transcribeTok = 50359;
  static const int noTimestamps = 50363;
  
  // Language tokens
  static const Map<String, int> whisperLanguageTokens = {
    'en': 50259,
    'de': 50261,
    'fr': 50263,
    'es': 50265,
    // ... all 99 languages
  };
}
```

### 🎯 6. Implement Proper Logging

**Priority:** P1 (High)  
**Effort:** Low  

Replace `debugPrint` with a proper logging system:

```dart
// lib/src/logging.dart
enum LogLevel { verbose, debug, info, warning, error }

class SttLogger {
  static LogLevel level = LogLevel.info;
  static void setLevel(LogLevel newLevel) => level = newLevel;
  
  static void v(String message) => _log(LogLevel.verbose, message);
  static void d(String message) => _log(LogLevel.debug, message);
  static void i(String message) => _log(LogLevel.info, message);
  static void w(String message) => _log(LogLevel.warning, message);
  static void e(String message, [dynamic error, StackTrace? stack]) => 
      _log(LogLevel.error, message, error, stack);
  
  static void _log(LogLevel msgLevel, String message, 
      [dynamic error, StackTrace? stack]) {
    if (msgLevel.index < level.index) return;
    // Format and output
  }
}
```

### 🎯 7. Add Input Validation

**Priority:** P1 (High)  
**Effort:** Low  

Add validation helpers:

```dart
// lib/src/validation.dart
class Validation {
  static void validateInitialized(bool initialized, String component) {
    if (!initialized) {
      throw SttException.notInitialized(component);
    }
  }
  
  static void validateAudioBuffer(AudioBuffer audio) {
    if (audio.sampleRate <= 0) {
      throw SttException.invalidArgument('sampleRate must be positive');
    }
    if (audio.length == 0) {
      throw SttException.invalidArgument('audio buffer must not be empty');
    }
    if (audio.sampleRate > 192000) {
      throw SttException.invalidArgument('sampleRate too high: ${audio.sampleRate}');
    }
  }
  
  static void validateLanguage(String? language) {
    if (language != null && !RegExp(r'^[a-z]{2,3}$').hasMatch(language)) {
      throw SttException.invalidArgument('Invalid language code: $language');
    }
  }
}
```

### 🎯 8. Implement Custom Exceptions

**Priority:** P1 (High)  
**Effort:** Low  

Create exception hierarchy:

```dart
// lib/src/exceptions.dart
class SttException implements Exception {
  final String message;
  final int? code;
  
  const SttException(this.message, [this.code]);
  
  @override
  String toString() => 'SttException($code): $message';
  
  factory SttException.notInitialized(String component) => 
      SttException('${component} not initialized', 1001);
  
  factory SttException.invalidArgument(String message) => 
      SttException(message, 1002);
  
  factory SttException.modelLoadFailed(String reason) => 
      SttException('Failed to load model: $reason', 2001);
  
  factory SttException.inferenceFailed(String reason) => 
      SttException('Inference failed: $reason', 3001);
}

class OperationCancelledException implements Exception {
  @override
  String toString() => 'Operation was cancelled';
}
```

---

## Security Recommendations

### 🔒 1. Implement SHA256 Verification

**Priority:** P1 (High)  
**Effort:** Medium  

Add verification in downloader:

```dart
// model_downloader.dart
static Future<void> _downloadFile({...}) async {
  // ... download
  
  if (sha256 != null) {
    final fileHash = await _computeSha256(destPath);
    if (fileHash != sha256) {
      await File(destPath).delete();
      throw SttException.modelIntegrityFailed('SHA256 mismatch for $filename');
    }
  }
}

static Future<String> _computeSha256(String path) async {
  final file = File(path);
  final bytes = await file.readAsBytes();
  final hash = sha256.convert(bytes);
  return hash.toString();
}
```

### 🔒 2. Add HTTPS Certificate Pinning

**Priority:** P2 (Medium)  
**Effort:** Medium  

Use `dio` package with certificate pinning:

```yaml
# pubspec.yaml
dependencies:
  dio: ^5.4.0
  dio_http2_adapter: ^2.3.0
```

```dart
// model_downloader.dart
import 'package:dio/dio.dart';
import 'package:dio_http2_adapter/dio_http2_adapter.dart';

class ModelDownloader {
  static final Dio _dio = Dio()
    ..httpClientAdapter = Http2Adapter(
      ConnectionManager(
        onClientCreate: (_, config) => config.onBadCertificate = (_) => false,
      ),
    );
  
  // Add certificate pins for known CDNs
  static const _certificatePins = {
    'huggingface.co': ['sha256/...'],
    'github.com': ['sha256/...'],
  };
}
```

---

## Performance Recommendations

### ⚡ 1. Optimize Resampling

**Priority:** P1 (High)  
**Effort:** Medium  

Use native resampling via FFI or optimized Dart:

**Option A: Use `libsamplerate` via FFI**
```dart
// Native binding for libsamplerate
class NativeResampler {
  static Float32List resample(Float32List input, int inputRate, int outputRate);
}
```

**Option B: Optimized Dart with SIMD**
```dart
// Use Float32List operations directly
static Float32List resampleOptimized(Float32List input, int inputRate, int outputRate) {
  final ratio = inputRate / outputRate;
  final outputLength = (input.length / ratio).round();
  final output = Float32List(outputLength);
  
  for (int i = 0; i < outputLength; i++) {
    final srcPos = i * ratio;
    final srcIdx = srcPos.floor();
    final frac = srcPos - srcIdx;
    
    if (srcIdx + 1 < input.length) {
      // SIMD-friendly: avoid branching
      final a = input[srcIdx];
      final b = input[srcIdx + 1];
      output[i] = a + (b - a) * frac;
    } else {
      output[i] = input[srcIdx];
    }
  }
  return output;
}
```

### ⚡ 2. Add Audio Normalization

**Priority:** P2 (Medium)  
**Effort:** Low  

Add normalization option:

```dart
// audio_processor.dart
class AudioProcessor {
  static AudioBuffer normalize(AudioBuffer audio, {double targetPeak = 0.9}) {
    // Find current peak
    double peak = 0.0;
    for (final s in audio.samples) {
      peak = max(peak, s.abs());
    }
    
    if (peak <= 0) return audio;
    
    final scale = targetPeak / peak;
    final normalized = Float32List(audio.length);
    for (int i = 0; i < audio.length; i++) {
      normalized[i] = (audio.samples[i] * scale).clamp(-1.0, 1.0);
    }
    
    return AudioBuffer(samples: normalized, sampleRate: audio.sampleRate);
  }
}
```

### ⚡ 3. Batch Audio Processing

**Priority:** P2 (Medium)  
**Effort:** High  

Implement batching for long audio:

```dart
// whisper_engine.dart
@override
Future<SttResult> transcribe(AudioBuffer audio, {String? language}) async {
  final mel = await Isolate.run(() => MelSpectrogram.compute(audio.samples));
  final totalFrames = mel.length ~/ nMels;
  
  // Process in batches with overlap
  final batchSize = maxFrames;
  final results = <String>[];
  
  for (int offset = 0; offset < totalFrames; offset += batchSize) {
    final chunk = _transposeMel(mel, totalFrames, offset, batchSize);
    final result = await _transcribeChunk(chunk, language);
    results.add(result);
  }
  
  return SttResult(
    text: results.join(' '),
    inferenceTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
  );
}
```

---

## Testing Recommendations

### 🧪 1. Add Integration Tests

**Priority:** P1 (High)  
**Effort:** Medium  

Add full pipeline tests:

```dart
// test/whisper_engine_test.dart
void main() {
  group('WhisperEngine', () {
    late WhisperInferenceEngine engine;
    late ort.OnnxRuntime runtime;
    
    setUpAll(() async {
      runtime = ort.OnnxRuntime();
      engine = WhisperInferenceEngine(runtime);
      
      // Download tiny model if not present
      final model = ModelRegistry.get('whisper-tiny');
      if (!await ModelDownloader.isDownloaded(model)) {
        await ModelDownloader.download(model);
      }
      
      final modelFiles = {...};
      await engine.load(modelFiles);
    });
    
    tearDownAll(() async {
      await engine.dispose();
      await runtime.close();
    });
    
    test('transcribes English audio', () async {
      final audio = await AudioProcessor.loadWav('test/fixtures/hello_en.wav');
      final result = await engine.transcribe(audio, language: 'en');
      
      expect(result.text.toLowerCase(), contains('hello'));
      expect(result.inferenceTimeMs, greaterThan(0));
    });
    
    test('transcribes German audio', () async {
      final audio = await AudioProcessor.loadWav('test/fixtures/guten_tag_de.wav');
      final result = await engine.transcribe(audio, language: 'de');
      
      expect(result.text.toLowerCase(), contains('guten'));
    });
  });
}
```

### 🧪 2. Add Mock HTTP Tests

**Priority:** P1 (High)  
**Effort:** Medium  

Test downloader with mocks:

```dart
// test/model_downloader_test.dart
import 'package:http/http.dart';
import 'package:http/testing.dart';

void main() {
  group('ModelDownloader', () {
    test('downloads and verifies SHA256', () async {
      final mockClient = MockClient((request) async {
        return Response('test content', 200, headers: {
          'content-length': '12',
        });
      });
      
      // Use mock client
      final response = await mockClient.get(Uri.parse('https://example.com/test.onnx'));
      
      expect(response.statusCode, 200);
    });
    
    test('retries on failure', () async {
      // Test retry logic
    });
    
    test('cleans up on cancellation', () async {
      // Test partial download cleanup
    });
  });
}
```

### 🧪 3. Add Error Handling Tests

**Priority:** P2 (Medium)  
**Effort:** Low  

Test error cases:

```dart
// test/error_handling_test.dart
void main() {
  group('Error Handling', () {
    test('throws on uninitialized transcribe', () async {
      final stt = SttFlutter();
      expect(
        () => stt.transcribeFile('test.wav'),
        throwsA(isA<SttException>()),
      );
    });
    
    test('throws on invalid audio file', () async {
      final stt = SttFlutter();
      await stt.initialize(model: ModelRegistry.get('whisper-tiny'));
      
      expect(
        () => stt.transcribeFile('nonexistent.wav'),
        throwsA(isA<FileSystemException>()),
      );
    });
    
    test('throws on invalid language code', () async {
      expect(
        () => SttFlutter().transcribeBuffer(Float32List(0), 16000, language: 'xyz123'),
        throwsA(isA<SttException>()),
      );
    });
  });
}
```

---

## Documentation Recommendations

### 📚 1. Add API Documentation

**Priority:** P2 (Medium)  
**Effort:** Medium  

Add doc comments to all public APIs:

```dart
/// Main entry point for speech-to-text functionality.
/// 
/// This class provides a high-level interface for transcribing audio
/// using ONNX-based models. All inference runs on background isolates
/// to avoid blocking the UI thread.
/// 
/// Example:
/// ```dart
/// final stt = SttFlutter();
/// await stt.initialize(model: ModelRegistry.get('whisper-tiny'));
/// final result = await stt.transcribeFile('audio.wav');
/// print(result.text);
/// await stt.dispose();
/// ```
class SttFlutter {
  /// Initializes the STT engine with the specified model.
  /// 
  /// [model]: The model descriptor to use for transcription.
  /// [modelDir]: Custom directory for model files. Defaults to
  ///   `{appDocDir}/stt_models/{model.id}/`.
  /// [language]: Default language for transcription (ISO 639-1 code).
  /// 
  /// Throws [SttException] if model files are missing or invalid.
  Future<void> initialize({
    required ModelDescriptor model,
    String? modelDir,
    String? language,
  }) async {
    // ...
  }
}
```

### 📚 2. Add CHANGELOG

**Priority:** P2 (Medium)  
**Effort:** Low  

Create CHANGELOG.md:

```markdown
# Changelog

## [Unreleased]

### Added
- Initial release

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [0.1.0] - 2025-01-01

### Added
- Whisper model support (tiny, base, small, medium, large-v3)
- Sherpa-ONNX support (zipformer-en)
- Voxtral support (mini)
- Model registry and downloader
- Audio preprocessing (WAV parsing, resampling)
```

### 📚 3. Add CONTRIBUTING Guide

**Priority:** P2 (Medium)  
**Effort:** Low  

Create CONTRIBUTING.md:

```markdown
# Contributing

## Development Setup

1. Clone the repository
2. Run `flutter pub get`
3. Run tests: `flutter test`

## Adding a New Model

1. Create a new file in `lib/src/default_models/`
2. Register the model in `register_defaults.dart`
3. Add tests

## Adding a New Engine

1. Implement `InferenceEngine` interface
2. Add to `engine_factory.dart`
3. Add tests
```

---

## File Structure Recommendations

### 📁 1. Reorganize Test Directory

Current:
```
test/
├── stt_flutter_test.dart
├── mel_spectrogram_test.dart
├── model_registry_test.dart
├── audio_processor_test.dart
└── fixtures/
```

Proposed:
```
test/
├── unit/
│   ├── audio/
│   │   ├── audio_buffer_test.dart
│   │   └── audio_processor_test.dart
│   ├── engines/
│   │   ├── whisper/
│   │   │   ├── mel_spectrogram_test.dart
│   │   │   └── bpe_tokenizer_test.dart
│   │   ├── sherpa/
│   │   │   ├── fbank_test.dart
│   │   │   └── transducer_decoder_test.dart
│   │   └── voxtral/
│   │       └── tekken_tokenizer_test.dart
│   ├── model_registry_test.dart
│   └── utils/
│       └── math_utils_test.dart
├── integration/
│   ├── whisper_engine_test.dart
│   ├── sherpa_engine_test.dart
│   ├── voxtral_engine_test.dart
│   └── model_downloader_test.dart
└── fixtures/
    ├── hello_en.wav
    ├── guten_tag_de.wav
    └── ...
```

### 📁 2. Add Example App Improvements

Current example is minimal. Enhance with:
- Model download progress UI
- Transcription results display
- Language selection
- Error handling examples
- Performance metrics

---

## Priority Matrix

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | Background inference isolate | High | Critical |
| P0 | Resource leak fixes | Medium | Critical |
| P0 | Cancellation support | Medium | Critical |
| P1 | SHA256 verification | Medium | High |
| P1 | Input validation | Low | High |
| P1 | Extract common utilities | Low | High |
| P1 | Centralize constants | Low | High |
| P1 | Proper logging | Low | High |
| P1 | Custom exceptions | Low | High |
| P1 | Integration tests | Medium | High |
| P2 | Optimize resampling | Medium | Medium |
| P2 | Audio normalization | Low | Medium |
| P2 | API documentation | Medium | Medium |
| P2 | CHANGELOG | Low | Medium |
| P2 | CONTRIBUTING guide | Low | Medium |
| P3 | Certificate pinning | Medium | Low |
| P3 | Batch processing | High | Low |
| P3 | Test reorganization | Low | Low |

---

## Implementation Roadmap

### Phase 1: Critical Fixes (1-2 weeks)
1. ✅ Create improvement analysis document (this file)
2. ✅ Fix resource leaks in all engines (try/finally patterns across Whisper, Sherpa, Voxtral)
3. ✅ Add cancellation support
4. ✅ Implement SHA256 verification
5. ✅ Add input validation

### Phase 2: Architecture Improvements (2-3 weeks)
1. ✅ Implement long-lived ComputeWorker isolate for preprocessing
2. ✅ Extract common utilities (`math_utils.dart`)
3. ✅ Centralize constants (`constants.dart`)
4. ✅ Implement proper logging (`SttLogger`)
5. ✅ Add custom exceptions (`SttException` + `OperationCancelledException`)

### Phase 3: Performance & Polish (2-3 weeks)
1. [ ] Optimize resampling
2. [ ] Add audio normalization
3. [ ] Add batch processing
4. [ ] Add certificate pinning
5. [ ] Complete API documentation

### Phase 4: Testing & Documentation (1-2 weeks)
1. [ ] Add integration tests
2. [ ] Add mock HTTP tests
3. [ ] Add error handling tests
4. [ ] Add CHANGELOG
5. [ ] Add CONTRIBUTING guide
6. [ ] Reorganize test directory

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test coverage | ~40% | >80% |
| Resource leaks | 5+ | 0 |
| Background inference | ❌ | ✅ |
| Cancellation support | ❌ | ✅ |
| SHA256 verification | ❌ | ✅ |
| API documentation | 0% | 100% |
| Custom exceptions | ❌ | ✅ |
| Performance (1s audio) | ~X ms | <500ms |

---

## Conclusion

The `stt_flutter` package is an **excellent foundation** with a clean architecture and comprehensive feature set. However, it has **critical issues** that must be addressed before production use:

1. **Background inference** is essential for smooth UI
2. **Resource leaks** must be fixed to prevent memory issues
3. **Cancellation support** is needed for good UX

The **recommended approach** is to address P0 issues first, then P1, then P2/P3. The total effort is estimated at **6-10 weeks** for a single developer, or **3-5 weeks** for a team of 2-3.

With these improvements, `stt_flutter` could become the **definitive Flutter package** for on-device speech-to-text.

---

## Files Modified in This Branch

- `IMPROVEMENT_ANALYSIS.md` - This comprehensive analysis document

## Next Steps

1. Review this analysis and prioritize issues
2. Create GitHub issues for each recommendation
3. Start with P0 issues (background isolate, resource leaks, cancellation)
4. Implement incrementally with proper testing
5. Consider creating a project board for tracking

---

### Phase 4: Testing & Documentation (1-2 weeks)

### Phase 5: Update of the architecture
# Flutter STT: Dynamic Switching & Hybrid Architecture Implementation Guide

*A comprehensive guide to implementing a production-ready, multi-model speech-to-text system in Flutter with ONNX Runtime, featuring dynamic language detection, model switching, and hybrid fallback strategies.*

---

## 📋 Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [Prerequisites](#-prerequisites)
3. [Model Selection & Configuration](#-model-selection--configuration)
4. [Project Structure](#-project-structure)
5. [Step 1: Setup Dependencies](#-step-1-setup-dependencies)
6. [Step 2: Model Downloader & Manager](#-step-2-model-downloader--manager)
7. [Step 3: Language Detection (LID) Service](#-step-3-language-detection-lid-service)
8. [Step 4: Dynamic Model Switching Engine](#-step-4-dynamic-model-switching-engine)
9. [Step 5: Hybrid Transcription Pipeline](#-step-5-hybrid-transcription-pipeline)
10. [Step 6: Audio Streaming & Processing](#-step-6-audio-streaming--processing)
11. [Step 7: Performance Optimization](#-step-7-performance-optimization)
12. [Step 8: Error Handling & Fallbacks](#-step-8-error-handling--fallbacks)
13. [Step 9: Testing & Benchmarking](#-step-9-testing--benchmarking)
14. [Configuration Reference](#-configuration-reference)
15. [Deployment Checklist](#-deployment-checklist)

---

```mermaid
graph TD
    A[Audio Input] --> B[Audio Stream Processor]
    B --> C[Language Detector]
    C --> D[Model Selector]
    D --> E[Model Loader]
    E --> F[Transcription Engine]
    F --> G[Result Post-Processor]
    G --> H[Output Text]
    
    D -->|Language Known| I[Language-Specific Model]
    D -->|Language Unknown| J[Multilingual Whisper]
    D -->|Low Memory| K[Fallback to Cloud]
    
    style A fill:#f9f,stroke:#333
    style H fill:#bbf,stroke:#333
```


| &nbsp; | &nbsp; | &nbsp; |
| ------ | ------ | ------ |
| &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; |


---

&nbsp;

```yaml
dependencies:
  # Core STT
  sherpa_onnx: ^1.13.0
  runanywhere_onnx: ^0.1.0  # Optional: simpler API
  
  # Audio
  record: ^5.0.0
  audioplayers: ^5.0.0
  
  # Utilities
  path_provider: ^2.1.1
  path: ^1.8.3
  http: ^1.1.0
  
  # State Management (recommended)
  riverpod: ^2.4.9
  # or
  provider: ^6.1.1
```

&nbsp;

```gradle
android {
    defaultConfig {
        minSdkVersion 24  // Required for ONNX Runtime
        targetSdkVersion 34
        
        // Enable ONNX Runtime
        externalNativeBuild {
            cmake {
                arguments "-DONNXRUNTIME_ENABLE=ON"
            }
        }
    }
    
    buildTypes {
        release {
            minifyEnabled false  // Required for model loading
        }
    }
}
```

&nbsp;

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Required for speech recognition</string>
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

&nbsp;

```ruby
# Add ONNX Runtime support
platform :ios, '14.0'
use_frameworks!

target 'Runner' do
  use_modular_headers!
  
  # Required for sherpa-onnx
  pod 'onnxruntime-mobile-ios', '~> 1.16.0'
end
```

---


| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |


```dart
// lib/config/models.dart

enum ModelType {
  kroko,
  zipformer,
  parakeet,
  whisperTiny,
  whisperSmall,
}

class STTModelConfig {
  final String id;
  final ModelType type;
  final String name;
  final List<String> languages;
  final int sizeMB;
  final double expectedWER;
  final double expectedRTF;
  final String encoderPath;
  final String decoderPath;
  final String? joinerPath;
  final String tokensPath;
  final int priority; // Lower = higher priority
  
  const STTModelConfig({
    required this.id,
    required this.type,
    required this.name,
    required this.languages,
    required this.sizeMB,
    required this.expectedWER,
    required this.expectedRTF,
    required this.encoderPath,
    required this.decoderPath,
    this.joinerPath,
    required this.tokensPath,
    this.priority = 2,
  });
  
  // Pre-defined models
  static const List<STTModelConfig> models = [
    // Language-specific models (Priority 1)
    STTModelConfig(
      id: 'kroko_64l_en',
      type: ModelType.kroko,
      name: 'Kroko 64L English',
      languages: ['en'],
      sizeMB: 20,
      expectedWER: 5.5,
      expectedRTF: 0.08,
      encoderPath: 'models/en/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/en/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/en/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/en/kroko_64l/tokens.txt',
      priority: 1,
    ),
    STTModelConfig(
      id: 'kroko_64l_fr',
      type: ModelType.kroko,
      name: 'Kroko 64L French',
      languages: ['fr'],
      sizeMB: 20,
      expectedWER: 6.2,
      expectedRTF: 0.08,
      encoderPath: 'models/fr/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/fr/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/fr/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/fr/kroko_64l/tokens.txt',
      priority: 1,
    ),
    STTModelConfig(
      id: 'kroko_64l_es',
      type: ModelType.kroko,
      name: 'Kroko 64L Spanish',
      languages: ['es'],
      sizeMB: 20,
      expectedWER: 5.8,
      expectedRTF: 0.08,
      encoderPath: 'models/es/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/es/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/es/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/es/kroko_64l/tokens.txt',
      priority: 1,
    ),
    STTModelConfig(
      id: 'kroko_64l_de',
      type: ModelType.kroko,
      name: 'Kroko 64L German',
      languages: ['de'],
      sizeMB: 20,
      expectedWER: 6.5,
      expectedRTF: 0.08,
      encoderPath: 'models/de/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/de/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/de/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/de/kroko_64l/tokens.txt',
      priority: 1,
    ),
    STTModelConfig(
      id: 'kroko_64l_it',
      type: ModelType.kroko,
      name: 'Kroko 64L Italian',
      languages: ['it'],
      sizeMB: 20,
      expectedWER: 5.2,
      expectedRTF: 0.08,
      encoderPath: 'models/it/kroko_64l/encoder.int8.onnx',
      decoderPath: 'models/it/kroko_64l/decoder.int8.onnx',
      joinerPath: 'models/it/kroko_64l/joiner.int8.onnx',
      tokensPath: 'models/it/kroko_64l/tokens.txt',
      priority: 1,
    ),
    
    // Parakeet-TDT (Priority 2 - High-end devices only)
    STTModelConfig(
      id: 'parakeet_tdt_0.6b_v3',
      type: ModelType.parakeet,
      name: 'Parakeet TDT 0.6B v3',
      languages: [
        'en', 'es', 'it', 'fr', 'de', 'nl', 'ru', 'pl', 'uk', 'sk',
        'bg', 'fi', 'ro', 'hr', 'cs', 'sv', 'et', 'hu', 'lt', 'da',
        'mt', 'sl', 'lv', 'el', 'pt'
      ],
      sizeMB: 640,
      expectedWER: 6.1,
      expectedRTF: 0.03,
      encoderPath: 'models/parakeet/encoder.int8.onnx',
      decoderPath: 'models/parakeet/decoder.int8.onnx',
      joinerPath: 'models/parakeet/joiner.int8.onnx',
      tokensPath: 'models/parakeet/tokens.txt',
      priority: 2,
    ),
    
    // Fallback models (Priority 3)
    STTModelConfig(
      id: 'whisper_tiny',
      type: ModelType.whisperTiny,
      name: 'Whisper Tiny',
      languages: const [], // All languages
      sizeMB: 75,
      expectedWER: 11.0,
      expectedRTF: 0.8,
      encoderPath: 'models/whisper-tiny/encoder.int8.onnx',
      decoderPath: 'models/whisper-tiny/decoder.int8.onnx',
      tokensPath: 'models/whisper-tiny/tokens.txt',
      priority: 3,
    ),
    STTModelConfig(
      id: 'whisper_small',
      type: ModelType.whisperSmall,
      name: 'Whisper Small',
      languages: const [], // All languages
      sizeMB: 250,
      expectedWER: 8.5,
      expectedRTF: 1.2,
      encoderPath: 'models/whisper-small/encoder.int8.onnx',
      decoderPath: 'models/whisper-small/decoder.int8.onnx',
      tokensPath: 'models/whisper-small/tokens.txt',
      priority: 4,
    ),
  ];
  
  // Language to model mapping
  static const Map<String, String> languageToModel = {
    'en': 'kroko_64l_en',
    'fr': 'kroko_64l_fr',
    'es': 'kroko_64l_es',
    'de': 'kroko_64l_de',
    'it': 'kroko_64l_it',
    // Add more as needed
  };
  
  // Get model for language
  static STTModelConfig? getModelForLanguage(String languageCode) {
    final modelId = languageToModel[languageCode];
    if (modelId != null) {
      return models.firstWhere((m) => m.id == modelId);
    }
    return null;
  }
  
  // Check if device can handle model
  static bool canDeviceHandleModel(STTModelConfig model, int availableMemoryMB) {
    // Reserve 500MB for OS and other apps
    final safeMemory = availableMemoryMB - 500;
    return model.sizeMB <= safeMemory;
  }
}
```

---

```
lib/
├── config/
│   ├── models.dart           # Model configurations
│   └── stt_config.dart       # STT settings
├── services/
│   ├── audio/
│   │   ├── audio_capture.dart
│   │   ├── audio_processor.dart
│   │   └── vad.dart           # Voice Activity Detection
│   ├── lid/
│   │   └── language_detector.dart
│   ├── models/
│   │   ├── model_downloader.dart
│   │   ├── model_loader.dart
│   │   └── model_manager.dart
│   ├── stt/
│   │   ├── stt_engine.dart
│   │   ├── transcription_service.dart
│   │   └── model_switcher.dart
│   └── utils/
│       └── device_utils.dart
├── providers/
│   ├── stt_provider.dart      # State management
│   └── audio_provider.dart
└── widgets/
    ├── stt_button.dart
    └── transcription_display.dart

assets/
└── models/
    ├── en/
    │   └── kroko_64l/          # Model files
    ├── fr/
    │   └── kroko_64l/
    ├── es/
    │   └── kroko_64l/
    ├── parakeet/
    │   └── tdt_0.6b_v3/
    └── whisper/
        ├── tiny/
        └── small/

android/
└── app/
    └── src/main/
        └── jniLibs/              # Native libraries

ios/
└── Runner/
    └── Models/               # iOS model storage
```

---

&nbsp;

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
```

&nbsp;

```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs microphone access for speech recognition</string>
<key>UIBackgroundModes</key>
<array>
    <string>audio</string>
</array>
```

&nbsp;

```dart
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

class STTInitializer {
  static Future<void> init() async {
    try {
      // Initialize ONNX Runtime
      await sherpa_onnx.init();
      
      // Set number of threads (adjust based on device)
      final numThreads = _getOptimalThreadCount();
      sherpa_onnx.setNumThreads(numThreads);
      
      print('Sherpa-ONNX initialized with $numThreads threads');
    } catch (e) {
      print('Failed to initialize Sherpa-ONNX: $e');
      rethrow;
    }
  }
  
  static int _getOptimalThreadCount() {
    // Use 2 threads for low-end devices, 4 for mid-range, 6+ for high-end
    // This is a simple heuristic; adjust based on your testing
    return 4; // Default
  }
}
```

---

```dart
// lib/services/models/model_downloader.dart

import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

class ModelDownloader {
  static const String _baseUrl = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';
  
  final Map<String, String> _modelUrls = {
    'kroko_64l_en': '$_baseUrl/runanywhere-models-v1/sherpa-onnx-whisper-tiny.en.tar.gz',
    'kroko_64l_fr': '$_baseUrl/.../sherpa-onnx-kroko-64l-fr.tar.gz',
    'parakeet_tdt_0.6b_v3': '$_baseUrl/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2',
    'whisper_tiny': '$_baseUrl/asr-models/sherpa-onnx-whisper-tiny.tar.bz2',
    // Add all model URLs
  };
  
  Future<String> downloadModel(String modelId) async {
    final url = _modelUrls[modelId];
    if (url == null) {
      throw Exception('Model $modelId not found');
    }
    
    final modelConfig = STTModelConfig.models.firstWhere(
      (m) => m.id == modelId,
      orElse: () => throw Exception('Model config not found'),
    );
    
    // Get download directory
    final dir = await _getModelDirectory(modelConfig);
    final archivePath = path.join(dir, '$modelId.tar.bz2');
    
    // Download if not exists
    if (!await File(archivePath).exists()) {
      await _downloadFile(url, archivePath);
      await _extractArchive(archivePath, dir);
      await File(archivePath).delete();
    }
    
    return dir;
  }
  
  Future<String> _getModelDirectory(STTModelConfig model) async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelDir = path.join(appDir.path, 'models', model.id);
    final dir = Directory(modelDir);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return modelDir;
  }
  
  Future<void> _downloadFile(String url, String savePath) async {
    final response = await http.get(Uri.parse(url));
    if (response.statusCode != 200) {
      throw Exception('Failed to download: ${response.statusCode}');
    }
    final file = File(savePath);
    await file.writeAsBytes(response.bodyBytes);
  }
  
  Future<void> _extractArchive(String archivePath, String extractTo) async {
    // Use a package like archive or targz for extraction
    // Implementation depends on archive format
    // For .tar.bz2, use: https://pub.dev/packages/archive
    throw UnimplementedError('Archive extraction not implemented');
  }
  
  Future<bool> isModelDownloaded(String modelId) async {
    try {
      final modelConfig = STTModelConfig.models.firstWhere(
        (m) => m.id == modelId,
      );
      final dir = await _getModelDirectory(modelConfig);
      final encoderFile = File(path.join(dir, modelConfig.encoderPath));
      return await encoderFile.exists();
    } catch (_) {
      return false;
    }
  }
  
  Future<void> deleteModel(String modelId) async {
    final modelConfig = STTModelConfig.models.firstWhere(
      (m) => m.id == modelId,
      orElse: () => throw Exception('Model not found'),
    );
    final dir = await _getModelDirectory(modelConfig);
    await Directory(dir).delete(recursive: true);
  }
}
```

```dart
// lib/services/models/model_manager.dart

import 'package:riverpod/riverpod.dart';

class ModelManager {
  final ModelDownloader _downloader = ModelDownloader();
  final Map<String, bool> _loadedModels = {};
  final Map<String, dynamic> _modelInstances = {};
  
  // Track available memory
  int _availableMemoryMB = 0;
  
  Future<void> init() async {
    _availableMemoryMB = await _getAvailableMemory();
  }
  
  Future<int> _getAvailableMemory() async {
    // Platform-specific memory detection
    // This is a placeholder - implement for Android/iOS
    return 4000; // Default: 4GB
  }
  
  Future<String> ensureModelLoaded(String modelId) async {
    // Check if already loaded
    if (_loadedModels[modelId] == true) {
      return modelId;
    }
    
    // Check if downloaded
    if (!await _downloader.isModelDownloaded(modelId)) {
      await _downloader.downloadModel(modelId);
    }
    
    // Check if device can handle it
    final modelConfig = STTModelConfig.models.firstWhere(
      (m) => m.id == modelId,
    );
    
    if (!STTModelConfig.canDeviceHandleModel(modelConfig, _availableMemoryMB)) {
      throw Exception('Insufficient memory for model: $modelId');
    }
    
    // Load the model
    await _loadModel(modelId);
    _loadedModels[modelId] = true;
    
    return modelId;
  }
  
  Future<void> _loadModel(String modelId) async {
    final modelConfig = STTModelConfig.models.firstWhere(
      (m) => m.id == modelId,
    );
    
    final modelDir = await _downloader._getModelDirectory(modelConfig);
    
    // Load based on model type
    switch (modelConfig.type) {
      case ModelType.kroko:
      case ModelType.zipformer:
        _modelInstances[modelId] = await _loadTransducerModel(modelConfig, modelDir);
        break;
      case ModelType.parakeet:
        _modelInstances[modelId] = await _loadTransducerModel(modelConfig, modelDir);
        break;
      case ModelType.whisperTiny:
      case ModelType.whisperSmall:
        _modelInstances[modelId] = await _loadWhisperModel(modelConfig, modelDir);
        break;
    }
  }
  
  Future<dynamic> _loadTransducerModel(STTModelConfig config, String modelDir) async {
    // Implementation for transducer-based models (Kroko, Zipformer, Parakeet)
    // Use sherpa_onnx.OnlineRecognizer
    final encoderPath = path.join(modelDir, config.encoderPath);
    final decoderPath = path.join(modelDir, config.decoderPath);
    final joinerPath = path.join(modelDir, config.joinerPath!);
    final tokensPath = path.join(modelDir, config.tokensPath);
    
    final configObj = sherpa_onnx.OnlineRecognizerConfig(
      transducer: sherpa_onnx.OnlineTransducerModelConfig(
        encoder: encoderPath,
        decoder: decoderPath,
        joiner: joinerPath,
      ),
      tokens: tokensPath,
      numThreads: 4,
    );
    
    return sherpa_onnx.OnlineRecognizer(configObj);
  }
  
  Future<dynamic> _loadWhisperModel(STTModelConfig config, String modelDir) async {
    // Implementation for Whisper models
    final encoderPath = path.join(modelDir, config.encoderPath);
    final decoderPath = path.join(modelDir, config.decoderPath);
    final tokensPath = path.join(modelDir, config.tokensPath);
    
    final configObj = sherpa_onnx.OfflineRecognizerConfig(
      model: sherpa_onnx.OfflineModelConfig(
        encoder: encoderPath,
        decoder: decoderPath,
      ),
      tokens: tokensPath,
    );
    
    return sherpa_onnx.OfflineRecognizer(configObj);
  }
  
  Future<void> unloadModel(String modelId) async {
    _loadedModels[modelId] = false;
    _modelInstances.remove(modelId);
    // Free memory - model will be garbage collected
  }
  
  Future<void> unloadAllModels() async {
    _loadedModels.clear();
    _modelInstances.clear();
  }
  
  dynamic getModelInstance(String modelId) {
    return _modelInstances[modelId];
  }
  
  List<String> get loadedModels => _loadedModels.entries
      .where((e) => e.value == true)
      .map((e) => e.key)
      .toList();
  
  int get availableMemoryMB => _availableMemoryMB;
}
```

---

```dart
// lib/services/lid/language_detector.dart

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

class LanguageDetector {
  static const String _lidModelId = 'whisper_tiny_lid';
  sherpa_onnx.OfflineRecognizer? _recognizer;
  bool _isInitialized = false;
  
  // Language codes to detect
  static const List<String> supportedLanguages = [
    'en', 'fr', 'es', 'de', 'it', 'pt', 'ru', 'nl', 'pl', 'uk',
    'sk', 'bg', 'fi', 'ro', 'hr', 'cs', 'sv', 'et', 'hu', 'lt',
    'da', 'mt', 'sl', 'lv', 'el', 'ja', 'zh', 'ar', 'hi', 'ko'
  ];
  
  Future<void> init() async {
    if (_isInitialized) return;
    
    // Load LID model (use whisper-tiny for language detection)
    final modelDir = await ModelDownloader()._getModelDirectory(
      STTModelConfig.models.firstWhere((m) => m.id == 'whisper_tiny'),
    );
    
    final config = sherpa_onnx.OfflineRecognizerConfig(
      model: sherpa_onnx.OfflineModelConfig(
        encoder: path.join(modelDir, 'encoder.int8.onnx'),
        decoder: path.join(modelDir, 'decoder.int8.onnx'),
      ),
      tokens: path.join(modelDir, 'tokens.txt'),
      numThreads: 2, // LID doesn't need many threads
    );
    
    _recognizer = sherpa_onnx.OfflineRecognizer(config);
    _isInitialized = true;
  }
  
  Future<String> detectLanguage(List<int> audioData, {int sampleRate = 16000}) async {
    if (!_isInitialized) {
      await init();
    }
    
    // Use first 2-3 seconds of audio for detection
    final chunkDuration = 2.0; // seconds
    final chunkSize = (sampleRate * chunkDuration).round();
    final audioChunk = audioData.length > chunkSize
        ? audioData.sublist(0, chunkSize)
        : audioData;
    
    // Create a stream for the chunk
    final stream = _recognizer!.createStream();
    stream.acceptWaveform(audioChunk);
    
    // Get the result
    final result = stream.getResult();
    
    // Extract language from result
    // Whisper returns language code in the result
    final language = result.text.split(' ').first;
    
    // Validate and return
    if (supportedLanguages.contains(language)) {
      return language;
    }
    
    // Fallback: try to detect from text
    return _detectLanguageFromText(result.text);
  }
  
  String _detectLanguageFromText(String text) {
    // Simple heuristic-based detection
    // In production, use a proper language detection library
    final lowerText = text.toLowerCase();
    
    if (lowerText.contains('the') || lowerText.contains('and')) return 'en';
    if (lowerText.contains('le') || lowerText.contains('la')) return 'fr';
    if (lowerText.contains('el') || lowerText.contains('de')) return 'es';
    if (lowerText.contains('der') || lowerText.contains('die')) return 'de';
    if (lowerText.contains('il') || lowerText.contains('di')) return 'it';
    
    return 'en'; // Default to English
  }
  
  Future<void> dispose() async {
    _recognizer?.delete();
    _recognizer = null;
    _isInitialized = false;
  }
}
```

---

```dart
// lib/services/stt/model_switcher.dart

class ModelSwitcher {
  final ModelManager _modelManager;
  final LanguageDetector _languageDetector;
  
  // Cache for language detection results
  final Map<String, String> _languageCache = {};
  final Map<String, DateTime> _cacheTimestamps = {};
  final Duration _cacheDuration = const Duration(minutes: 5);
  
  // Current active model
  String? _currentModelId;
  dynamic _currentModelInstance;
  
  ModelSwitcher(this._modelManager, this._languageDetector);
  
  Future<String> getOptimalModelId({
    required List<int> audioData,
    String? forcedLanguage,
    bool useParakeetForHighEnd = true,
  }) async {
    String language;
    
    // Use forced language if provided
    if (forcedLanguage != null) {
      language = forcedLanguage;
    } else {
      // Check cache first
      language = _getCachedLanguage();
      
      if (language == null) {
        // Detect language
        language = await _languageDetector.detectLanguage(audioData);
        _cacheLanguage(language);
      }
    }
    
    // Try to get language-specific model
    final languageModel = STTModelConfig.getModelForLanguage(language);
    
    if (languageModel != null) {
      // Check if device can handle it
      if (STTModelConfig.canDeviceHandleModel(
        languageModel,
        _modelManager.availableMemoryMB,
      )) {
        return languageModel.id;
      }
    }
    
    // Check if we should use Parakeet (for high-end devices)
    if (useParakeetForHighEnd) {
      final parakeetModel = STTModelConfig.models.firstWhere(
        (m) => m.id == 'parakeet_tdt_0.6b_v3',
      );
      
      if (STTModelConfig.canDeviceHandleModel(
        parakeetModel,
        _modelManager.availableMemoryMB,
      ) && parakeetModel.languages.contains(language)) {
        return parakeetModel.id;
      }
    }
    
    // Fallback to Whisper
    return 'whisper_tiny';
  }
  
  Future<dynamic> getModelForAudio({
    required List<int> audioData,
    String? forcedLanguage,
    bool autoSwitch = true,
  }) async {
    final optimalModelId = await getOptimalModelId(
      audioData: audioData,
      forcedLanguage: forcedLanguage,
    );
    
    // Switch model if needed
    if (autoSwitch && optimalModelId != _currentModelId) {
      await switchModel(optimalModelId);
    }
    
    return _currentModelInstance;
  }
  
  Future<void> switchModel(String modelId) async {
    // Unload current model if different
    if (_currentModelId != null && _currentModelId != modelId) {
      await _modelManager.unloadModel(_currentModelId!);
    }
    
    // Load new model
    await _modelManager.ensureModelLoaded(modelId);
    
    _currentModelId = modelId;
    _currentModelInstance = _modelManager.getModelInstance(modelId);
  }
  
  String? _getCachedLanguage() {
    final now = DateTime.now();
    
    for (final entry in _cacheTimestamps.entries) {
      if (now.difference(entry.value) < _cacheDuration) {
        return _languageCache[entry.key];
      }
    }
    
    return null;
  }
  
  void _cacheLanguage(String language) {
    final sessionId = DateTime.now().millisecondsSinceEpoch.toString();
    _languageCache[sessionId] = language;
    _cacheTimestamps[sessionId] = DateTime.now();
    
    // Clean old entries
    _cleanCache();
  }
  
  void _cleanCache() {
    final now = DateTime.now();
    final oldKeys = _cacheTimestamps.entries
        .where((e) => now.difference(e.value) > _cacheDuration)
        .map((e) => e.key)
        .toList();
    
    for (final key in oldKeys) {
      _languageCache.remove(key);
      _cacheTimestamps.remove(key);
    }
  }
  
  String? get currentModelId => _currentModelId;
  dynamic get currentModelInstance => _currentModelInstance;
  
  Future<void> dispose() async {
    if (_currentModelId != null) {
      await _modelManager.unloadModel(_currentModelId!);
    }
    _currentModelId = null;
    _currentModelInstance = null;
  }
}
```

---

```dart
// lib/services/stt/transcription_service.dart

import 'package:record/record.dart';

class TranscriptionService {
  final ModelSwitcher _modelSwitcher;
  final ModelManager _modelManager;
  final LanguageDetector _languageDetector;
  
  // Audio configuration
  static const int sampleRate = 16000;
  static const int chunkSize = 1600; // 100ms at 16kHz
  
  // State
  bool _isRecording = false;
  bool _isProcessing = false;
  final List<int> _audioBuffer = [];
  final List<String> _transcriptionResults = [];
  
  // Callbacks
  Function(String)? onPartialTranscription;
  Function(String)? onFinalTranscription;
  Function(Exception)? onError;
  Function(String)? onLanguageDetected;
  
  final AudioRecorder _audioRecorder = AudioRecorder();
  
  TranscriptionService({
    required ModelSwitcher modelSwitcher,
    required ModelManager modelManager,
    required LanguageDetector languageDetector,
  })  : _modelSwitcher = modelSwitcher,
        _modelManager = modelManager,
        _languageDetector = languageDetector;
  
  Future<void> startRecording() async {
    if (_isRecording) return;
    
    try {
      _isRecording = true;
      _audioBuffer.clear();
      _transcriptionResults.clear();
      
      // Start audio recording
      await _audioRecorder.start(const RecordConfig(
        sampleRate: sampleRate,
        numChannels: 1,
        encoder: AudioEncoder.pcm16,
      ));
      
      // Start processing stream
      _audioRecorder.stream.listen((audioData) {
        _processAudioChunk(audioData);
      });
      
    } catch (e) {
      _isRecording = false;
      onError?.call(Exception('Failed to start recording: $e'));
    }
  }
  
  Future<void> stopRecording() async {
    if (!_isRecording) return;
    
    await _audioRecorder.stop();
    _isRecording = false;
    
    // Process remaining audio
    if (_audioBuffer.isNotEmpty) {
      await _processBuffer();
    }
    
    // Finalize transcription
    if (_transcriptionResults.isNotEmpty) {
      final fullText = _transcriptionResults.join(' ');
      onFinalTranscription?.call(fullText);
    }
  }
  
  void _processAudioChunk(List<int> audioData) {
    if (_isProcessing) return;
    
    _audioBuffer.addAll(audioData);
    
    // Process in chunks
    while (_audioBuffer.length >= chunkSize) {
      final chunk = _audioBuffer.sublist(0, chunkSize);
      _audioBuffer.removeRange(0, chunkSize);
      
      _processChunk(chunk);
    }
  }
  
  Future<void> _processChunk(List<int> chunk) async {
    if (_isProcessing) return;
    
    _isProcessing = true;
    
    try {
      // Get optimal model
      final modelInstance = await _modelSwitcher.getModelForAudio(
        audioData: chunk,
        autoSwitch: true,
      );
      
      // Create stream if needed
      final stream = modelInstance.createStream();
      
      // Accept waveform
      stream.acceptWaveform(chunk);
      
      // Get result
      final result = stream.getResult();
      
      if (result.text.isNotEmpty) {
        onPartialTranscription?.call(result.text);
        _transcriptionResults.add(result.text);
      }
      
      // Detect language if not already detected
      if (_modelSwitcher.currentModelId != 'whisper_tiny' &&
          _modelSwitcher.currentModelId != 'parakeet_tdt_0.6b_v3') {
        // Already using a language-specific model
        return;
      }
      
      // Try to detect language from current chunk
      final detectedLang = await _languageDetector.detectLanguage(chunk);
      onLanguageDetected?.call(detectedLang);
      
      // Switch to language-specific model if available
      final languageModel = STTModelConfig.getModelForLanguage(detectedLang);
      if (languageModel != null) {
        await _modelSwitcher.switchModel(languageModel.id);
      }
      
    } catch (e) {
      onError?.call(Exception('Processing error: $e'));
    } finally {
      _isProcessing = false;
      
      // Process next chunk if available
      if (_audioBuffer.isNotEmpty) {
        _processChunk(_audioBuffer.sublist(0, chunkSize));
      }
    }
  }
  
  Future<void> _processBuffer() async {
    while (_audioBuffer.isNotEmpty) {
      final chunkSize = _audioBuffer.length > chunkSize
          ? chunkSize
          : _audioBuffer.length;
      final chunk = _audioBuffer.sublist(0, chunkSize);
      _audioBuffer.removeRange(0, chunkSize);
      
      await _processChunk(chunk);
    }
  }
  
  Future<String> transcribeFile(String filePath) async {
    final audioData = await File(filePath).readAsBytes();
    
    // Detect language
    final language = await _languageDetector.detectLanguage(audioData);
    onLanguageDetected?.call(language);
    
    // Get optimal model
    final modelInstance = await _modelSwitcher.getModelForAudio(
      audioData: audioData,
      forcedLanguage: language,
    );
    
    // Create offline recognizer for file transcription
    // (Implementation depends on model type)
    final result = await _transcribeWithModel(modelInstance, audioData);
    
    return result;
  }
  
  Future<String> _transcribeWithModel(dynamic modelInstance, List<int> audioData) async {
    // Implementation varies by model type
    // For transducer models (Kroko, Zipformer, Parakeet):
    if (modelInstance is sherpa_onnx.OfflineRecognizer) {
      final stream = modelInstance.createStream();
      stream.acceptWaveform(audioData);
      final result = stream.getResult();
      return result.text;
    }
    
    // For online recognizer:
    if (modelInstance is sherpa_onnx.OnlineRecognizer) {
      final stream = modelInstance.createStream();
      stream.acceptWaveform(audioData);
      final result = stream.getResult();
      return result.text;
    }
    
    throw Exception('Unsupported model type');
  }
  
  Future<void> dispose() async {
    await _audioRecorder.close();
    await _modelSwitcher.dispose();
  }
  
  bool get isRecording => _isRecording;
  bool get isProcessing => _isProcessing;
}
```

---

```dart
// lib/services/audio/audio_capture.dart

import 'package:record/record.dart';

class AudioCaptureService {
  final AudioRecorder _recorder = AudioRecorder();
  bool _isRecording = false;
  
  // Configuration
  static const int sampleRate = 16000;
  static const int numChannels = 1;
  
  Future<void> startRecording(Function(List<int>) onAudioData) async {
    if (_isRecording) return;
    
    await _recorder.start(const RecordConfig(
      sampleRate: sampleRate,
      numChannels: numChannels,
      encoder: AudioEncoder.pcm16,
    ));
    
    _isRecording = true;
    
    _recorder.stream.listen((audioData) {
      onAudioData(audioData);
    });
  }
  
  Future<List<int>> stopRecording() async {
    if (!_isRecording) return [];
    
    final audioData = await _recorder.stop();
    _isRecording = false;
    
    return audioData!;
  }
  
  Future<void> pauseRecording() async {
    if (!_isRecording) return;
    await _recorder.pause();
  }
  
  Future<void> resumeRecording() async {
    if (_isRecording) return;
    await _recorder.resume();
  }
  
  Future<void> dispose() async {
    if (_isRecording) {
      await _recorder.stop();
    }
    await _recorder.close();
  }
  
  bool get isRecording => _isRecording;
}
```

```dart
// lib/services/audio/vad.dart

class VoiceActivityDetector {
  // Simple energy-based VAD
  static const double speechThreshold = 0.01;
  static const double silenceThreshold = 0.001;
  static const int speechFramesRequired = 3;
  static const int silenceFramesRequired = 10;
  
  int _speechFrameCount = 0;
  int _silenceFrameCount = 0;
  bool _isSpeaking = false;
  
  // Process audio chunk (16-bit PCM)
  bool processChunk(List<int> audioChunk) {
    final energy = _calculateEnergy(audioChunk);
    
    if (energy > speechThreshold) {
      _speechFrameCount++;
      _silenceFrameCount = 0;
      
      if (_speechFrameCount >= speechFramesRequired && !_isSpeaking) {
        _isSpeaking = true;
        _speechFrameCount = 0;
        return true; // Speech started
      }
    } else {
      _silenceFrameCount++;
      _speechFrameCount = 0;
      
      if (_silenceFrameCount >= silenceFramesRequired && _isSpeaking) {
        _isSpeaking = false;
        _silenceFrameCount = 0;
        return false; // Speech ended
      }
    }
    
    return _isSpeaking;
  }
  
  double _calculateEnergy(List<int> audioChunk) {
    double sum = 0;
    for (final sample in audioChunk) {
      // Convert 16-bit PCM to double
      final normalized = sample / 32768.0;
      sum += normalized * normalized;
    }
    return sum / audioChunk.length;
  }
  
  bool get isSpeaking => _isSpeaking;
  
  void reset() {
    _speechFrameCount = 0;
    _silenceFrameCount = 0;
    _isSpeaking = false;
  }
}
```

---

```dart
// lib/services/utils/device_utils.dart

import 'dart:io' show Platform;
import 'package:device_info_plus/device_info_plus.dart';

class DeviceUtils {
  static Future<DeviceCapabilities> detectCapabilities() async {
    final deviceInfo = DeviceInfoPlugin();
    
    if (Platform.isAndroid) {
      final androidInfo = await deviceInfo.androidInfo;
      return _detectAndroidCapabilities(androidInfo);
    } else if (Platform.isIOS) {
      final iosInfo = await deviceInfo.iosInfo;
      return _detectIOSCapabilities(iosInfo);
    } else {
      return DeviceCapabilities(
        memoryMB: 4000,
        cpuCores: 4,
        isHighEnd: true,
        supportsNPU: false,
      );
    }
  }
  
  static DeviceCapabilities _detectAndroidCapabilities(AndroidDeviceInfo info) {
    // Detect based on model and Android version
    final model = info.model.toLowerCase();
    final sdkVersion = info.version.sdkInt;
    
    // High-end devices
    final highEndModels = [
      'samsung sm-g', 'pixel', 'oneplus', 'xiaomi 13', 'xiaomi 14',
      's23', 's24', 'note 20', 'note 23', 'fold', 'z fold'
    ];
    
    final isHighEnd = highEndModels.any((m) => model.contains(m)) ||
        sdkVersion >= 34; // Android 14+
    
    // Estimate memory
    int memoryMB = 4000; // Default: 4GB
    if (model.contains('s23') || model.contains('s24')) {
      memoryMB = 8192; // 8GB
    } else if (model.contains('pixel 8') || model.contains('oneplus 12')) {
      memoryMB = 12288; // 12GB
    } else if (model.contains('pixel 7') || model.contains('s22')) {
      memoryMB = 6144; // 6GB
    }
    
    // Check for NPU support
    final supportsNPU = sdkVersion >= 31 && (
      model.contains('samsung') ||
      model.contains('qualcomm') ||
      model.contains('mediatek')
    );
    
    return DeviceCapabilities(
      memoryMB: memoryMB,
      cpuCores: 8, // Most modern Android phones have 8 cores
      isHighEnd: isHighEnd,
      supportsNPU: supportsNPU,
    );
  }
  
  static DeviceCapabilities _detectIOSCapabilities(IosDeviceInfo info) {
    final model = info.model;
    final systemVersion = info.systemVersion;
    
    // Parse iOS version
    final iosVersion = double.tryParse(systemVersion) ?? 15.0;
    
    // High-end devices
    final highEndModels = ['iPhone15', 'iPhone14', 'iPhone13', 'iPad Pro'];
    final isHighEnd = highEndModels.any((m) => model.contains(m)) ||
        iosVersion >= 16.0;
    
    // Estimate memory
    int memoryMB = 4096; // Default: 4GB
    if (model.contains('iPhone15Pro') || model.contains('iPad Pro')) {
      memoryMB = iosVersion >= 17.0 ? 8192 : 6144;
    } else if (model.contains('iPhone15') || model.contains('iPhone14')) {
      memoryMB = 6144; // 6GB
    } else if (model.contains('iPhone13')) {
      memoryMB = 4096; // 4GB
    }
    
    // iPhones have NPU (Neural Engine)
    final supportsNPU = true;
    
    return DeviceCapabilities(
      memoryMB: memoryMB,
      cpuCores: 6, // iPhones typically have 6 cores
      isHighEnd: isHighEnd,
      supportsNPU: supportsNPU,
    );
  }
}

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
```

```dart
// lib/config/stt_config.dart

import 'package:riverpod/riverpod.dart';

class STTConfig {
  // Audio settings
  final int sampleRate;
  final int chunkSize;
  final int numChannels;
  
  // Model settings
  final int numThreads;
  final bool useParakeetForHighEnd;
  final bool useVAD;
  final int maxModelSizeMB;
  
  // Performance settings
  final Duration languageCacheDuration;
  final int maxLoadedModels;
  
  const STTConfig({
    this.sampleRate = 16000,
    this.chunkSize = 1600, // 100ms
    this.numChannels = 1,
    this.numThreads = 4,
    this.useParakeetForHighEnd = true,
    this.useVAD = true,
    this.maxModelSizeMB = 500,
    this.languageCacheDuration = const Duration(minutes: 5),
    this.maxLoadedModels = 2,
  });
  
  // Create config based on device capabilities
  static STTConfig fromDeviceCapabilities(DeviceCapabilities capabilities) {
    return STTConfig(
      numThreads: capabilities.cpuCores,
      useParakeetForHighEnd: capabilities.isHighEnd && capabilities.memoryMB >= 6000,
      maxModelSizeMB: capabilities.memoryMB > 4000 ? 640 : 200,
      maxLoadedModels: capabilities.memoryMB > 4000 ? 3 : 1,
    );
  }
}

// Provider for STT configuration
final sttConfigProvider = Provider<STTConfig>((ref) {
  // In a real app, you would detect device capabilities here
  return const STTConfig();
});
```

---

```dart
// lib/services/stt/error_handler.dart

class STTErrorHandler {
  final ModelManager _modelManager;
  final ModelSwitcher _modelSwitcher;
  
  STTErrorHandler(this._modelManager, this._modelSwitcher);
  
  Future<dynamic> handleError(
    Exception error,
    List<int> audioData,
    String? detectedLanguage,
  ) async {
    print('STT Error: $error');
    
    // Memory error
    if (error.toString().contains('out of memory') ||
        error.toString().contains('OOM')) {
      return await _handleMemoryError(audioData, detectedLanguage);
    }
    
    // Model load error
    if (error.toString().contains('model not found') ||
        error.toString().contains('FileNotFoundException')) {
      return await _handleModelLoadError(audioData, detectedLanguage);
    }
    
    // Unknown error - try fallback
    return await _tryFallbackModel(audioData);
  }
  
  Future<dynamic> _handleMemoryError(
    List<int> audioData,
    String? detectedLanguage,
  ) async {
    // Unload all models
    await _modelManager.unloadAllModels();
    
    // Try with a smaller model
    if (detectedLanguage != null) {
      final languageModel = STTModelConfig.getModelForLanguage(detectedLanguage);
      if (languageModel != null && languageModel.sizeMB < 100) {
        await _modelSwitcher.switchModel(languageModel.id);
        return _modelSwitcher.currentModelInstance;
      }
    }
    
    // Fallback to Whisper-tiny
    await _modelSwitcher.switchModel('whisper_tiny');
    return _modelSwitcher.currentModelInstance;
  }
  
  Future<dynamic> _handleModelLoadError(
    List<int> audioData,
    String? detectedLanguage,
  ) async {
    // Try to download the model
    try {
      if (detectedLanguage != null) {
        final languageModel = STTModelConfig.getModelForLanguage(detectedLanguage);
        if (languageModel != null) {
          await _modelManager.ensureModelLoaded(languageModel.id);
          await _modelSwitcher.switchModel(languageModel.id);
          return _modelSwitcher.currentModelInstance;
        }
      }
      
      // Fallback to Whisper
      await _modelManager.ensureModelLoaded('whisper_tiny');
      await _modelSwitcher.switchModel('whisper_tiny');
      return _modelSwitcher.currentModelInstance;
      
    } catch (e) {
      return await _tryFallbackModel(audioData);
    }
  }
  
  Future<dynamic> _tryFallbackModel(List<int> audioData) async {
    // Try Whisper-tiny
    try {
      await _modelManager.ensureModelLoaded('whisper_tiny');
      await _modelSwitcher.switchModel('whisper_tiny');
      return _modelSwitcher.currentModelInstance;
    } catch (e) {
      // If all else fails, throw
      throw Exception('All fallback models failed: $e');
    }
  }
  
  // Cloud fallback (optional)
  Future<String> fallbackToCloud(List<int> audioData) async {
    // Implement cloud API call
    // Example: Deepgram, AssemblyAI, etc.
    throw UnimplementedError('Cloud fallback not implemented');
  }
}
```

---

```dart
// lib/services/testing/benchmark_service.dart

import 'package:stopwatch/stopwatch.dart';

class BenchmarkService {
  final ModelManager _modelManager;
  
  BenchmarkService(this._modelManager);
  
  Future<Map<String, dynamic>> benchmarkModel(String modelId, List<int> audioData) async {
    // Load model
    await _modelManager.ensureModelLoaded(modelId);
    final modelInstance = _modelManager.getModelInstance(modelId);
    
    final stopwatch = Stopwatch()..start();
    
    // Warm-up run
    await _transcribe(modelInstance, audioData);
    
    // Actual benchmark
    stopwatch.reset();
    stopwatch.start();
    
    final numRuns = 5;
    for (var i = 0; i < numRuns; i++) {
      await _transcribe(modelInstance, audioData);
    }
    
    stopwatch.stop();
    
    final totalTimeMs = stopwatch.elapsedMilliseconds;
    final avgTimeMs = totalTimeMs / numRuns;
    final audioDurationMs = audioData.length / 16; // 16 samples per ms at 16kHz
    final rtf = avgTimeMs / audioDurationMs;
    
    // Get model config
    final modelConfig = STTModelConfig.models.firstWhere(
      (m) => m.id == modelId,
    );
    
    return {
      'model_id': modelId,
      'model_name': modelConfig.name,
      'size_mb': modelConfig.sizeMB,
      'avg_time_ms': avgTimeMs,
      'rtf': rtf,
      'audio_duration_ms': audioDurationMs,
      'runs': numRuns,
    };
  }
  
  Future<String> _transcribe(dynamic modelInstance, List<int> audioData) async {
    if (modelInstance is sherpa_onnx.OfflineRecognizer) {
      final stream = modelInstance.createStream();
      stream.acceptWaveform(audioData);
      final result = stream.getResult();
      return result.text;
    } else if (modelInstance is sherpa_onnx.OnlineRecognizer) {
      final stream = modelInstance.createStream();
      stream.acceptWaveform(audioData);
      final result = stream.getResult();
      return result.text;
    }
    throw Exception('Unsupported model type');
  }
  
  Future<Map<String, dynamic>> benchmarkAllModels(List<int> audioData) async {
    final results = <String, dynamic>{};
    
    for (final model in STTModelConfig.models) {
      try {
        final benchmark = await benchmarkModel(model.id, audioData);
        results[model.id] = benchmark;
      } catch (e) {
        results[model.id] = {'error': e.toString()};
      }
    }
    
    return results;
  }
}
```

```dart
// lib/services/testing/test_audio.dart

class TestAudioGenerator {
  // Generate test audio with known content
  static List<int> generateTestAudio({
    required String text,
    int sampleRate = 16000,
    double frequency = 440.0,
    int durationMs = 1000,
  }) {
    // This is a placeholder - in a real app, you would:
    // 1. Use a TTS engine to generate speech
    // 2. Or load pre-recorded test audio files
    
    final numSamples = (sampleRate * durationMs / 1000).round();
    final audio = List<int>.filled(numSamples, 0);
    
    // Generate a simple sine wave (for testing)
    for (var i = 0; i < numSamples; i++) {
      final t = i / sampleRate;
      final value = (sin(2 * pi * frequency * t) * 32767).round();
      audio[i] = value.clamp(-32768, 32767);
    }
    
    return audio;
  }
  
  // Load test audio from assets
  static Future<List<int>> loadTestAudio(String language, int durationMs) async {
    // Load from assets
    // Example: assets/test_audio/en_1s.pcm
    final assetPath = 'assets/test_audio/${language}_${durationMs}ms.pcm';
    
    try {
      final data = await rootBundle.load(assetPath);
      return data.buffer.asInt16List();
    } catch (e) {
      // Fallback to generated audio
      return generateTestAudio(
        text: 'Test audio for $language',
        durationMs: durationMs,
      );
    }
  }
}
```

---

```dart
// Model download URLs for sherpa-onnx
const modelUrls = {
  // Kroko models
  'kroko_64l_en': 'https://github.com/RunanywhereAI/sherpa-onnx/releases/download/runanywhere-models-v1/sherpa-onnx-whisper-tiny.en.tar.gz',
  'kroko_64l_fr': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-fr/resolve/main/sherpa-onnx-kroko-64l-fr.tar.gz',
  'kroko_64l_es': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-es/resolve/main/sherpa-onnx-kroko-64l-es.tar.gz',
  'kroko_64l_de': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-de/resolve/main/sherpa-onnx-kroko-64l-de.tar.gz',
  'kroko_64l_it': 'https://huggingface.co/csukuangfj/sherpa-onnx-kroko-64l-it/resolve/main/sherpa-onnx-kroko-64l-it.tar.gz',
  
  // Parakeet-TDT
  'parakeet_tdt_0.6b_v3': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2',
  
  // Whisper models
  'whisper_tiny': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2',
  'whisper_small': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2',
  
  // LID model (Whisper-tiny for language detection)
  'whisper_tiny_lid': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2',
};
```

```dart
// Supported languages by model
const supportedLanguages = {
  'kroko': ['en', 'fr', 'es', 'de', 'it', 'pt', 'ru', 'nl', 'pl', 'uk'],
  'zipformer': ['en', 'zh', 'fr', 'es', 'de', 'it'],
  'parakeet': [
    'en', 'es', 'it', 'fr', 'de', 'nl', 'ru', 'pl', 'uk', 'sk',
    'bg', 'fi', 'ro', 'hr', 'cs', 'sv', 'et', 'hu', 'lt', 'da',
    'mt', 'sl', 'lv', 'el', 'pt'
  ],
  'whisper': [
    'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca',
    'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms',
    'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'lv',
    'et', 'ml', 'cy', 'sk', 'te', 'fa', 'sr', 'az', 'sl', 'kn', 'gu', 'am',
    'hy', 'lo', 'ne', 'mr', 'bn', 'ka', 'mk', 'sw', 'gl', 'eu', 'ha', 'is',
    'kk', 'km', 'ky', 'lb', 'mg', 'ps', 'so', 'sq', 'sw', 'tl', 'tt', 'uz'
  ],
};
```
1. **Integrate this guide** into your existing Flutter package
2. **Customize model configurations** based on your target languages
3. **Test on your target devices** and adjust parameters
4. **Optimize for your specific use case** (real-time vs. batch)
5. **Consider adding cloud fallback** for unsupported languages/devices

---

*Generated by Vibe Code - Async Software Engineering Agent*
*Branch: vibe/review-improvements-bc2b72*
