# stt_flutter_example

Demo app for `stt_flutter` — fully local on-device speech-to-text.

Supports **Android, iOS, Linux** (and macOS, Windows, Web with the same Dart code).

## Usage

1. Launch the app → model selection screen (models grouped by engine type)
2. Download a model (or use one already downloaded on the device)
3. Tap a downloaded card to open the transcription screen
4. Pick a language mode:
   - **Auto** — engine decides (auto-detect)
   - **Default** — uses the language set on `loadModel(defaultLanguage: …)`; tap the tune icon to change it
   - **Force** — overrides the default for the next call
5. Toggle the **Language detector (SLI)** switch to populate `result.lang` when the engine doesn't return one (uses Whisper-tiny SLI on top of the encoder/decoder files you already downloaded)
6. Choose an input:
   - **Sample** — bundled `hello_en.wav`
   - **Pick file** — any local audio file (WAV / MP3 / M4A / FLAC / OGG / Opus)
   - **Mic** — live recording with VAD; tap Mic again (now red) to stop
7. Result card shows `text`, `lang`, `conf`, `audio`, `infer`, `mode` chips; full history is kept below

## Models

All models are downloaded at runtime from HuggingFace / GitHub on first use.
No bundled (embedded) model files are included in the example app.

## Build

### Android

```bash
cd example
flutter pub get
flutter build apk --debug          # debug APK
flutter build apk --release        # release APK (will need signing)
flutter install                    # install on connected device
```

Requires Android SDK + toolchain. The example pins `compileSdk = 36` to satisfy `flutter_plugin_android_lifecycle`.

### iOS

```bash
cd example/ios
pod install
cd ..
flutter build ios --debug --no-codesign
flutter build ios --release         # needs Apple Developer signing
```

Requires macOS with Xcode 15+ and CocoaPods.

The example `Info.plist` declares `NSMicrophoneUsageDescription` and `NSSpeechRecognitionUsageDescription` so iOS prompts the user on first mic access. Swift Package Manager (default in Flutter 3.24+) is used for `permission_handler` — no `Podfile` post-install block is required; the SPM config auto-detects the plist keys.

Deployment target is iOS 13.0 (compatible with `record_ios 2.x`, `file_picker 10.x`, `permission_handler 11.x`).

### Linux

```bash
cd example
flutter pub get
flutter build linux
./build/linux/x64/release/bundle/stt_flutter_example
```

Requires:

- Flutter 3.7+ with Linux desktop enabled (`flutter config --enable-linux-desktop`)
- `clang`, `cmake`, `ninja-build`, `libgtk-3-dev`, `liblzma-dev`
- For mic capture: `libpulse-dev` (or `libasound2-dev` for ALSA) and a running audio daemon
- For file picker: `xdg-desktop-portal` + a portal implementation (e.g. `xdg-desktop-portal-gtk` or `xdg-desktop-portal-kde`)

The example uses `permission_handler` for Android/iOS. On Linux, `permission_handler` does not support the microphone, so `_ensureMicPermission` falls back to `record.hasPermission(request: true)` automatically — no app changes needed.

### macOS / Windows / Web

The same `lib/` Dart code is portable. To scaffold a desktop or web target:

```bash
cd example
flutter create --platforms=macos .       # or: windows, web
flutter build macos
flutter build windows
flutter build web
```

The Info.plist for macOS already mirrors the iOS one (both share `Runner/Info.plist`).
For Windows, mic permission is granted at install time by the OS — no runtime prompt.

## Permissions

| Platform | What is needed |
| --- | --- |
| Android | `RECORD_AUDIO` (runtime), `READ_MEDIA_AUDIO` (API 33+), `READ_EXTERNAL_STORAGE` (≤ API 32) — all already declared in `android/app/src/main/AndroidManifest.xml` |
| iOS     | `NSMicrophoneUsageDescription` in `Info.plist` — already added |
| Linux   | mic is opened via `record_linux`; `permission_handler` falls back gracefully on this platform |
| macOS   | `NSMicrophoneUsageDescription` (same as iOS) |
| Windows | mic is OS-granted at install time |
| Web     | mic is requested via the browser's `getUserMedia()` automatically |
