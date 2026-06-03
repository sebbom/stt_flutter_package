import 'package:flutter/material.dart';
import 'screens/model_selection_screen.dart';

void main() {
  runApp(const SttExampleApp());
}

class SttExampleApp extends StatelessWidget {
  const SttExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'STT Flutter',
      theme: ThemeData(
        colorSchemeSeed: Colors.blue,
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      home: const ModelSelectionScreen(),
    );
  }
}
