import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:stt_flutter_example/main.dart';

void main() {
  testWidgets('App boots and shows the model selection screen',
      (WidgetTester tester) async {
    await tester.pumpWidget(const SttExampleApp());
    await tester.pump();
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
