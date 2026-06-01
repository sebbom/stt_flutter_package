import 'dart:typed_data';

int argmax(List<dynamic> logits) {
  int best = 0;
  double bestVal = double.negativeInfinity;
  for (int i = 0; i < logits.length; i++) {
    final v = (logits[i] as num).toDouble();
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

Float32List transposeMel(Float64List mel, int nMels, int totalFrames, int chunkOffset, int chunkSize) {
  final actual = (totalFrames - chunkOffset).clamp(0, chunkSize);
  final out = Float32List(nMels * chunkSize);
  for (int t = 0; t < actual; t++) {
    final srcBase = (chunkOffset + t) * nMels;
    for (int m = 0; m < nMels; m++) {
      out[m * chunkSize + t] = mel[srcBase + m];
    }
  }
  return out;
}
