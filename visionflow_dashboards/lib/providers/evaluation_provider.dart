import 'package:flutter/foundation.dart';

class EvaluationProvider extends ChangeNotifier {
  List<VideoEvaluation> _evaluations = [];
  bool _isLoading = false;
  String? _error;

  List<VideoEvaluation> get evaluations => _evaluations;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> loadEvaluations() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // TODO: Implement API call to load evaluations
      await Future.delayed(const Duration(seconds: 2)); // Simulate API call
      
      // Mock data for now
      _evaluations = [
        VideoEvaluation(
          id: '1',
          videoUrl: 'https://example.com/video1.mp4',
          prompt: 'A cat playing with a ball',
          scores: {
            'LPIPS': 0.85,
            'FVMD': 0.92,
            'CLIP': 0.78,
            'ETVA': 0.89,
          },
          confidence: 0.87,
          decision: 'auto_approve',
          metadata: {
            'duration': '10s',
            'resolution': '1920x1080',
            'category': 'animals',
          },
        ),
        VideoEvaluation(
          id: '2',
          videoUrl: 'https://example.com/video2.mp4',
          prompt: 'A sunset over mountains',
          scores: {
            'LPIPS': 0.72,
            'FVMD': 0.68,
            'CLIP': 0.81,
            'ETVA': 0.75,
          },
          confidence: 0.74,
          decision: 'queue_review',
          metadata: {
            'duration': '15s',
            'resolution': '1920x1080',
            'category': 'landscape',
          },
        ),
      ];
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void addEvaluation(VideoEvaluation evaluation) {
    _evaluations.add(evaluation);
    notifyListeners();
  }

  void updateEvaluation(String id, VideoEvaluation updatedEvaluation) {
    final index = _evaluations.indexWhere((e) => e.id == id);
    if (index != -1) {
      _evaluations[index] = updatedEvaluation;
      notifyListeners();
    }
  }

  void deleteEvaluation(String id) {
    _evaluations.removeWhere((e) => e.id == id);
    notifyListeners();
  }
}

class VideoEvaluation {
  final String id;
  final String videoUrl;
  final String prompt;
  final Map<String, double> scores;
  final double confidence;
  final String decision;
  final Map<String, String> metadata;

  VideoEvaluation({
    required this.id,
    required this.videoUrl,
    required this.prompt,
    required this.scores,
    required this.confidence,
    required this.decision,
    required this.metadata,
  });

  double get averageScore {
    if (scores.isEmpty) return 0.0;
    return scores.values.reduce((a, b) => a + b) / scores.length;
  }

  String get decisionLabel {
    switch (decision) {
      case 'auto_approve':
        return 'Auto Approve';
      case 'queue_review':
        return 'Queue Review';
      case 'flag_monitoring':
        return 'Flag Monitoring';
      default:
        return 'Unknown';
    }
  }
}
