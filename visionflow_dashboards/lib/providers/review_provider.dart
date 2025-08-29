import 'package:flutter/foundation.dart';

class ReviewProvider extends ChangeNotifier {
  List<ReviewTask> _reviewTasks = [];
  bool _isLoading = false;
  String? _error;
  ReviewStats _stats = ReviewStats();

  List<ReviewTask> get reviewTasks => _reviewTasks;
  bool get isLoading => _isLoading;
  String? get error => _error;
  ReviewStats get stats => _stats;

  Future<void> loadReviewTasks() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // TODO: Implement API call to load review tasks
      await Future.delayed(const Duration(seconds: 1)); // Simulate API call
      
      // Mock data for now
      _reviewTasks = [
        ReviewTask(
          id: '1',
          videoId: 'video_001',
          prompt: 'A cat playing with a ball',
          priority: 'high',
          status: 'pending',
          assignedTo: 'reviewer_1',
          createdAt: DateTime.now().subtract(const Duration(hours: 2)),
        ),
        ReviewTask(
          id: '2',
          videoId: 'video_002',
          prompt: 'A sunset over mountains',
          priority: 'medium',
          status: 'in_progress',
          assignedTo: 'reviewer_2',
          createdAt: DateTime.now().subtract(const Duration(hours: 1)),
        ),
        ReviewTask(
          id: '3',
          videoId: 'video_003',
          prompt: 'A car driving on a highway',
          priority: 'low',
          status: 'completed',
          assignedTo: 'reviewer_1',
          createdAt: DateTime.now().subtract(const Duration(hours: 3)),
        ),
      ];

      _updateStats();
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void _updateStats() {
    _stats = ReviewStats(
      total: _reviewTasks.length,
      pending: _reviewTasks.where((task) => task.status == 'pending').length,
      inProgress: _reviewTasks.where((task) => task.status == 'in_progress').length,
      completed: _reviewTasks.where((task) => task.status == 'completed').length,
      highPriority: _reviewTasks.where((task) => task.priority == 'high').length,
    );
  }

  void addReviewTask(ReviewTask task) {
    _reviewTasks.add(task);
    _updateStats();
    notifyListeners();
  }

  void updateReviewTask(String id, ReviewTask updatedTask) {
    final index = _reviewTasks.indexWhere((t) => t.id == id);
    if (index != -1) {
      _reviewTasks[index] = updatedTask;
      _updateStats();
      notifyListeners();
    }
  }

  void deleteReviewTask(String id) {
    _reviewTasks.removeWhere((t) => t.id == id);
    _updateStats();
    notifyListeners();
  }

  void assignTask(String taskId, String reviewerId) {
    final task = _reviewTasks.firstWhere((t) => t.id == taskId);
    if (task != null) {
      task.assignedTo = reviewerId;
      task.status = 'in_progress';
      _updateStats();
      notifyListeners();
    }
  }
}

class ReviewTask {
  final String id;
  final String videoId;
  final String prompt;
  final String priority;
  String status;
  String assignedTo;
  final DateTime createdAt;
  DateTime? completedAt;

  ReviewTask({
    required this.id,
    required this.videoId,
    required this.prompt,
    required this.priority,
    required this.status,
    required this.assignedTo,
    required this.createdAt,
    this.completedAt,
  });

  String get priorityLabel {
    switch (priority) {
      case 'high':
        return 'High';
      case 'medium':
        return 'Medium';
      case 'low':
        return 'Low';
      default:
        return 'Unknown';
    }
  }

  String get statusLabel {
    switch (status) {
      case 'pending':
        return 'Pending';
      case 'in_progress':
        return 'In Progress';
      case 'completed':
        return 'Completed';
      default:
        return 'Unknown';
    }
  }
}

class ReviewStats {
  final int total;
  final int pending;
  final int inProgress;
  final int completed;
  final int highPriority;

  ReviewStats({
    this.total = 0,
    this.pending = 0,
    this.inProgress = 0,
    this.completed = 0,
    this.highPriority = 0,
  });

  double get completionRate {
    if (total == 0) return 0.0;
    return completed / total;
  }
}
