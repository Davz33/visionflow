import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/evaluation_provider.dart';
import '../widgets/video_player_widget.dart';
import '../widgets/evaluation_metrics_widget.dart';

class EvaluationViewerScreen extends StatefulWidget {
  const EvaluationViewerScreen({super.key});

  @override
  State<EvaluationViewerScreen> createState() => _EvaluationViewerScreenState();
}

class _EvaluationViewerScreenState extends State<EvaluationViewerScreen> {
  @override
  void initState() {
    super.initState();
    // Load evaluations when screen initializes
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<EvaluationProvider>().loadEvaluations();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('📊 Video Evaluation Viewer'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        elevation: 2,
      ),
      body: Consumer<EvaluationProvider>(
        builder: (context, evaluationProvider, child) {
          if (evaluationProvider.isLoading) {
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Loading evaluations...'),
                ],
              ),
            );
          }

          if (evaluationProvider.error != null) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, size: 64, color: Colors.red),
                  const SizedBox(height: 16),
                  Text('Error: ${evaluationProvider.error}'),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () => evaluationProvider.loadEvaluations(),
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          }

          if (evaluationProvider.evaluations.isEmpty) {
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.video_library_outlined, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text('No evaluations found'),
                  SizedBox(height: 8),
                  Text('Start by running some video evaluations'),
                ],
              ),
            );
          }

          return Row(
            children: [
              // Left side - Video list
              Expanded(
                flex: 1,
                child: Container(
                  decoration: BoxDecoration(
                    border: Border(
                      right: BorderSide(
                        color: Theme.of(context).dividerColor,
                        width: 1,
                      ),
                    ),
                  ),
                  child: Column(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.surfaceVariant,
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.video_library),
                            const SizedBox(width: 8),
                                                           const Text(
                                 'Evaluations',
                                 style: TextStyle(
                                   fontSize: 24,
                                   fontWeight: FontWeight.bold,
                                 ),
                               ),
                            const Spacer(),
                            Text(
                              '${evaluationProvider.evaluations.length} videos',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: ListView.builder(
                          itemCount: evaluationProvider.evaluations.length,
                          itemBuilder: (context, index) {
                            final evaluation = evaluationProvider.evaluations[index];
                            return _buildEvaluationListItem(evaluation, index);
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              // Right side - Video player and metrics
              Expanded(
                flex: 2,
                child: Container(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      // Video player placeholder
                      Container(
                        height: 300,
                        decoration: BoxDecoration(
                          color: Colors.black87,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.play_circle_outline, size: 64, color: Colors.white54),
                              SizedBox(height: 16),
                              Text(
                                'Select a video to view',
                                style: TextStyle(color: Colors.white54),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                      // Metrics section
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Theme.of(context).colorScheme.surfaceVariant,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Evaluation Metrics',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              SizedBox(height: 16),
                              Text('Select a video to view detailed metrics'),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildEvaluationListItem(VideoEvaluation evaluation, int index) {
    final isSelected = false; // TODO: Implement selection state
    
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      color: isSelected 
          ? Theme.of(context).colorScheme.primaryContainer
          : Theme.of(context).colorScheme.surface,
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: _getDecisionColor(evaluation.decision),
          child: Text(
            evaluation.id,
            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
          ),
        ),
        title: Text(
          evaluation.prompt,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Confidence: ${(evaluation.confidence * 100).toStringAsFixed(1)}%'),
            Text('Decision: ${evaluation.decisionLabel}'),
          ],
        ),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              'Score: ${(evaluation.averageScore * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            Text(
              evaluation.metadata['category'] ?? 'Unknown',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
        onTap: () {
          // TODO: Implement video selection
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Selected: ${evaluation.prompt}'),
              duration: const Duration(seconds: 1),
            ),
          );
        },
      ),
    );
  }

  Color _getDecisionColor(String decision) {
    switch (decision) {
      case 'auto_approve':
        return Colors.green;
      case 'queue_review':
        return Colors.orange;
      case 'flag_monitoring':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }
}
