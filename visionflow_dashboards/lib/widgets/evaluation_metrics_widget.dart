import 'package:flutter/material.dart';

class EvaluationMetricsWidget extends StatelessWidget {
  final Map<String, double>? scores;
  final double? confidence;
  final String? decision;

  const EvaluationMetricsWidget({
    super.key,
    this.scores,
    this.confidence,
    this.decision,
  });

  @override
  Widget build(BuildContext context) {
    if (scores == null) {
      return const Center(
        child: Text('No metrics available'),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Row(
          children: [
            const Icon(Icons.analytics),
            const SizedBox(width: 8),
            const Text(
              'Evaluation Metrics',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Spacer(),
            if (confidence != null)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: _getConfidenceColor(confidence!),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(
                  '${(confidence! * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(height: 16),

        // Decision badge
        if (decision != null) ...[
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: _getDecisionColor(decision!),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              _getDecisionLabel(decision!),
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const SizedBox(height: 24),
        ],

        // Scores grid
        Expanded(
          child: GridView.builder(
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 16,
              mainAxisSpacing: 16,
              childAspectRatio: 2.5,
            ),
            itemCount: scores!.length,
            itemBuilder: (context, index) {
              final metric = scores!.keys.elementAt(index);
              final score = scores!.values.elementAt(index);
              
              return Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        metric,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 8),
                      LinearProgressIndicator(
                        value: score,
                        backgroundColor: Colors.grey[300],
                        valueColor: AlwaysStoppedAnimation<Color>(
                          _getScoreColor(score),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '${(score * 100).toStringAsFixed(1)}%',
                        style: TextStyle(
                          color: _getScoreColor(score),
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.orange;
    return Colors.red;
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

  String _getDecisionLabel(String decision) {
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

  Color _getScoreColor(double score) {
    if (score >= 0.8) return Colors.green;
    if (score >= 0.6) return Colors.orange;
    return Colors.red;
  }
}
