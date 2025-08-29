import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home_screen.dart';
import 'screens/evaluation_viewer_screen.dart';
import 'screens/human_review_screen.dart';
import 'providers/evaluation_provider.dart';
import 'providers/review_provider.dart';

void main() {
  runApp(const VisionFlowDashboardsApp());
}

class VisionFlowDashboardsApp extends StatelessWidget {
  const VisionFlowDashboardsApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => EvaluationProvider()),
        ChangeNotifierProvider(create: (_) => ReviewProvider()),
      ],
      child: MaterialApp(
        title: 'VisionFlow Dashboards',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          brightness: Brightness.light,
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF667eea),
            brightness: Brightness.light,
          ),
        ),
        darkTheme: ThemeData(
          primarySwatch: Colors.blue,
          brightness: Brightness.dark,
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF667eea),
            brightness: Brightness.dark,
          ),
        ),
        themeMode: ThemeMode.system,
        home: const HomeScreen(),
        routes: {
          '/evaluation': (context) => const EvaluationViewerScreen(),
          '/review': (context) => const HumanReviewScreen(),
        },
      ),
    );
  }
}
