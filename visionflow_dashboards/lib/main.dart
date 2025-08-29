import 'package:flutter/material.dart';

void main() {
  runApp(const VisionFlowApp());
}

class VisionFlowApp extends StatelessWidget {
  const VisionFlowApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VisionFlow Dashboard',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const DashboardHome(title: 'VisionFlow Dashboard'),
    );
  }
}

class DashboardHome extends StatefulWidget {
  const DashboardHome({super.key, required this.title});

  final String title;

  @override
  State<DashboardHome> createState() => _DashboardHomeState();
}

class _DashboardHomeState extends State<DashboardHome> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'Welcome to VisionFlow Dashboard',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            Text(
              'Advanced Video Generation & Evaluation Platform',
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}
