#!/usr/bin/env python3
"""
Human Review Workflow Service
Handles review queue management, reviewer assignments, and workflow state persistence.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from contextlib import contextmanager

@dataclass
class ReviewDecision:
    """Human review decision"""
    decision: str  # approved, rejected, flagged, needs_revision
    score: Optional[float]
    comments: str
    tags: List[str]
    reviewer_id: str
    timestamp: str
    confidence: float = 1.0

@dataclass
class ReviewItem:
    """Review queue item"""
    video_id: str
    filename: str
    original_evaluation: Dict[str, Any]
    priority: str  # high, medium, low
    status: str  # pending, in_review, completed, escalated
    assigned_reviewer: Optional[str]
    created_at: str
    due_date: Optional[str]
    review_decisions: List[ReviewDecision]
    consensus_score: Optional[float]
    final_decision: Optional[str]

class ReviewWorkflowService:
    """
    Service for managing human review workflow operations
    """
    
    def __init__(self, data_dir: str = "review_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.db_path = self.data_dir / "review_workflow.db"
        self.init_database()
        
        # Configuration
        self.config = {
            "max_review_time_hours": 24,
            "high_priority_sla_hours": 4,
            "consensus_threshold": 0.8,
            "escalation_score_diff": 0.3,
            "auto_approve_threshold": 0.9,
            "auto_reject_threshold": 0.2
        }
    
    def init_database(self):
        """Initialize SQLite database for review workflow"""
        with self.get_db_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS review_items (
                    video_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    original_evaluation TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    assigned_reviewer TEXT,
                    created_at TEXT NOT NULL,
                    due_date TEXT,
                    consensus_score REAL,
                    final_decision TEXT
                );
                
                CREATE TABLE IF NOT EXISTS review_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    score REAL,
                    comments TEXT,
                    tags TEXT,
                    reviewer_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (video_id) REFERENCES review_items (video_id)
                );
                
                CREATE TABLE IF NOT EXISTS reviewers (
                    reviewer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    specialization TEXT,
                    active BOOLEAN DEFAULT 1,
                    reviews_completed INTEGER DEFAULT 0,
                    avg_review_time_hours REAL DEFAULT 0,
                    agreement_rate REAL DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS workflow_metrics (
                    date TEXT PRIMARY KEY,
                    total_reviews INTEGER DEFAULT 0,
                    avg_review_time_hours REAL DEFAULT 0,
                    ai_human_agreement_rate REAL DEFAULT 0,
                    throughput_videos_per_day REAL DEFAULT 0,
                    escalation_rate REAL DEFAULT 0
                );
                
                CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status);
                CREATE INDEX IF NOT EXISTS idx_review_priority ON review_items(priority);
                CREATE INDEX IF NOT EXISTS idx_reviewer_decisions ON review_decisions(reviewer_id);
            """)

    @contextmanager
    def get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_review_items_from_evaluation(self, evaluation_results_path: str) -> int:
        """
        Create review items from evaluation results JSON
        Returns number of items created
        """
        with open(evaluation_results_path, 'r') as f:
            evaluation_data = json.load(f)
        
        created_count = 0
        current_time = datetime.now().isoformat()
        
        with self.get_db_connection() as conn:
            for result in evaluation_data.get('results', []):
                # Only create review items for videos that need review
                if not (result.get('requires_review') or result.get('decision') == 'queue_review'):
                    continue
                
                video_id = result['filename'].replace('.mp4', '')
                
                # Check if already exists
                existing = conn.execute(
                    "SELECT video_id FROM review_items WHERE video_id = ?",
                    (video_id,)
                ).fetchone()
                
                if existing:
                    continue
                
                # Calculate priority
                priority = self._calculate_priority(result)
                
                # Calculate due date based on priority
                hours_offset = {
                    'high': self.config['high_priority_sla_hours'],
                    'medium': 12,
                    'low': 24
                }[priority]
                
                due_date = (datetime.now() + timedelta(hours=hours_offset)).isoformat()
                
                # Insert review item
                conn.execute("""
                    INSERT INTO review_items 
                    (video_id, filename, original_evaluation, priority, status, 
                     created_at, due_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    result['filename'],
                    json.dumps(result),
                    priority,
                    'pending',
                    current_time,
                    due_date
                ))
                
                created_count += 1
        
        return created_count

    def _calculate_priority(self, evaluation_result: Dict[str, Any]) -> str:
        """Calculate review priority based on evaluation results"""
        score = evaluation_result.get('overall_score', 0.5)
        confidence = evaluation_result.get('confidence', 0.5)
        
        # High priority: very low scores or very low confidence
        if score < 0.3 or confidence < 0.5:
            return 'high'
        
        # Medium priority: moderate issues
        if score < 0.6 or confidence < 0.7:
            return 'medium'
        
        # Low priority: borderline cases
        return 'low'

    def get_review_queue(self, 
                        status_filter: Optional[str] = None,
                        priority_filter: Optional[str] = None,
                        reviewer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get review queue with optional filters"""
        query = """
            SELECT ri.*, 
                   COUNT(rd.id) as review_count,
                   GROUP_CONCAT(rd.decision) as decisions,
                   AVG(rd.score) as avg_human_score
            FROM review_items ri
            LEFT JOIN review_decisions rd ON ri.video_id = rd.video_id
            WHERE 1=1
        """
        params = []
        
        if status_filter:
            query += " AND ri.status = ?"
            params.append(status_filter)
        
        if priority_filter:
            query += " AND ri.priority = ?"
            params.append(priority_filter)
        
        if reviewer_id:
            query += " AND ri.assigned_reviewer = ?"
            params.append(reviewer_id)
        
        query += """
            GROUP BY ri.video_id
            ORDER BY 
                CASE ri.priority 
                    WHEN 'high' THEN 3 
                    WHEN 'medium' THEN 2 
                    ELSE 1 
                END DESC,
                ri.created_at ASC
        """
        
        with self.get_db_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
            return [{
                'video_id': row['video_id'],
                'filename': row['filename'],
                'original_evaluation': json.loads(row['original_evaluation']),
                'priority': row['priority'],
                'status': row['status'],
                'assigned_reviewer': row['assigned_reviewer'],
                'created_at': row['created_at'],
                'due_date': row['due_date'],
                'consensus_score': row['consensus_score'],
                'final_decision': row['final_decision'],
                'review_count': row['review_count'] or 0,
                'decisions': row['decisions'].split(',') if row['decisions'] else [],
                'avg_human_score': row['avg_human_score']
            } for row in rows]

    def assign_reviewer(self, video_id: str, reviewer_id: str) -> bool:
        """Assign a reviewer to a video"""
        with self.get_db_connection() as conn:
            result = conn.execute("""
                UPDATE review_items 
                SET assigned_reviewer = ?, status = 'in_review'
                WHERE video_id = ? AND status = 'pending'
            """, (reviewer_id, video_id))
            
            return result.rowcount > 0

    def submit_review(self, 
                     video_id: str,
                     reviewer_id: str,
                     decision: str,
                     score: Optional[float] = None,
                     comments: str = "",
                     tags: List[str] = None) -> bool:
        """Submit a review decision"""
        if tags is None:
            tags = []
        
        timestamp = datetime.now().isoformat()
        
        with self.get_db_connection() as conn:
            # Insert review decision
            conn.execute("""
                INSERT INTO review_decisions 
                (video_id, decision, score, comments, tags, reviewer_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id, decision, score, comments, 
                json.dumps(tags), reviewer_id, timestamp
            ))
            
            # Update reviewer stats
            conn.execute("""
                UPDATE reviewers 
                SET reviews_completed = reviews_completed + 1
                WHERE reviewer_id = ?
            """, (reviewer_id,))
            
            # Check if consensus reached or final decision needed
            self._update_consensus(conn, video_id)
            
            return True

    def _update_consensus(self, conn, video_id: str):
        """Update consensus score and final decision for a video"""
        # Get all review decisions for this video
        decisions = conn.execute("""
            SELECT decision, score, reviewer_id
            FROM review_decisions
            WHERE video_id = ?
            ORDER BY timestamp DESC
        """, (video_id,)).fetchall()
        
        if not decisions:
            return
        
        # Get original AI evaluation
        original_eval = conn.execute("""
            SELECT original_evaluation 
            FROM review_items 
            WHERE video_id = ?
        """, (video_id,)).fetchone()
        
        if not original_eval:
            return
        
        eval_data = json.loads(original_eval['original_evaluation'])
        ai_score = eval_data.get('overall_score', 0.5)
        
        # Calculate consensus
        human_scores = [d['score'] for d in decisions if d['score'] is not None]
        decisions_list = [d['decision'] for d in decisions]
        
        consensus_score = None
        final_decision = None
        status = 'completed'
        
        if human_scores:
            consensus_score = sum(human_scores) / len(human_scores)
            
            # Determine final decision based on consensus
            if consensus_score >= self.config['auto_approve_threshold']:
                final_decision = 'approved'
            elif consensus_score <= self.config['auto_reject_threshold']:
                final_decision = 'rejected'
            else:
                # Check for agreement between reviewers
                decision_counts = {}
                for decision in decisions_list:
                    decision_counts[decision] = decision_counts.get(decision, 0) + 1
                
                # If majority agreement, use that decision
                max_count = max(decision_counts.values())
                if max_count > len(decisions_list) / 2:
                    final_decision = max(decision_counts, key=decision_counts.get)
                else:
                    # No clear consensus, escalate
                    final_decision = 'escalated'
                    status = 'escalated'
        
        # Update review item
        conn.execute("""
            UPDATE review_items 
            SET consensus_score = ?, final_decision = ?, status = ?
            WHERE video_id = ?
        """, (consensus_score, final_decision, status, video_id))

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow performance statistics"""
        with self.get_db_connection() as conn:
            # Basic counts
            stats = {}
            
            # Status distribution
            status_counts = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM review_items
                GROUP BY status
            """).fetchall()
            
            stats['status_distribution'] = {row['status']: row['count'] for row in status_counts}
            
            # Priority distribution
            priority_counts = conn.execute("""
                SELECT priority, COUNT(*) as count
                FROM review_items
                GROUP BY priority
            """).fetchall()
            
            stats['priority_distribution'] = {row['priority']: row['count'] for row in priority_counts}
            
            # Average review time (placeholder calculation)
            avg_review_time = conn.execute("""
                SELECT AVG(
                    CASE 
                        WHEN status = 'completed' 
                        THEN (julianday('now') - julianday(created_at)) * 24
                        ELSE NULL
                    END
                ) as avg_hours
                FROM review_items
            """).fetchone()
            
            stats['avg_review_time_hours'] = avg_review_time['avg_hours'] or 0
            
            # AI-Human agreement rate
            agreement_data = conn.execute("""
                SELECT 
                    ri.original_evaluation,
                    rd.score,
                    rd.decision
                FROM review_items ri
                JOIN review_decisions rd ON ri.video_id = rd.video_id
                WHERE ri.status = 'completed'
            """).fetchall()
            
            agreement_count = 0
            total_comparisons = 0
            
            for row in agreement_data:
                eval_data = json.loads(row['original_evaluation'])
                ai_score = eval_data.get('overall_score', 0.5)
                human_score = row['score']
                
                if human_score is not None:
                    # Consider agreement if scores are within 0.2 of each other
                    if abs(ai_score - (human_score / 100)) < 0.2:
                        agreement_count += 1
                    total_comparisons += 1
            
            stats['ai_human_agreement_rate'] = (
                agreement_count / total_comparisons if total_comparisons > 0 else 0
            )
            
            # Reviewer performance
            reviewer_stats = conn.execute("""
                SELECT 
                    r.reviewer_id,
                    r.name,
                    r.reviews_completed,
                    COUNT(rd.id) as actual_reviews,
                    AVG(rd.score) as avg_score
                FROM reviewers r
                LEFT JOIN review_decisions rd ON r.reviewer_id = rd.reviewer_id
                WHERE r.active = 1
                GROUP BY r.reviewer_id
            """).fetchall()
            
            stats['reviewer_performance'] = [{
                'reviewer_id': row['reviewer_id'],
                'name': row['name'],
                'reviews_completed': row['reviews_completed'],
                'avg_score': row['avg_score']
            } for row in reviewer_stats]
            
            return stats

    def create_reviewer(self, 
                       reviewer_id: str,
                       name: str,
                       email: str = "",
                       specialization: str = "") -> bool:
        """Create a new reviewer"""
        with self.get_db_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO reviewers (reviewer_id, name, email, specialization)
                    VALUES (?, ?, ?, ?)
                """, (reviewer_id, name, email, specialization))
                return True
            except sqlite3.IntegrityError:
                return False

    def export_review_data(self, output_path: str):
        """Export all review data for analysis"""
        with self.get_db_connection() as conn:
            # Get all review items with decisions
            query = """
                SELECT 
                    ri.*,
                    GROUP_CONCAT(
                        json_object(
                            'decision', rd.decision,
                            'score', rd.score,
                            'comments', rd.comments,
                            'tags', rd.tags,
                            'reviewer_id', rd.reviewer_id,
                            'timestamp', rd.timestamp
                        )
                    ) as review_decisions
                FROM review_items ri
                LEFT JOIN review_decisions rd ON ri.video_id = rd.video_id
                GROUP BY ri.video_id
            """
            
            rows = conn.execute(query).fetchall()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_items': len(rows),
                'review_items': []
            }
            
            for row in rows:
                item_data = dict(row)
                item_data['original_evaluation'] = json.loads(item_data['original_evaluation'])
                
                if item_data['review_decisions']:
                    # Parse the concatenated JSON objects
                    decisions_str = item_data['review_decisions']
                    item_data['review_decisions'] = [
                        json.loads(d) for d in decisions_str.split('},{')
                    ]
                else:
                    item_data['review_decisions'] = []
                
                export_data['review_items'].append(item_data)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

def main():
    """CLI interface for the review workflow service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Human Review Workflow Service")
    parser.add_argument("--init", action="store_true", help="Initialize from evaluation results")
    parser.add_argument("--evaluation-file", type=str, 
                       default="evaluation_datasets/large_scale_samples/evaluation_results.json",
                       help="Path to evaluation results JSON")
    parser.add_argument("--stats", action="store_true", help="Show workflow statistics")
    parser.add_argument("--export", type=str, help="Export review data to JSON file")
    parser.add_argument("--create-reviewer", nargs=3, metavar=("ID", "NAME", "EMAIL"),
                       help="Create a new reviewer")
    
    args = parser.parse_args()
    
    service = ReviewWorkflowService()
    
    if args.init:
        if not Path(args.evaluation_file).exists():
            print(f"❌ Evaluation file not found: {args.evaluation_file}")
            return
        
        count = service.create_review_items_from_evaluation(args.evaluation_file)
        print(f"✅ Created {count} review items from evaluation results")
    
    if args.create_reviewer:
        reviewer_id, name, email = args.create_reviewer
        if service.create_reviewer(reviewer_id, name, email):
            print(f"✅ Created reviewer: {name} ({reviewer_id})")
        else:
            print(f"❌ Failed to create reviewer (may already exist): {reviewer_id}")
    
    if args.stats:
        stats = service.get_workflow_statistics()
        print("\n📊 WORKFLOW STATISTICS")
        print("=" * 50)
        
        print("\n📋 Status Distribution:")
        for status, count in stats['status_distribution'].items():
            print(f"   {status}: {count}")
        
        print("\n🎯 Priority Distribution:")
        for priority, count in stats['priority_distribution'].items():
            print(f"   {priority}: {count}")
        
        print(f"\n⏱️  Average Review Time: {stats['avg_review_time_hours']:.1f} hours")
        print(f"🤝 AI-Human Agreement: {stats['ai_human_agreement_rate']:.1%}")
        
        if stats['reviewer_performance']:
            print("\n👥 Reviewer Performance:")
            for reviewer in stats['reviewer_performance']:
                avg_score = reviewer['avg_score'] or 0
                print(f"   {reviewer['name']}: {reviewer['reviews_completed']} reviews, "
                      f"avg score {avg_score:.1f}%")
    
    if args.export:
        service.export_review_data(args.export)
        print(f"✅ Exported review data to: {args.export}")

if __name__ == "__main__":
    main()
