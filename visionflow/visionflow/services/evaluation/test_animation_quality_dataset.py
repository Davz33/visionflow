"""
Test Dataset for Animation Quality Assessment (Test Scenario 1)
Creates sample videos with various quality issues for testing the assessment framework
"""

import asyncio
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ...shared.monitoring import get_logger
from .animation_quality_assessor import AnimationQualityIssue, QualitySeverity

logger = get_logger("test_animation_dataset")


@dataclass
class TestVideoMetadata:
    """Metadata for test videos with known quality issues"""
    video_path: str
    original_prompt: str
    expected_issues: List[AnimationQualityIssue]
    expected_severity: QualitySeverity
    description: str
    generation_parameters: Dict[str, Any]


class TestAnimationQualityDataset:
    """
    Creates and manages test dataset for animation quality assessment
    Generates sample videos with controlled quality issues
    """
    
    def __init__(self, output_dir: str = "test_animation_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test video configurations
        self.test_configs = self._create_test_configurations()
        
        logger.info(f"🎬 Test Animation Quality Dataset initialized at {self.output_dir}")
    
    def _create_test_configurations(self) -> List[TestVideoMetadata]:
        """Create test video configurations with known quality issues"""
        
        configs = [
            # Test Case 1: Character Drift
            TestVideoMetadata(
                video_path=str(self.output_dir / "character_drift_test.mp4"),
                original_prompt="A young woman walking through a forest, maintaining consistent appearance",
                expected_issues=[AnimationQualityIssue.CHARACTER_DRIFT, AnimationQualityIssue.CHARACTER_INCONSISTENCY],
                expected_severity=QualitySeverity.HIGH,
                description="Character appearance changes between frames",
                generation_parameters={
                    "character_consistency": "poor",
                    "style_consistency": "variable",
                    "frame_sampling": "inconsistent"
                }
            ),
            
            # Test Case 2: Technical Artifacts
            TestVideoMetadata(
                video_path=str(self.output_dir / "technical_artifacts_test.mp4"),
                original_prompt="A cat playing with a ball in a sunny garden",
                expected_issues=[AnimationQualityIssue.TECHNICAL_ARTIFACT, AnimationQualityIssue.GENERATION_FAILURE],
                expected_severity=QualitySeverity.MEDIUM,
                description="Blurry areas and generation glitches",
                generation_parameters={
                    "noise_level": "high",
                    "generation_stability": "poor",
                    "quality_consistency": "variable"
                }
            ),
            
            # Test Case 3: Color Inconsistency
            TestVideoMetadata(
                video_path=str(self.output_dir / "color_inconsistency_test.mp4"),
                original_prompt="A sunset scene with warm colors throughout",
                expected_issues=[AnimationQualityIssue.COLOR_INCONSISTENCY],
                expected_severity=QualitySeverity.MEDIUM,
                description="Color temperature and saturation vary between frames",
                generation_parameters={
                    "color_consistency": "poor",
                    "lighting_consistency": "variable",
                    "color_grading": "inconsistent"
                }
            ),
            
            # Test Case 4: Resolution Drop
            TestVideoMetadata(
                video_path=str(self.output_dir / "resolution_drop_test.mp4"),
                original_prompt="A detailed cityscape with many buildings and people",
                expected_issues=[AnimationQualityIssue.RESOLUTION_DROP],
                expected_severity=QualitySeverity.HIGH,
                description="Resolution decreases in middle frames",
                generation_parameters={
                    "resolution_consistency": "poor",
                    "detail_level": "variable",
                    "upscaling_quality": "low"
                }
            ),
            
            # Test Case 5: Frame Corruption
            TestVideoMetadata(
                video_path=str(self.output_dir / "frame_corruption_test.mp4"),
                original_prompt="A peaceful lake with birds flying overhead",
                expected_issues=[AnimationQualityIssue.FRAME_CORRUPTION],
                expected_severity=QualitySeverity.CRITICAL,
                description="Some frames are corrupted or completely black",
                generation_parameters={
                    "frame_stability": "poor",
                    "generation_reliability": "low",
                    "error_handling": "none"
                }
            ),
            
            # Test Case 6: Motion Inconsistency
            TestVideoMetadata(
                video_path=str(self.output_dir / "motion_inconsistency_test.mp4"),
                original_prompt="A car driving smoothly down a highway",
                expected_issues=[AnimationQualityIssue.MOTION_INCONSISTENCY],
                expected_severity=QualitySeverity.MEDIUM,
                description="Motion is jerky and inconsistent",
                generation_parameters={
                    "motion_consistency": "poor",
                    "temporal_stability": "low",
                    "frame_interpolation": "none"
                }
            ),
            
            # Test Case 7: High Quality (Control)
            TestVideoMetadata(
                video_path=str(self.output_dir / "high_quality_control.mp4"),
                original_prompt="A professional animation of a character in a detailed environment",
                expected_issues=[],  # No issues expected
                expected_severity=QualitySeverity.LOW,
                description="High quality animation with no significant issues",
                generation_parameters={
                    "quality_level": "high",
                    "consistency": "excellent",
                    "stability": "high"
                }
            ),
            
            # Test Case 8: Mixed Issues
            TestVideoMetadata(
                video_path=str(self.output_dir / "mixed_issues_test.mp4"),
                original_prompt="A complex scene with multiple characters and dynamic lighting",
                expected_issues=[
                    AnimationQualityIssue.CHARACTER_DRIFT,
                    AnimationQualityIssue.COLOR_INCONSISTENCY,
                    AnimationQualityIssue.TECHNICAL_ARTIFACT
                ],
                expected_severity=QualitySeverity.HIGH,
                description="Multiple quality issues present",
                generation_parameters={
                    "complexity": "high",
                    "consistency": "poor",
                    "stability": "medium"
                }
            )
        ]
        
        return configs
    
    async def generate_test_videos(self) -> List[TestVideoMetadata]:
        """Generate all test videos with controlled quality issues"""
        
        logger.info("🎬 Generating test animation quality dataset...")
        
        generated_videos = []
        
        for config in self.test_configs:
            try:
                logger.info(f"📹 Generating: {config.description}")
                
                # Generate video based on configuration
                video_path = await self._generate_test_video(config)
                
                if video_path:
                    generated_videos.append(config)
                    logger.info(f"✅ Generated: {config.video_path}")
                else:
                    logger.error(f"❌ Failed to generate: {config.video_path}")
                    
            except Exception as e:
                logger.error(f"❌ Error generating {config.video_path}: {e}")
        
        # Save dataset metadata
        await self._save_dataset_metadata(generated_videos)
        
        logger.info(f"🎬 Test dataset generation completed: {len(generated_videos)} videos")
        return generated_videos
    
    async def _generate_test_video(self, config: TestVideoMetadata) -> str:
        """Generate a single test video with controlled quality issues"""
        
        # Video parameters
        width, height = 512, 512
        fps = 24
        duration = 3  # 3 seconds
        total_frames = fps * duration
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config.video_path, fourcc, fps, (width, height))
        
        try:
            for frame_idx in range(total_frames):
                # Generate frame based on test configuration
                frame = await self._generate_test_frame(
                    frame_idx, total_frames, config, width, height
                )
                
                # Write frame to video
                out.write(frame)
            
            out.release()
            return config.video_path
            
        except Exception as e:
            logger.error(f"Error generating video {config.video_path}: {e}")
            out.release()
            return None
    
    async def _generate_test_frame(
        self, 
        frame_idx: int, 
        total_frames: int, 
        config: TestVideoMetadata, 
        width: int, 
        height: int
    ) -> np.ndarray:
        """Generate a single test frame with controlled quality issues"""
        
        # Base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply different effects based on test configuration
        if "character_drift" in config.video_path:
            frame = self._create_character_drift_frame(frame, frame_idx, total_frames)
        elif "technical_artifacts" in config.video_path:
            frame = self._create_technical_artifacts_frame(frame, frame_idx, total_frames)
        elif "color_inconsistency" in config.video_path:
            frame = self._create_color_inconsistency_frame(frame, frame_idx, total_frames)
        elif "resolution_drop" in config.video_path:
            frame = self._create_resolution_drop_frame(frame, frame_idx, total_frames)
        elif "frame_corruption" in config.video_path:
            frame = self._create_frame_corruption_frame(frame, frame_idx, total_frames)
        elif "motion_inconsistency" in config.video_path:
            frame = self._create_motion_inconsistency_frame(frame, frame_idx, total_frames)
        elif "high_quality_control" in config.video_path:
            frame = self._create_high_quality_frame(frame, frame_idx, total_frames)
        elif "mixed_issues" in config.video_path:
            frame = self._create_mixed_issues_frame(frame, frame_idx, total_frames)
        else:
            frame = self._create_default_frame(frame, frame_idx, total_frames)
        
        return frame
    
    def _create_character_drift_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create frame with character drift issues"""
        
        # Base character (simple circle for head)
        center_x = width // 2 + int(20 * np.sin(frame_idx * 0.1))  # Slight movement
        center_y = height // 2
        
        # Character appearance changes over time (drift)
        drift_factor = frame_idx / total_frames
        
        # Head size changes (inconsistency)
        head_radius = int(30 + 10 * np.sin(frame_idx * 0.2))
        
        # Color changes (character appearance drift)
        color_shift = int(50 * drift_factor)
        color = (100 + color_shift, 150 + color_shift, 200 + color_shift)
        
        # Draw character
        cv2.circle(frame, (center_x, center_y), head_radius, color, -1)
        
        # Add background
        cv2.rectangle(frame, (0, 0), (width, height), (50, 100, 50), -1)
        
        return frame
    
    def _create_technical_artifacts_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create frame with technical artifacts"""
        
        # Base scene
        cv2.rectangle(frame, (0, 0), (width, height), (100, 150, 100), -1)
        
        # Add some content
        cv2.circle(frame, (width//2, height//2), 50, (255, 255, 255), -1)
        
        # Add artifacts based on frame
        if frame_idx % 10 < 3:  # Every 10th frame, add blur
            # Add blur effect
            kernel = np.ones((15, 15), np.float32) / 225
            frame = cv2.filter2D(frame, -1, kernel)
        
        if frame_idx % 15 == 0:  # Random corruption
            # Add noise
            noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)
        
        # Add generation glitches
        if 20 <= frame_idx <= 30:
            # Add horizontal lines (glitch effect)
            for i in range(0, height, 20):
                cv2.line(frame, (0, i), (width, i), (255, 0, 0), 2)
        
        return frame
    
    def _create_color_inconsistency_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create frame with color inconsistency"""
        
        # Color temperature changes over time
        temp_factor = frame_idx / total_frames
        
        # Warm to cool color transition
        if temp_factor < 0.5:
            # Warm colors
            base_color = (100, 150, 200)
            temp_shift = int(50 * (0.5 - temp_factor))
            color = (base_color[0] + temp_shift, base_color[1] + temp_shift, base_color[2])
        else:
            # Cool colors
            base_color = (200, 150, 100)
            temp_shift = int(50 * (temp_factor - 0.5))
            color = (base_color[0] - temp_shift, base_color[1], base_color[2] + temp_shift)
        
        # Fill frame with varying color
        frame[:] = color
        
        # Add some content
        cv2.circle(frame, (width//2, height//2), 80, (255, 255, 255), -1)
        
        return frame
    
    def _create_resolution_drop_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create frame with resolution drop"""
        
        # Normal resolution for most frames
        if frame_idx < total_frames // 3 or frame_idx > 2 * total_frames // 3:
            # High resolution
            cv2.rectangle(frame, (0, 0), (width, height), (100, 150, 100), -1)
            cv2.circle(frame, (width//2, height//2), 60, (255, 255, 255), -1)
            # Add details
            for i in range(0, width, 20):
                cv2.line(frame, (i, 0), (i, height), (200, 200, 200), 1)
        else:
            # Low resolution (simulated by downscaling and upscaling)
            small_frame = np.zeros((height//4, width//4, 3), dtype=np.uint8)
            cv2.rectangle(small_frame, (0, 0), (width//4, height//4), (100, 150, 100), -1)
            cv2.circle(small_frame, (width//8, height//8), 15, (255, 255, 255), -1)
            
            # Upscale back to original size (creates pixelated effect)
            frame = cv2.resize(small_frame, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return frame
    
    def _create_frame_corruption_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create frame with corruption issues"""
        
        # Normal frames for most of the video
        if frame_idx < total_frames // 4 or frame_idx > 3 * total_frames // 4:
            cv2.rectangle(frame, (0, 0), (width, height), (100, 150, 100), -1)
            cv2.circle(frame, (width//2, height//2), 50, (255, 255, 255), -1)
        else:
            # Corrupted frames
            if frame_idx % 3 == 0:
                # Completely black frame
                frame[:] = 0
            elif frame_idx % 3 == 1:
                # Completely white frame
                frame[:] = 255
            else:
                # Random noise frame
                frame[:] = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
        
        return frame
    
    def _create_motion_inconsistency_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create frame with motion inconsistency"""
        
        # Base scene
        cv2.rectangle(frame, (0, 0), (width, height), (50, 50, 100), -1)
        
        # Moving object with inconsistent motion
        if frame_idx % 2 == 0:  # Every other frame
            # Smooth motion
            x = width // 4 + int(frame_idx * 2)
            y = height // 2
        else:
            # Jerky motion (skip frames)
            x = width // 4 + int((frame_idx - 1) * 2)
            y = height // 2 + int(10 * np.sin(frame_idx * 0.5))
        
        # Draw moving object
        cv2.circle(frame, (x % width, y), 30, (255, 255, 255), -1)
        
        return frame
    
    def _create_high_quality_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create high quality frame (control)"""
        
        # Smooth background gradient
        for y in range(height):
            color_val = int(100 + 50 * y / height)
            cv2.line(frame, (0, y), (width, y), (color_val, color_val + 50, color_val + 100))
        
        # Smooth character movement
        x = width // 2 + int(30 * np.sin(frame_idx * 0.1))
        y = height // 2 + int(10 * np.cos(frame_idx * 0.1))
        
        # Detailed character
        cv2.circle(frame, (x, y), 40, (255, 255, 255), -1)
        cv2.circle(frame, (x - 15, y - 10), 5, (0, 0, 0), -1)  # Eye
        cv2.circle(frame, (x + 15, y - 10), 5, (0, 0, 0), -1)  # Eye
        
        # Add smooth details
        for i in range(0, width, 10):
            alpha = 0.3
            color = (int(200 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.line(frame, (i, 0), (i, height), color, 1)
        
        return frame
    
    def _create_mixed_issues_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create frame with mixed quality issues"""
        
        # Combine multiple issues
        frame = self._create_character_drift_frame(frame, frame_idx, total_frames)
        
        # Add some technical artifacts
        if frame_idx % 8 < 2:
            noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)
        
        # Add color inconsistency
        if frame_idx > total_frames // 2:
            # Apply color shift
            frame[:, :, 0] = np.clip(frame[:, :, 0] + 30, 0, 255)
        
        return frame
    
    def _create_default_frame(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Create default test frame"""
        
        # Simple animated scene
        cv2.rectangle(frame, (0, 0), (width, height), (100, 150, 100), -1)
        
        # Moving circle
        x = int(width * frame_idx / total_frames)
        y = height // 2
        cv2.circle(frame, (x, y), 30, (255, 255, 255), -1)
        
        return frame
    
    async def _save_dataset_metadata(self, generated_videos: List[TestVideoMetadata]):
        """Save dataset metadata for reference"""
        
        metadata = {
            "dataset_info": {
                "name": "Animation Quality Assessment Test Dataset",
                "description": "Test dataset for evaluating animation quality assessment capabilities",
                "total_videos": len(generated_videos),
                "generated_at": asyncio.get_event_loop().time()
            },
            "test_cases": []
        }
        
        for video in generated_videos:
            test_case = {
                "video_path": video.video_path,
                "original_prompt": video.original_prompt,
                "expected_issues": [issue.value for issue in video.expected_issues],
                "expected_severity": video.expected_severity.value,
                "description": video.description,
                "generation_parameters": video.generation_parameters
            }
            metadata["test_cases"].append(test_case)
        
        # Save metadata file
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 Dataset metadata saved to {metadata_path}")


# Global dataset instance
_test_dataset = None

def get_test_animation_dataset() -> TestAnimationQualityDataset:
    """Get or create test animation dataset instance"""
    global _test_dataset
    if _test_dataset is None:
        _test_dataset = TestAnimationQualityDataset()
    return _test_dataset


async def generate_test_dataset():
    """Convenience function to generate the complete test dataset"""
    dataset = get_test_animation_dataset()
    return await dataset.generate_test_videos()
