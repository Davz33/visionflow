"""
Rubric Examples and Best Practices for Video Generation Evaluation
Demonstrates how to use qualitative rubrics to complement numeric scores
"""

from typing import List, Dict, Any
from .quality_metrics import (
    RubricCriteria,
    QualitativeAssessment,
    create_qualitative_assessment,
    get_default_video_quality_rubrics,
    validate_evaluation_consistency,
    TechnicalMetrics,
    ContentMetrics,
    AestheticMetrics
)


def create_custom_video_rubrics() -> Dict[str, RubricCriteria]:
    """Create custom rubric criteria for specific video generation use cases"""
    
    # Custom rubric for educational video content
    educational_content = RubricCriteria(
        criteria_name="Educational Value",
        description="How effectively the video conveys educational information and concepts",
        examples=[
            "Clear explanation of complex concepts with visual aids",
            "Logical progression from basic to advanced topics",
            "Engaging presentation that maintains viewer attention",
            "Accurate information with appropriate depth for target audience"
        ],
        thresholds={
            "excellent": "Exceptional educational value with clear, engaging, and accurate content",
            "good": "Strong educational content with minor areas for improvement",
            "acceptable": "Decent educational value with some unclear or dry sections",
            "below_average": "Limited educational value due to poor presentation or unclear content",
            "poor": "Minimal educational value with confusing or incorrect information"
        }
    )
    
    # Custom rubric for creative/artistic videos
    artistic_creativity = RubricCriteria(
        criteria_name="Artistic Innovation",
        description="Level of creative originality and artistic expression in the video",
        examples=[
            "Unique visual style and creative camera angles",
            "Innovative use of color, lighting, and composition",
            "Creative storytelling techniques and narrative structure",
            "Original artistic vision that stands out from typical content"
        ],
        thresholds={
            "excellent": "Highly innovative and original artistic expression",
            "good": "Creative approach with some innovative elements",
            "acceptable": "Standard creative techniques with occasional originality",
            "below_average": "Limited creativity, mostly conventional approaches",
            "poor": "Unoriginal content with no creative innovation"
        }
    )
    
    # Custom rubric for commercial/brand videos
    brand_effectiveness = RubricCriteria(
        criteria_name="Brand Effectiveness",
        description="How well the video serves its commercial or branding purpose",
        examples=[
            "Clear brand message and consistent visual identity",
            "Appropriate tone and style for target audience",
            "Effective call-to-action and engagement elements",
            "Professional quality that enhances brand perception"
        ],
        thresholds={
            "excellent": "Highly effective brand communication with strong impact",
            "good": "Effective brand messaging with minor improvements possible",
            "acceptable": "Adequate brand representation with some effectiveness",
            "below_average": "Limited brand effectiveness due to poor execution",
            "poor": "Ineffective brand communication that may harm perception"
        }
    )
    
    return {
        "educational_value": educational_content,
        "artistic_creativity": artistic_creativity,
        "brand_effectiveness": brand_effectiveness
    }


def demonstrate_rubric_usage() -> Dict[str, Any]:
    """Demonstrate how to use rubrics to address LLM scoring inconsistencies"""
    
    # Get default rubrics
    default_rubrics = get_default_video_quality_rubrics()
    custom_rubrics = create_custom_video_rubrics()
    
    # Example: Multiple LLM evaluations of the same video with inconsistent scores
    # This simulates the problem mentioned in the evaluation documents
    
    # LLM Evaluation 1 (might be overly optimistic)
    llm_score_1 = 0.92  # High score
    reasoning_1 = "Video shows excellent prompt adherence with high visual quality"
    confidence_1 = 0.85
    
    # LLM Evaluation 2 (might be overly critical)
    llm_score_2 = 0.73  # Lower score for same content
    reasoning_2 = "Video matches prompt but has some technical issues and could be more engaging"
    confidence_2 = 0.78
    
    # LLM Evaluation 3 (middle ground)
    llm_score_3 = 0.84  # Moderate score
    reasoning_3 = "Good overall quality with room for improvement in visual coherence"
    confidence_3 = 0.82
    
    # Create qualitative assessments using rubrics
    prompt_adherence_rubric = default_rubrics["prompt_adherence"]
    
    assessment_1 = create_qualitative_assessment(
        rubric=prompt_adherence_rubric,
        numeric_score=llm_score_1,
        reasoning=reasoning_1,
        confidence=confidence_1,
        contextual_factors=["high_confidence", "detailed_analysis"]
    )
    
    assessment_2 = create_qualitative_assessment(
        rubric=prompt_adherence_rubric,
        numeric_score=llm_score_2,
        reasoning=reasoning_2,
        confidence=confidence_2,
        contextual_factors=["technical_focus", "critical_review"]
    )
    
    assessment_3 = create_qualitative_assessment(
        rubric=prompt_adherence_rubric,
        numeric_score=llm_score_3,
        reasoning=reasoning_3,
        confidence=confidence_3,
        contextual_factors=["balanced_assessment", "moderate_confidence"]
    )
    
    # Analyze the inconsistency
    scores = [llm_score_1, llm_score_2, llm_score_3]
    import statistics
    
    analysis = {
        "raw_scores": scores,
        "mean_score": statistics.mean(scores),
        "std_deviation": statistics.stdev(scores),
        "coefficient_of_variation": statistics.stdev(scores) / statistics.mean(scores),
        "inconsistency_level": "high" if statistics.stdev(scores) / statistics.mean(scores) > 0.15 else "medium",
        "qualitative_assessments": [
            assessment_1.to_human_readable(),
            assessment_2.to_human_readable(),
            assessment_3.to_human_readable()
        ]
    }
    
    return analysis


def create_robust_evaluation_prompt(rubric: RubricCriteria) -> str:
    """Create a robust evaluation prompt that reduces LLM inconsistencies"""
    
    prompt = f"""
Please evaluate the following video content using the specific criteria below.

EVALUATION CRITERIA: {rubric.criteria_name}
DESCRIPTION: {rubric.description}

EVALUATION SCALE:
- EXCELLENT (0.9-1.0): {rubric.thresholds.get('excellent', 'Exceptional quality')}
- GOOD (0.8-0.89): {rubric.thresholds.get('good', 'High quality with minor issues')}
- ACCEPTABLE (0.7-0.79): {rubric.thresholds.get('acceptable', 'Decent quality with some issues')}
- BELOW AVERAGE (0.6-0.69): {rubric.thresholds.get('below_average', 'Limited quality with significant issues')}
- POOR (0.0-0.59): {rubric.thresholds.get('poor', 'Poor quality with major issues')}

EXAMPLES OF WHAT TO LOOK FOR:
"""
    
    for example in rubric.examples:
        prompt += f"• {example}\n"
    
    prompt += """
INSTRUCTIONS:
1. Watch the video carefully, focusing on the specific criteria above
2. Consider the examples provided as reference points
3. Assign a score between 0.0 and 1.0 based on the evaluation scale
4. Provide a brief reasoning for your score
5. Indicate your confidence level (0.0-1.0) in this assessment

RESPONSE FORMAT:
Score: [0.0-1.0]
Reasoning: [Your explanation]
Confidence: [0.0-1.0]
"""
    
    return prompt


def get_evaluation_consistency_tips() -> List[str]:
    """Get practical tips for improving evaluation consistency"""
    
    return [
        "Use structured rubrics with specific examples and thresholds",
        "Implement multiple evaluation passes with different LLM models",
        "Combine LLM evaluations with human review for critical assessments",
        "Track evaluation confidence scores and flag low-confidence results",
        "Use ensemble methods by averaging multiple LLM evaluations",
        "Implement evaluation validation checks for outlier scores",
        "Create evaluation templates that force structured responses",
        "Use contextual prompts that reference previous successful evaluations",
        "Implement evaluation quality gates based on confidence thresholds",
        "Regularly review and update evaluation rubrics based on feedback"
    ]


def create_evaluation_workflow() -> Dict[str, Any]:
    """Create a complete evaluation workflow that addresses LLM inconsistencies"""
    
    workflow = {
        "step_1": "Define clear evaluation rubrics with specific criteria and examples",
        "step_2": "Create structured evaluation prompts that force consistent responses",
        "step_3": "Run multiple LLM evaluations using different models/prompts",
        "step_4": "Calculate consistency metrics and identify outliers",
        "step_5": "Apply confidence-weighted scoring for final assessment",
        "step_6": "Generate qualitative assessments using rubric criteria",
        "step_7": "Flag inconsistent evaluations for human review",
        "step_8": "Update evaluation system based on consistency analysis"
    }
    
    return workflow


if __name__ == "__main__":
    # Demonstrate the rubric system
    print("🎬 Video Generation Evaluation Rubric System")
    print("=" * 50)
    
    # Show default rubrics
    default_rubrics = get_default_video_quality_rubrics()
    print(f"\n📋 Default Rubrics Available: {list(default_rubrics.keys())}")
    
    # Show custom rubrics
    custom_rubrics = create_custom_video_rubrics()
    print(f"🎨 Custom Rubrics Available: {list(custom_rubrics.keys())}")
    
    # Demonstrate inconsistency analysis
    print("\n🔍 Demonstrating Inconsistency Analysis:")
    analysis = demonstrate_rubric_usage()
    print(f"   Raw Scores: {analysis['raw_scores']}")
    print(f"   Mean Score: {analysis['mean_score']:.2f}")
    print(f"   Standard Deviation: {analysis['std_deviation']:.2f}")
    print(f"   Inconsistency Level: {analysis['inconsistency_level']}")
    
    # Show evaluation tips
    print("\n💡 Evaluation Consistency Tips:")
    tips = get_evaluation_consistency_tips()
    for i, tip in enumerate(tips[:5], 1):  # Show first 5 tips
        print(f"   {i}. {tip}")
    
    print("\n✅ Rubric system ready for production use!")
