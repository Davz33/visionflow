#!/bin/bash

# VisionFlow Local Setup Test Script
# Tests the real video generation implementation

set -e

echo "🧪 VisionFlow Local Setup Test"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test functions
test_api_health() {
    echo -n "🔗 Testing API health... "
    if curl -f -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_generation_service() {
    echo -n "🎮 Testing Generation service... "
    if curl -f -s http://localhost:8002/health > /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        
        # Check GPU status
        echo -n "    GPU status: "
        GPU_STATUS=$(curl -s http://localhost:8002/models/status | jq -r '.device' 2>/dev/null || echo "unknown")
        if [[ "$GPU_STATUS" == *"cuda"* ]]; then
            echo -e "${GREEN}GPU (${GPU_STATUS})${NC}"
        elif [[ "$GPU_STATUS" == "mps" ]]; then
            echo -e "${YELLOW}Apple Silicon (${GPU_STATUS})${NC}"
        else
            echo -e "${YELLOW}CPU (${GPU_STATUS})${NC}"
        fi
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_basic_generation() {
    echo "🎬 Testing basic video generation..."
    
    # Create test request
    TEST_REQUEST='{
        "prompt": "A peaceful mountain lake at sunset",
        "duration": 3,
        "quality": "medium",
        "resolution": "512x512"
    }'
    
    echo "    Submitting generation request..."
    RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/generate \
        -H "Content-Type: application/json" \
        -d "$TEST_REQUEST")
    
    JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id' 2>/dev/null || echo "")
    
    if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
        echo -e "    ${RED}❌ Failed to submit job${NC}"
        echo "    Response: $RESPONSE"
        return 1
    fi
    
    echo "    Job ID: $JOB_ID"
    echo "    Waiting for completion..."
    
    # Poll for completion (max 5 minutes)
    for i in {1..60}; do
        STATUS_RESPONSE=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID/status)
        STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status' 2>/dev/null || echo "")
        
        case "$STATUS" in
            "completed")
                echo -e "    ${GREEN}✅ Generation completed!${NC}"
                
                # Check if video file exists
                VIDEO_PATH=$(echo "$STATUS_RESPONSE" | jq -r '.generation_result.video_path' 2>/dev/null || echo "")
                if [ -n "$VIDEO_PATH" ] && [ "$VIDEO_PATH" != "null" ]; then
                    echo "    Video saved: $VIDEO_PATH"
                    
                    # Check file size
                    if [ -f "$VIDEO_PATH" ]; then
                        FILE_SIZE=$(stat -f%z "$VIDEO_PATH" 2>/dev/null || stat -c%s "$VIDEO_PATH" 2>/dev/null || echo "0")
                        echo "    File size: $((FILE_SIZE / 1024)) KB"
                        
                        if [ "$FILE_SIZE" -gt 1000 ]; then
                            echo -e "    ${GREEN}✅ Video file looks valid${NC}"
                            return 0
                        else
                            echo -e "    ${YELLOW}⚠️  Video file seems small${NC}"
                            return 0
                        fi
                    else
                        echo -e "    ${YELLOW}⚠️  Video file not found${NC}"
                        return 0
                    fi
                else
                    echo -e "    ${YELLOW}⚠️  No video path in response${NC}"
                    return 0
                fi
                ;;
            "failed")
                echo -e "    ${RED}❌ Generation failed${NC}"
                ERROR=$(echo "$STATUS_RESPONSE" | jq -r '.error_message' 2>/dev/null || echo "Unknown error")
                echo "    Error: $ERROR"
                return 1
                ;;
            "processing")
                echo -n "."
                ;;
            *)
                echo -n "?"
                ;;
        esac
        
        sleep 5
    done
    
    echo -e "\n    ${RED}❌ Timeout waiting for completion${NC}"
    return 1
}

test_evaluation_system() {
    echo "📊 Testing evaluation system..."
    
    # First need a completed job
    echo "    Submitting evaluation test job..."
    
    TEST_REQUEST='{
        "prompt": "A simple test scene",
        "duration": 2,
        "quality": "low",
        "resolution": "256x256"
    }'
    
    RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/generate \
        -H "Content-Type: application/json" \
        -d "$TEST_REQUEST")
    
    JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id' 2>/dev/null || echo "")
    
    if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
        echo -e "    ${YELLOW}⚠️  Could not submit test job for evaluation${NC}"
        return 1
    fi
    
    # Wait for completion (shorter timeout)
    echo "    Waiting for job completion..."
    for i in {1..24}; do  # 2 minutes
        STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID/status | jq -r '.status' 2>/dev/null || echo "")
        
        if [ "$STATUS" = "completed" ]; then
            echo "    Testing evaluation..."
            
            EVAL_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/evaluate/$JOB_ID \
                -H "Content-Type: application/json" \
                -d '{"include_benchmarks": false, "evaluation_level": "basic"}')
            
            EVAL_ID=$(echo "$EVAL_RESPONSE" | jq -r '.evaluation_id' 2>/dev/null || echo "")
            
            if [ -n "$EVAL_ID" ] && [ "$EVAL_ID" != "null" ]; then
                echo -e "    ${GREEN}✅ Evaluation system working${NC}"
                OVERALL_SCORE=$(echo "$EVAL_RESPONSE" | jq -r '.overall_score' 2>/dev/null || echo "0")
                echo "    Overall score: $OVERALL_SCORE"
                return 0
            else
                echo -e "    ${RED}❌ Evaluation failed${NC}"
                return 1
            fi
        elif [ "$STATUS" = "failed" ]; then
            echo -e "    ${YELLOW}⚠️  Test job failed, skipping evaluation test${NC}"
            return 1
        fi
        
        sleep 5
    done
    
    echo -e "    ${YELLOW}⚠️  Test job didn't complete in time${NC}"
    return 1
}

show_system_info() {
    echo ""
    echo "💻 System Information"
    echo "===================="
    
    # Docker info
    echo "Docker:"
    docker --version
    docker compose version
    
    # GPU info
    echo ""
    echo "GPU:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1
    else
        echo "  No NVIDIA GPU detected"
    fi
    
    # Service status
    echo ""
    echo "Services:"
    curl -s http://localhost:8000/health | jq -r '"  API: \(.status)"' 2>/dev/null || echo "  API: unavailable"
    curl -s http://localhost:8002/health | jq -r '"  Generation: \(.status)"' 2>/dev/null || echo "  Generation: unavailable"
    
    # Memory usage
    echo ""
    echo "Memory usage:"
    curl -s http://localhost:8002/models/status | jq '.memory_usage' 2>/dev/null || echo "  Unavailable"
}

# Main execution
main() {
    echo "Starting comprehensive test suite..."
    echo ""
    
    # Basic health checks
    test_api_health || exit 1
    test_generation_service || exit 1
    
    echo ""
    
    # Real generation test
    echo "🎬 Running real video generation test..."
    echo "This may take a few minutes depending on your hardware..."
    echo ""
    
    if test_basic_generation; then
        echo ""
        echo "🎉 Real video generation is working!"
    else
        echo ""
        echo "⚠️  Video generation test had issues (check logs)"
    fi
    
    echo ""
    
    # Evaluation system test
    if test_evaluation_system; then
        echo ""
        echo "📊 Evaluation system is working!"
    else
        echo ""
        echo "⚠️  Evaluation system test had issues"
    fi
    
    # System info
    show_system_info
    
    echo ""
    echo "🏁 Test suite completed!"
    echo ""
    echo "Next steps:"
    echo "- Check logs: docker compose -f docker-compose.local.yml logs -f"
    echo "- Monitor GPU: nvidia-smi -l 1"
    echo "- API docs: http://localhost:8000/docs"
    echo "- Generation docs: http://localhost:8002/docs"
}

# Help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "VisionFlow Local Setup Test Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --quick        Run only health checks"
    echo ""
    exit 0
fi

# Quick test mode
if [ "$1" = "--quick" ]; then
    test_api_health
    test_generation_service
    show_system_info
    exit 0
fi

# Run full test
main
