#!/bin/bash

# RunPod Repository Synchronization Script
# Keeps your local repo in sync with RunPod, excluding large directories

set -e  # Exit on any error

# Configuration
RUNPOD_IP="213.173.110.221"
RUNPOD_PORT="30960"
SSH_KEY="~/.ssh/runpod"
REMOTE_PATH="/workspace/visionflow"
LOCAL_PATH="visionflow"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Exclude patterns for large directories
EXCLUDE_PATTERNS=(
    "--exclude=hf_cache"
    "--exclude=generated"
    "--exclude=venv"
    "--exclude=__pycache__"
    "--exclude=*.pyc"
    "--exclude=*.pyo"
    "--exclude=.git"
    "--exclude=.DS_Store"
    "--exclude=*.log"
    "--exclude=logs"
    "--exclude=cache"
    "--exclude=tmp"
    "--exclude=temp"
    "--exclude=node_modules"
    "--exclude=*.egg-info"
    "--exclude=.pytest_cache"
    "--exclude=coverage"
    "--exclude=.coverage"
    "--exclude=htmlcov"
    "--exclude=.tox"
    "--exclude=.mypy_cache"
    "--exclude=.ruff_cache"
    "--exclude=.black_cache"
)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if rsync is available
check_rsync() {
    if ! command -v rsync &> /dev/null; then
        print_error "rsync is not installed. Please install it first."
        exit 1
    fi
}

# Function to check SSH connection
check_ssh() {
    print_status "Testing SSH connection to RunPod..."
    if ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "root@$RUNPOD_IP" "echo 'SSH connection successful'" 2>/dev/null; then
        print_success "SSH connection established"
        return 0
    else
        print_error "SSH connection failed. Please check your credentials and connection."
        exit 1
    fi
}

# Function to perform full sync
full_sync() {
    print_status "Starting full repository synchronization..."
    
    # Build rsync command with all exclude patterns
    RSYNC_CMD="rsync -avz --progress --delete --no-perms --no-owner --no-group"
    
    # Add exclude patterns
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        RSYNC_CMD="$RSYNC_CMD $pattern"
    done
    
    # Add source and destination
    RSYNC_CMD="$RSYNC_CMD -e \"ssh -p $RUNPOD_PORT -i $SSH_KEY\" \"$LOCAL_PATH/\" \"root@$RUNPOD_IP:$REMOTE_PATH/\""
    
    print_status "Executing: $RSYNC_CMD"
    eval "$RSYNC_CMD"
    
    if [ $? -eq 0 ]; then
        print_success "Full synchronization completed successfully!"
    else
        print_error "Synchronization failed!"
        exit 1
    fi
}

# Function to perform selective sync (specific files/directories)
selective_sync() {
    local target="$1"
    
    if [ -z "$target" ]; then
        print_error "Please specify a file or directory to sync"
        echo "Usage: $0 selective <file_or_directory>"
        exit 1
    fi
    
    if [ ! -e "$LOCAL_PATH/$target" ]; then
        print_error "Target '$target' does not exist in $LOCAL_PATH"
        exit 1
    fi
    
    print_status "Syncing specific target: $target"
    
    RSYNC_CMD="rsync -avz --progress --no-perms --no-owner --no-group -e \"ssh -p $RUNPOD_PORT -i $SSH_KEY\" \"$LOCAL_PATH/$target\" \"root@$RUNPOD_IP:$REMOTE_PATH/\""
    
    print_status "Executing: $RSYNC_CMD"
    eval "$RSYNC_CMD"
    
    if [ $? -eq 0 ]; then
        print_success "Selective synchronization completed successfully!"
    else
        print_error "Selective synchronization failed!"
        exit 1
    fi
}

# Function to perform quick sync (only changed files)
quick_sync() {
    print_status "Starting quick synchronization (changed files only)..."
    
    RSYNC_CMD="rsync -avz --progress --no-perms --no-owner --no-group"
    
    # Add exclude patterns
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        RSYNC_CMD="$RSYNC_CMD $pattern"
    done
    
    # Add source and destination
    RSYNC_CMD="$RSYNC_CMD -e \"ssh -p $RUNPOD_PORT -i $SSH_KEY\" \"$LOCAL_PATH/\" \"root@$RUNPOD_IP:$REMOTE_PATH/\""
    
    print_status "Executing: $RSYNC_CMD"
    eval "$RSYNC_CMD"
    
    if [ $? -eq 0 ]; then
        print_success "Quick synchronization completed successfully!"
    else
        print_error "Quick synchronization failed!"
        exit 1
    fi
}

# Function to show sync status
show_status() {
    print_status "Checking synchronization status..."
    
    # Get local file count and size
    LOCAL_COUNT=$(find "$LOCAL_PATH" -type f | wc -l)
    LOCAL_SIZE=$(du -sh "$LOCAL_PATH" | cut -f1)
    
    # Get remote file count and size
    REMOTE_COUNT=$(ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "root@$RUNPOD_IP" "find $REMOTE_PATH -type f | wc -l" 2>/dev/null || echo "N/A")
    REMOTE_SIZE=$(ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "root@$RUNPOD_IP" "du -sh $REMOTE_PATH | cut -f1" 2>/dev/null || echo "N/A")
    
    echo "📊 Synchronization Status:"
    echo "   Local:  $LOCAL_COUNT files, $LOCAL_SIZE"
    echo "   Remote: $REMOTE_COUNT files, $REMOTE_SIZE"
    
    # Check for common sync issues
    print_status "Checking for common sync issues..."
    
    # Check if remote directory exists
    if ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "root@$RUNPOD_IP" "[ -d $REMOTE_PATH ]" 2>/dev/null; then
        print_success "Remote directory exists"
    else
        print_warning "Remote directory does not exist"
    fi
    
    # Check SSH key permissions
    if [ -r "$SSH_KEY" ]; then
        print_success "SSH key is readable"
    else
        print_warning "SSH key permissions may be too open"
    fi
}

# Function to show help
show_help() {
    echo "🚀 RunPod Repository Synchronization Script"
    echo "=========================================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  full       Perform full repository synchronization (default)"
    echo "  quick      Perform quick sync (changed files only)"
    echo "  selective  Sync specific file or directory"
    echo "  status     Show synchronization status"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full sync (default)"
    echo "  $0 full              # Full sync"
    echo "  $0 quick             # Quick sync"
    echo "  $0 selective scripts # Sync only scripts directory"
    echo "  $0 selective visionflow/services/generation/wan_video_service.py # Sync specific file"
    echo "  $0 status            # Show sync status"
    echo ""
    echo "Excluded directories (not synced):"
    echo "  - hf_cache (HuggingFace models)"
    echo "  - generated (video outputs)"
    echo "  - venv (virtual environment)"
    echo "  - __pycache__ (Python cache)"
    echo "  - .git (version control)"
    echo ""
    echo "Configuration:"
    echo "  RunPod IP: $RUNPOD_IP"
    echo "  RunPod Port: $RUNPOD_PORT"
    echo "  SSH Key: $SSH_KEY"
    echo "  Remote Path: $REMOTE_PATH"
    echo "  Local Path: $LOCAL_PATH"
}

# Main script logic
main() {
    # Check if we're in the right directory
    if [ ! -d "$LOCAL_PATH" ]; then
        print_error "Local path '$LOCAL_PATH' not found. Please run this script from the project root."
        exit 1
    fi
    
    # Check dependencies
    check_rsync
    
    # Parse command line arguments
    case "${1:-full}" in
        "full")
            check_ssh
            full_sync
            ;;
        "quick")
            check_ssh
            quick_sync
            ;;
        "selective")
            check_ssh
            selective_sync "$2"
            ;;
        "status")
            show_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
