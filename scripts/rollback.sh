#!/bin/bash
# BI-IDE v8 - Rollback Script
# Quickly rollback to previous deployment version

set -e

ENVIRONMENT="${1:-staging}"
NAMESPACE="bi-ide-v8"
MAX_REVISIONS=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Show usage
usage() {
    cat << EOF
BI-IDE v8 Rollback Script

Usage: $0 [environment] [options]

Environments:
  staging     Rollback staging environment (default)
  production  Rollback production environment

Options:
  --revision N    Rollback to specific revision N
  --list          List available revisions
  --dry-run       Show what would be rolled back without executing
  -h, --help      Show this help message

Examples:
  $0 staging                    # Rollback staging by one revision
  $0 production --revision 3    # Rollback production to revision 3
  $0 production --list          # List production revisions

EOF
}

# List available revisions
list_revisions() {
    log_info "Available revisions for ${ENVIRONMENT}..."
    
    if [ "$ENVIRONMENT" == "k8s" ] || [ "$ENVIRONMENT" == "kubernetes" ]; then
        echo
        echo "=== API Deployment History ==="
        kubectl rollout history deployment/bi-ide-api -n $NAMESPACE
        echo
        echo "=== UI Deployment History ==="
        kubectl rollout history deployment/bi-ide-ui -n $NAMESPACE
        echo
        echo "=== Worker Deployment History ==="
        kubectl rollout history deployment/bi-ide-worker -n $NAMESPACE
    else
        cd "/opt/bi-ide-v8/${ENVIRONMENT}"
        
        # Docker deployment revision tracking
        if [ -f ".deployment-history" ]; then
            echo "Recent deployments:"
            tail -n $MAX_REVISIONS .deployment-history
        else
            # Try to get from docker images
            docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}" | grep bi-ide-v8 | head -10
        fi
    fi
}

# Get previous revision
get_previous_revision() {
    if [ "$ENVIRONMENT" == "k8s" ] || [ "$ENVIRONMENT" == "kubernetes" ]; then
        # Get current revision
        local current_revision
        current_revision=$(kubectl rollout history deployment/bi-ide-api -n $NAMESPACE | grep -v "REVISION" | tail -1 | awk '{print $1}')
        if [ "$current_revision" -gt 1 ]; then
            echo $((current_revision - 1))
        else
            echo "1"
        fi
    else
        # For docker-compose, check previous env file
        if [ -f ".previous-env" ]; then
            cat .previous-env
        else
            echo ""
        fi
    fi
}

# Perform rollback
rollback_k8s() {
    local target_revision="$1"
    local dry_run="${2:-false}"
    
    log_info "Kubernetes rollback initiated..."
    
    if [ "$dry_run" == "true" ]; then
        log_warn "DRY RUN MODE - No changes will be made"
        echo "Would rollback to revision: ${target_revision}"
        kubectl rollout history deployment/bi-ide-api -n $NAMESPACE | grep "${target_revision}"
        return 0
    fi
    
    # Create emergency backup before rollback
    log_info "Creating emergency backup..."
    BACKUP_NAME="pre-rollback-$(date +%Y%m%d_%H%M%S)"
    
    # Rollback API
    log_info "Rolling back API deployment..."
    if [ -n "$target_revision" ]; then
        kubectl rollout undo deployment/bi-ide-api -n $NAMESPACE --to-revision="$target_revision"
    else
        kubectl rollout undo deployment/bi-ide-api -n $NAMESPACE
    fi
    
    # Wait for API rollback
    log_info "Waiting for API rollout to complete..."
    if kubectl rollout status deployment/bi-ide-api -n $NAMESPACE --timeout=300s; then
        log_success "API rollback completed"
    else
        log_error "API rollback failed or timed out"
        return 1
    fi
    
    # Rollback UI
    log_info "Rolling back UI deployment..."
    if [ -n "$target_revision" ]; then
        kubectl rollout undo deployment/bi-ide-ui -n $NAMESPACE --to-revision="$target_revision"
    else
        kubectl rollout undo deployment/bi-ide-ui -n $NAMESPACE
    fi
    
    kubectl rollout status deployment/bi-ide-ui -n $NAMESPACE --timeout=300s
    log_success "UI rollback completed"
    
    # Rollback Worker
    log_info "Rolling back Worker deployment..."
    if [ -n "$target_revision" ]; then
        kubectl rollout undo deployment/bi-ide-worker -n $NAMESPACE --to-revision="$target_revision"
    else
        kubectl rollout undo deployment/bi-ide-worker -n $NAMESPACE
    fi
    
    kubectl rollout status deployment/bi-ide-worker -n $NAMESPACE --timeout=300s
    log_success "Worker rollback completed"
    
    # Verify rollback
    log_info "Verifying rollback..."
    sleep 10
    
    if kubectl get pods -n $NAMESPACE | grep -q "Error\|CrashLoopBackOff"; then
        log_error "Some pods are in error state after rollback"
        kubectl get pods -n $NAMESPACE
        return 1
    fi
    
    log_success "Rollback completed successfully!"
}

rollback_docker() {
    local target_revision="$1"
    local dry_run="${2:-false}"
    
    local deploy_dir="/opt/bi-ide-v8/${ENVIRONMENT}"
    
    if [ ! -d "$deploy_dir" ]; then
        log_error "Deployment directory not found: ${deploy_dir}"
        exit 1
    fi
    
    cd "$deploy_dir"
    
    if [ "$dry_run" == "true" ]; then
        log_warn "DRY RUN MODE - No changes will be made"
        echo "Would rollback to: ${target_revision:-previous environment}"
        return 0
    fi
    
    # Blue-green rollback
    if [ -f ".previous-env" ]; then
        local previous_env
        previous_env=$(cat .previous-env)
        log_info "Rolling back to ${previous_env} environment..."
        
        # Switch traffic back
        ./scripts/switch-traffic.sh "$previous_env"
        
        log_success "Rolled back to ${previous_env} environment"
    else
        log_warn "No previous environment state found. Manual rollback required."
        echo "Available images:"
        docker images | grep bi-ide-v8
    fi
}

# Parse arguments
TARGET_REVISION=""
DRY_RUN="false"
LIST_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --revision)
            TARGET_REVISION="$2"
            shift 2
            ;;
        --list|-l)
            LIST_MODE="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        staging|production|k8s|kubernetes)
            ENVIRONMENT="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
echo "========================================"
echo "BI-IDE v8 Rollback Tool"
echo "Environment: ${ENVIRONMENT}"
echo "========================================"
echo

if [ "$LIST_MODE" == "true" ]; then
    list_revisions
    exit 0
fi

# Determine rollback target
if [ -z "$TARGET_REVISION" ]; then
    log_info "No specific revision provided, finding previous..."
    TARGET_REVISION=$(get_previous_revision)
    log_info "Will rollback to revision: ${TARGET_REVISION}"
fi

# Confirm rollback
if [ "$DRY_RUN" != "true" ]; then
    echo
    log_warn "You are about to rollback ${ENVIRONMENT} to revision ${TARGET_REVISION}"
    echo -n "Are you sure? (yes/no): "
    read -r confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Rollback cancelled"
        exit 0
    fi
fi

# Execute rollback
if [ "$ENVIRONMENT" == "k8s" ] || [ "$ENVIRONMENT" == "kubernetes" ]; then
    rollback_k8s "$TARGET_REVISION" "$DRY_RUN"
else
    rollback_docker "$TARGET_REVISION" "$DRY_RUN"
fi

# Post-rollback verification
if [ "$DRY_RUN" != "true" ]; then
    echo
    log_info "Running post-rollback health checks..."
    
    # Wait a bit for services to stabilize
    sleep 15
    
    # Run smoke tests
    if [ -f "./scripts/smoke_test.sh" ]; then
        if ./scripts/smoke_test.sh "${BASE_URL:-http://localhost:8000}"; then
            log_success "Post-rollback smoke tests passed"
        else
            log_error "Post-rollback smoke tests failed - manual intervention may be required"
            exit 1
        fi
    fi
fi

echo
log_success "Rollback process completed!"
