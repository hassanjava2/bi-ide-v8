#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BI-IDE v8 - Master Deployment Script
# النص الرئيسي لنشر BI-IDE في جميع البيئات
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# الإعدادات العامة / Global Settings
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
BACKUP_DIR="$PROJECT_ROOT/deploy/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/deploy_all_${TIMESTAMP}.log"
DRY_RUN=false
ROLLBACK_ON_FAILURE=true

# ═══════════════════════════════════════════════════════════════════
# بيئات النشر / Deployment Environments
# ═══════════════════════════════════════════════════════════════════
declare -A ENVIRONMENTS=(
    ["hostinger"]="app.bi-iq.com:root@147.93.121.163"
    ["windows"]="windows://localhost"
    ["rtx5090"]="rtx5090://localhost"
)

# ═══════════════════════════════════════════════════════════════════
# الألوان / Colors
# ═══════════════════════════════════════════════════════════════════
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════════
# دوال التسجيل / Logging Functions
# ═══════════════════════════════════════════════════════════════════
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# ═══════════════════════════════════════════════════════════════════
# دالة التشغيل الجاف / Dry Run Function
# ═══════════════════════════════════════════════════════════════════
execute_or_dry() {
    local cmd="$1"
    local description="${2:-$cmd}"
    
    if [ "$DRY_RUN" = true ]; then
        log_warn "[DRY RUN] Would execute: $description"
        return 0
    else
        log_info "Executing: $description"
        eval "$cmd"
        return $?
    fi
}

# ═══════════════════════════════════════════════════════════════════
# معالجة الأخطاء / Error Handler
# ═══════════════════════════════════════════════════════════════════
handle_error() {
    local line=$1
    local error_code=$2
    log_error "Error occurred at line $line (exit code: $error_code)"
    
    if [ "$ROLLBACK_ON_FAILURE" = true ]; then
        log_warn "Initiating rollback procedure..."
        rollback_deployment
    fi
    
    exit $error_code
}

trap 'handle_error $LINENO $?' ERR

# ═══════════════════════════════════════════════════════════════════
# إنشاء الأدلة / Create Directories
# ═══════════════════════════════════════════════════════════════════
init_directories() {
    mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    touch "$LOG_FILE"
}

# ═══════════════════════════════════════════════════════════════════
# فحص المتطلبات الأساسية / Check Prerequisites
# ═══════════════════════════════════════════════════════════════════
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local missing=()
    
    # فحص Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
        log_warn "Docker not found"
    else
        local docker_version=$(docker --version | awk '{print $3}' | tr -d ',')
        log_success "Docker: $docker_version"
    fi
    
    # فحص Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing+=("docker-compose")
        log_warn "Docker Compose not found"
    else
        log_success "Docker Compose: OK"
    fi
    
    # فحص Python
    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
        log_warn "Python3 not found"
    else
        local py_version=$(python3 --version)
        log_success "Python: $py_version"
    fi
    
    # فحص Git
    if ! command -v git &> /dev/null; then
        missing+=("git")
        log_warn "Git not found"
    else
        log_success "Git: $(git --version | awk '{print $3}')"
    fi
    
    # فحص NVIDIA (اختياري)
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA drivers: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    else
        log_warn "NVIDIA drivers not found (optional for GPU deployment)"
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing prerequisites: ${missing[*]}"
        log_info "Install with: ./scripts/install_prerequisites.sh"
        return 1
    fi
    
    log_success "All prerequisites satisfied"
}

# ═══════════════════════════════════════════════════════════════════
# النسخ الاحتياطي / Backup Function
# ═══════════════════════════════════════════════════════════════════
create_backup() {
    log_step "Creating backup..."
    
    local backup_name="backup_${TIMESTAMP}"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    execute_or_dry "mkdir -p '$backup_path'" "Create backup directory"
    
    # نسخ إعدادات البيئة
    if [ -f "$PROJECT_ROOT/.env" ]; then
        execute_or_dry "cp '$PROJECT_ROOT/.env' '$backup_path/'" "Backup .env file"
    fi
    
    # نسخ قاعدة البيانات
    if command -v pg_dump &> /dev/null && [ -f "$PROJECT_ROOT/.env" ]; then
        local db_url=$(grep DATABASE_URL "$PROJECT_ROOT/.env" 2>/dev/null | cut -d= -f2-)
        if [ -n "$db_url" ]; then
            execute_or_dry "pg_dump '$db_url' > '$backup_path/database.sql' 2>/dev/null || true" "Backup database"
        fi
    fi
    
    # نسخ Docker volumes
    execute_or_dry "docker ps -q | xargs -I{} docker commit {} '$backup_path/container_{}.tar' 2>/dev/null || true" "Backup containers"
    
    # إنشاء ملف معلومات النسخ
    cat > "$backup_path/backup_info.txt" << EOF
Backup created: $(date)
Backup type: Pre-deployment
Git commit: $(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo "N/A")
Git branch: $(cd "$PROJECT_ROOT" && git branch --show-current 2>/dev/null || echo "N/A")
EOF
    
    log_success "Backup created: $backup_path"
    echo "$backup_path" > "$BACKUP_DIR/latest.txt"
}

# ═══════════════════════════════════════════════════════════════════
# نشر قاعدة البيانات / Deploy Database
# ═══════════════════════════════════════════════════════════════════
deploy_database() {
    log_step "Deploying Database..."
    
    execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml up -d postgres redis" "Start database services"
    
    # انتظار تشغيل قاعدة البيانات
    if [ "$DRY_RUN" = false ]; then
        log_info "Waiting for database to be ready..."
        sleep 5
        local retries=0
        while ! docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T postgres pg_isready -U biide &>/dev/null; do
            sleep 2
            retries=$((retries + 1))
            if [ $retries -gt 30 ]; then
                log_error "Database failed to start"
                return 1
            fi
        done
    fi
    
    # تشغيل الترحيلات
    execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml exec -T api alembic upgrade head 2>/dev/null || true" "Run database migrations"
    
    log_success "Database deployed successfully"
}

# ═══════════════════════════════════════════════════════════════════
# نشر API / Deploy API
# ═══════════════════════════════════════════════════════════════════
deploy_api() {
    log_step "Deploying API Server..."
    
    # بناء صورة API
    execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml build api" "Build API image"
    
    # نشر API
    execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml up -d api" "Start API service"
    
    # انتظار API
    if [ "$DRY_RUN" = false ]; then
        log_info "Waiting for API to be ready..."
        sleep 5
        local retries=0
        while ! curl -sf http://localhost:8000/health &>/dev/null; do
            sleep 2
            retries=$((retries + 1))
            if [ $retries -gt 30 ]; then
                log_error "API failed to start"
                return 1
            fi
        done
    fi
    
    log_success "API deployed successfully"
}

# ═══════════════════════════════════════════════════════════════════
# نشر العمال / Deploy Workers
# ═══════════════════════════════════════════════════════════════════
deploy_workers() {
    log_step "Deploying Workers..."
    
    # بناء صورة العمال
    execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml build worker" "Build worker image"
    
    # نشر العمال
    execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml up -d worker" "Start worker service"
    
    # نشر GPU trainer إذا كان متاحاً
    if command -v nvidia-smi &> /dev/null && [ -f "$PROJECT_ROOT/docker-compose.gpu.yml" ]; then
        log_info "GPU detected - deploying GPU trainer"
        execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.gpu.yml up -d gpu-trainer" "Start GPU trainer"
    fi
    
    log_success "Workers deployed successfully"
}

# ═══════════════════════════════════════════════════════════════════
# نشر المراقبة / Deploy Monitoring
# ═══════════════════════════════════════════════════════════════════
deploy_monitoring() {
    log_step "Deploying Monitoring..."
    
    if [ -f "$PROJECT_ROOT/docker-compose.prod.yml" ]; then
        execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml --profile monitoring up -d" "Start monitoring services"
        log_success "Monitoring deployed successfully"
    else
        log_warn "Monitoring configuration not found"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# فحوصات الصحة / Health Checks
# ═══════════════════════════════════════════════════════════════════
health_check() {
    log_step "Running health checks..."
    
    local all_healthy=true
    
    # فحص API
    if curl -sf http://localhost:8000/health &>/dev/null; then
        log_success "API health check: PASSED"
    else
        log_error "API health check: FAILED"
        all_healthy=false
    fi
    
    # فحص قاعدة البيانات
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T postgres pg_isready -U biide &>/dev/null 2>/dev/null; then
        log_success "Database health check: PASSED"
    else
        log_warn "Database health check: FAILED"
    fi
    
    # فحص Redis
    if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T redis redis-cli ping | grep -q PONG 2>/dev/null; then
        log_success "Redis health check: PASSED"
    else
        log_warn "Redis health check: FAILED"
    fi
    
    if [ "$all_healthy" = true ]; then
        log_success "All critical health checks passed"
        return 0
    else
        log_error "Some health checks failed"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════
# استعادة النسخة الاحتياطية / Rollback
# ═══════════════════════════════════════════════════════════════════
rollback_deployment() {
    log_step "Rolling back deployment..."
    
    if [ -f "$BACKUP_DIR/latest.txt" ]; then
        local backup_path=$(cat "$BACKUP_DIR/latest.txt")
        log_info "Restoring from: $backup_path"
        
        # استعادة الإعدادات
        if [ -f "$backup_path/.env" ]; then
            execute_or_dry "cp '$backup_path/.env' '$PROJECT_ROOT/'" "Restore .env file"
        fi
        
        # إعادة تشغيل الخدمات
        execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml restart" "Restart services"
        
        log_success "Rollback completed"
    else
        log_error "No backup found for rollback"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# نشر إلى Hostinger / Deploy to Hostinger
# ═══════════════════════════════════════════════════════════════════
deploy_hostinger() {
    log_step "Deploying to Hostinger VPS..."
    
    local host="147.93.121.163"
    local user="root"
    
    log_info "Connecting to $host..."
    
    # نسخ الملفات
    execute_or_dry "rsync -avz --exclude '.git' --exclude '__pycache__' '$PROJECT_ROOT/' '$user@$host:/opt/bi-ide/'" "Sync files to Hostinger"
    
    # تشغيل النص على الخادم
    execute_or_dry "ssh '$user@$host' 'cd /opt/bi-ide && bash deploy/vps/deploy.sh'" "Execute remote deployment"
    
    log_success "Hostinger deployment completed"
}

# ═══════════════════════════════════════════════════════════════════
# نشر إلى Windows / Deploy to Windows
# ═══════════════════════════════════════════════════════════════════
deploy_windows() {
    log_step "Deploying to Windows..."
    
    if [ -f "$SCRIPT_DIR/deploy_windows.ps1" ]; then
        execute_or_dry "powershell.exe -File '$SCRIPT_DIR/deploy_windows.ps1'" "Execute Windows deployment"
        log_success "Windows deployment completed"
    else
        log_warn "Windows deployment script not found"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# نشر إلى RTX 5090 / Deploy to RTX 5090
# ═══════════════════════════════════════════════════════════════════
deploy_rtx5090() {
    log_step "Deploying to RTX 5090..."
    
    if [ -f "$SCRIPT_DIR/deploy_rtx.sh" ]; then
        execute_or_dry "bash '$SCRIPT_DIR/deploy_rtx.sh'" "Execute RTX 5090 deployment"
        log_success "RTX 5090 deployment completed"
    else
        log_warn "RTX 5090 deployment script not found"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# عرض الاستخدام / Show Usage
# ═══════════════════════════════════════════════════════════════════
show_usage() {
    cat << EOF
${CYAN}BI-IDE v8 - Master Deployment Script${NC}
${MAGENTA}====================================${NC}

Usage: $0 [OPTIONS] [ENVIRONMENT]

Environments:
    all         Deploy to all environments
    hostinger   Deploy to Hostinger VPS
    windows     Deploy to Windows
    rtx5090     Deploy to RTX 5090 machine

Options:
    -d, --dry-run        Run in dry-run mode (no actual changes)
    --no-rollback        Disable automatic rollback on failure
    -h, --help           Show this help message

Examples:
    # نشر في جميع البيئات
    $0 all

    # نشر في بيئة محددة
    $0 hostinger

    # وضع التشغيل الجاف
    $0 --dry-run all

    # بدون استعادة تلقائية
    $0 --no-rollback rtx5090

EOF
}

# ═══════════════════════════════════════════════════════════════════
# الدالة الرئيسية / Main Function
# ═══════════════════════════════════════════════════════════════════
main() {
    local target_env=""
    
    # معالجة المعاملات
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dry-run)
                DRY_RUN=true
                log_warn "Running in DRY-RUN mode"
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE=false
                log_warn "Rollback disabled"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                target_env="$1"
                shift
                ;;
        esac
    done
    
    if [ -z "$target_env" ]; then
        log_error "No environment specified"
        show_usage
        exit 1
    fi
    
    # بدء النشر
    init_directories
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   BI-IDE v8 - Master Deployment${NC}"
    echo -e "${CYAN}   Target: $target_env | Mode: $([ "$DRY_RUN" = true ] && echo 'DRY-RUN' || echo 'LIVE')${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    log_info "Starting deployment process..."
    
    # فحص المتطلبات
    check_prerequisites
    
    # إنشاء نسخ احتياطي
    create_backup
    
    # النشر حسب البيئة
    case "$target_env" in
        all)
            deploy_database
            deploy_api
            deploy_workers
            deploy_monitoring
            health_check
            log_success "All services deployed successfully!"
            ;;
        hostinger)
            deploy_hostinger
            ;;
        windows)
            deploy_windows
            ;;
        rtx5090)
            deploy_rtx5090
            ;;
        *)
            log_error "Unknown environment: $target_env"
            show_usage
            exit 1
            ;;
    esac
    
    # ملخص النشر
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   DEPLOYMENT COMPLETE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Log file:${NC} $LOG_FILE"
    echo -e "  ${CYAN}Backup:${NC} $BACKUP_DIR/backup_$TIMESTAMP"
    echo ""
    echo -e "  ${CYAN}Health Check:${NC} http://localhost:8000/health"
    echo -e "  ${CYAN}API Docs:${NC} http://localhost:8000/docs"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
}

# تشغيل الدالة الرئيسية
main "$@"
