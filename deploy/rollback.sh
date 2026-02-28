#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BI-IDE v8 - Rollback Script
# نص استعادة النسخة السابقة لـ BI-IDE
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# الإعدادات / Settings
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/deploy/backup"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/rollback_${TIMESTAMP}.log"
DRY_RUN=false
QUICK_MODE=false

# ═══════════════════════════════════════════════════════════════════
# الألوان / Colors
# ═══════════════════════════════════════════════════════════════════
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

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
# عرض النسخ المتاحة / List Available Versions
# ═══════════════════════════════════════════════════════════════════
list_versions() {
    log_step "Listing available backup versions..."
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    echo ""
    echo -e "${CYAN}Available Backup Versions:${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    printf "${YELLOW}%-5s %-20s %-30s %-15s${NC}\n" "No." "Timestamp" "Type" "Size"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    local count=0
    local backups=()
    
    for backup in "$BACKUP_DIR"/backup_*; do
        if [ -d "$backup" ]; then
            count=$((count + 1))
            backups+=("$backup")
            local basename=$(basename "$backup")
            local timestamp=$(echo "$basename" | sed 's/backup_//' | sed 's/_/ /')
            local type="Unknown"
            local size=$(du -sh "$backup" 2>/dev/null | cut -f1 || echo "N/A")
            
            # قراءة نوع النسخ من ملف المعلومات
            if [ -f "$backup/backup_info.txt" ]; then
                type=$(grep "Backup type:" "$backup/backup_info.txt" | cut -d: -f2 | xargs)
            fi
            
            printf "%-5s %-20s %-30s %-15s\n" "$count" "$timestamp" "$type" "$size"
        fi
    done
    
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $count -eq 0 ]; then
        log_warn "No backups found"
        return 1
    fi
    
    echo ""
    log_success "Found $count backup(s)"
    
    # حفظ القائمة للاستخدام اللاحق
    printf "%s\n" "${backups[@]}" > /tmp/bi_ide_backups.txt
    
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# استعادة سريعة / Quick Rollback
# ═══════════════════════════════════════════════════════════════════
quick_rollback() {
    log_step "Performing quick rollback to previous version..."
    
    # العثور على آخر نسخة احتياطية
    local latest_backup=$(ls -td "$BACKUP_DIR"/backup_* 2>/dev/null | head -1)
    
    if [ -z "$latest_backup" ]; then
        log_error "No backup found for quick rollback"
        return 1
    fi
    
    log_info "Rolling back to: $(basename "$latest_backup")"
    
    # استدعاء دالة الاستعادة
    restore_backup "$latest_backup"
}

# ═══════════════════════════════════════════════════════════════════
# استعادة نسخة محددة / Restore Specific Backup
# ═══════════════════════════════════════════════════════════════════
restore_backup() {
    local backup_path="${1:-}"
    
    if [ -z "$backup_path" ]; then
        log_error "No backup path specified"
        return 1
    fi
    
    if [ ! -d "$backup_path" ]; then
        log_error "Backup not found: $backup_path"
        return 1
    fi
    
    log_step "Restoring from backup: $(basename "$backup_path")"
    
    # إنشاء نسخة احتياطية من الحالة الحالية أولاً
    log_info "Creating safety backup of current state..."
    local safety_backup="$BACKUP_DIR/pre_rollback_${TIMESTAMP}"
    execute_or_dry "mkdir -p '$safety_backup'" "Create safety backup directory"
    
    if [ -f "$PROJECT_ROOT/.env" ]; then
        execute_or_dry "cp '$PROJECT_ROOT/.env' '$safety_backup/'" "Backup current .env"
    fi
    
    # حفظ commit الحالي
    local current_commit=$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo "unknown")
    echo "Pre-rollback commit: $current_commit" > "$safety_backup/rollback_info.txt"
    
    # استعادة الإعدادات
    if [ -f "$backup_path/.env" ]; then
        log_info "Restoring .env file..."
        execute_or_dry "cp '$backup_path/.env' '$PROJECT_ROOT/'" "Restore .env file"
        execute_or_dry "chmod 600 '$PROJECT_ROOT/.env'" "Set .env permissions"
    fi
    
    # استعادة قاعدة البيانات
    restore_database "$backup_path"
    
    # استعادة الكود
    restore_code "$backup_path"
    
    # إعادة تشغيل الخدمات
    restart_services
    
    log_success "Rollback completed successfully"
}

# ═══════════════════════════════════════════════════════════════════
# استعادة قاعدة البيانات / Restore Database
# ═══════════════════════════════════════════════════════════════════
restore_database() {
    local backup_path="$1"
    local db_backup="$backup_path/database.sql"
    
    log_step "Restoring database..."
    
    if [ ! -f "$db_backup" ]; then
        log_warn "Database backup not found, skipping database restore"
        return 0
    fi
    
    # قراءة إعدادات قاعدة البيانات
    local db_url=""
    if [ -f "$PROJECT_ROOT/.env" ]; then
        db_url=$(grep DATABASE_URL "$PROJECT_ROOT/.env" | cut -d= -f2-)
    fi
    
    if [ -z "$db_url" ]; then
        log_warn "DATABASE_URL not found, skipping database restore"
        return 0
    fi
    
    # استخراج معلومات الاتصال
    local db_user=$(echo "$db_url" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    local db_pass=$(echo "$db_url" | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    local db_host=$(echo "$db_url" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    local db_port=$(echo "$db_url" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    local db_name=$(echo "$db_url" | sed -n 's/.*\/\(.*\)$/\1/p')
    
    # إنشاء نسخة احتياطية من قاعدة البيانات الحالية
    log_info "Creating current database backup..."
    execute_or_dry "PGPASSWORD='$db_pass' pg_dump -h '$db_host' -p '$db_port' -U '$db_user' '$db_name' > '$BACKUP_DIR/pre_rollback_db_${TIMESTAMP}.sql' 2>/dev/null || true" "Backup current database"
    
    # استعادة قاعدة البيانات
    log_info "Restoring database from backup..."
    execute_or_dry "PGPASSWORD='$db_pass' psql -h '$db_host' -p '$db_port' -U '$db_user' -d '$db_name' -f '$db_backup' 2>/dev/null || true" "Restore database"
    
    log_success "Database restored"
}

# ═══════════════════════════════════════════════════════════════════
# استعادة الكود / Restore Code
# ═══════════════════════════════════════════════════════════════════
restore_code() {
    local backup_path="$1"
    
    log_step "Restoring code..."
    
    # استعادة commit Git إذا كان متوفراً
    if [ -f "$backup_path/backup_info.txt" ]; then
        local commit=$(grep "Git commit:" "$backup_path/backup_info.txt" | cut -d: -f2 | xargs)
        
        if [ -n "$commit" ] && [ "$commit" != "N/A" ]; then
            log_info "Restoring Git commit: $commit"
            execute_or_dry "cd '$PROJECT_ROOT' && git checkout '$commit'" "Checkout Git commit"
        fi
    fi
    
    log_success "Code restored"
}

# ═══════════════════════════════════════════════════════════════════
# إعادة تشغيل الخدمات / Restart Services
# ═══════════════════════════════════════════════════════════════════
restart_services() {
    log_step "Restarting services..."
    
    # Docker Compose
    if [ -f "$PROJECT_ROOT/docker-compose.prod.yml" ]; then
        execute_or_dry "cd '$PROJECT_ROOT' && docker-compose -f docker-compose.prod.yml restart" "Restart Docker services"
    fi
    
    # Systemd services
    local services=("bi-ide-api" "bi-ide-worker" "bi-ide-gpu-trainer")
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            execute_or_dry "systemctl restart '$service'" "Restart $service"
        fi
    done
    
    log_success "Services restarted"
}

# ═══════════════════════════════════════════════════════════════════
# التحقق بعد الاستعادة / Verify After Rollback
# ═══════════════════════════════════════════════════════════════════
verify_rollback() {
    log_step "Verifying rollback..."
    
    local all_good=true
    
    # فحص API
    if curl -sf http://localhost:8000/health &>/dev/null; then
        log_success "API health check: PASSED"
    else
        log_error "API health check: FAILED"
        all_good=false
    fi
    
    # فحص قاعدة البيانات
    if [ -f "$PROJECT_ROOT/.env" ]; then
        local db_url=$(grep DATABASE_URL "$PROJECT_ROOT/.env" | cut -d= -f2-)
        if [ -n "$db_url" ]; then
            # محاولة الاتصال بقاعدة البيانات
            if python3 -c "import asyncio; from sqlalchemy import create_engine; engine = create_engine('$db_url'.replace('+asyncpg', '')); engine.connect()" 2>/dev/null; then
                log_success "Database connection: PASSED"
            else
                log_warn "Database connection: FAILED"
            fi
        fi
    fi
    
    # فحص Git
    local current_commit=$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null)
    log_info "Current commit: ${current_commit:0:12}"
    
    if [ "$all_good" = true ]; then
        log_success "Rollback verification completed successfully"
        return 0
    else
        log_error "Rollback verification failed"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════
# اختيار نسخة للاستعادة / Select Version to Restore
# ═══════════════════════════════════════════════════════════════════
select_version() {
    if ! list_versions; then
        return 1
    fi
    
    echo ""
    read -p "Enter backup number to restore (or 'q' to quit): " choice
    
    if [ "$choice" = "q" ]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
    
    if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
        log_error "Invalid selection"
        return 1
    fi
    
    local backup_path=$(sed -n "${choice}p" /tmp/bi_ide_backups.txt 2>/dev/null)
    
    if [ -z "$backup_path" ] || [ ! -d "$backup_path" ]; then
        log_error "Invalid backup selection"
        return 1
    fi
    
    echo ""
    log_warn "You are about to restore: $(basename "$backup_path")"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Rollback cancelled"
        exit 0
    fi
    
    restore_backup "$backup_path"
}

# ═══════════════════════════════════════════════════════════════════
# عرض الاستخدام / Show Usage
# ═══════════════════════════════════════════════════════════════════
show_usage() {
    cat << EOF
${CYAN}BI-IDE v8 - Rollback Script${NC}
${MAGENTA}============================${NC}

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    list                List available backup versions
    quick               Quick rollback to latest backup
    restore <version>   Restore specific version
    verify              Verify current installation

Options:
    -d, --dry-run       Run in dry-run mode (no actual changes)
    -y, --yes           Skip confirmation prompts
    -h, --help          Show this help message

Examples:
    # عرض النسخ المتاحة
    $0 list

    # استعادة سريعة
    $0 quick

    # استعادة نسخة محددة
    $0 restore backup_20240115_120000

    # وضع التشغيل الجاف
    $0 --dry-run quick

EOF
}

# ═══════════════════════════════════════════════════════════════════
# الدالة الرئيسية / Main Function
# ═══════════════════════════════════════════════════════════════════
main() {
    local command=""
    local backup_target=""
    local skip_confirm=false
    
    # معالجة المعاملات
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -y|--yes)
                skip_confirm=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            list|quick|restore|verify)
                command="$1"
                shift
                ;;
            backup_*)
                backup_target="$1"
                shift
                ;;
            *)
                if [ -z "$command" ]; then
                    log_error "Unknown command: $1"
                    show_usage
                    exit 1
                fi
                backup_target="$1"
                shift
                ;;
        esac
    done
    
    # إنشاء الأدلة
    mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    touch "$LOG_FILE"
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   BI-IDE v8 - Rollback Tool${NC}"
    echo -e "${CYAN}   Mode: $([ "$DRY_RUN" = true ] && echo 'DRY-RUN' || echo 'LIVE')${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # تنفيذ الأمر
    case "$command" in
        list)
            list_versions
            ;;
        quick)
            if [ "$skip_confirm" = false ]; then
                log_warn "This will rollback to the latest backup"
                read -p "Continue? (yes/no): " confirm
                if [ "$confirm" != "yes" ]; then
                    log_info "Cancelled"
                    exit 0
                fi
            fi
            quick_rollback
            verify_rollback
            ;;
        restore)
            if [ -z "$backup_target" ]; then
                select_version
            else
                if [[ ! "$backup_target" == */* ]]; then
                    backup_target="$BACKUP_DIR/$backup_target"
                fi
                restore_backup "$backup_target"
            fi
            verify_rollback
            ;;
        verify)
            verify_rollback
            ;;
        "")
            # وضع تفاعلي
            echo -e "${CYAN}Select an option:${NC}"
            echo "  1) List available versions"
            echo "  2) Quick rollback"
            echo "  3) Restore specific version"
            echo "  4) Verify current installation"
            echo "  q) Quit"
            echo ""
            read -p "Choice: " choice
            
            case "$choice" in
                1) list_versions ;;
                2) quick_rollback; verify_rollback ;;
                3) select_version; verify_rollback ;;
                4) verify_rollback ;;
                q) exit 0 ;;
                *) log_error "Invalid choice" ;;
            esac
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   ROLLBACK OPERATION COMPLETE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Log file:${NC} $LOG_FILE"
    echo ""
}

# تشغيل الدالة الرئيسية
main "$@"
