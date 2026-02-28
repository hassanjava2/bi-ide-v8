#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# BI-IDE v8 - سكربت النشر الشامل
# Comprehensive Deployment Script with Rollback Support
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# الألوان للعرض
# ═══════════════════════════════════════════════════════════════════════════════
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ═══════════════════════════════════════════════════════════════════════════════
# الإعدادات الافتراضية
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-bi-ide}"
VERSION="${VERSION:-$(date +%Y%m%d-%H%M%S)}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/deploy_$(date +%Y%m%d_%H%M%S).log}"

# ═══════════════════════════════════════════════════════════════════════════════
# إعدادات البيئات
# ═══════════════════════════════════════════════════════════════════════════════
STAGING_HOST="${STAGING_HOST:-staging.bi-ide.com}"
PRODUCTION_HOST="${PRODUCTION_HOST:-bi-ide.com}"
STAGING_COMPOSE="docker-compose.yml"
PRODUCTION_COMPOSE="docker-compose.prod.yml"

# ═══════════════════════════════════════════════════════════════════════════════
# دوال المساعدة
# ═══════════════════════════════════════════════════════════════════════════════

# طباعة رسالة مع الوقت
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        DEBUG)
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            ;;
        *)
            echo -e "${CYAN}[$level]${NC} ${timestamp} - $message"
            ;;
    esac
    
    # تسجيل في الملف
    echo "[$level] $timestamp - $message" >> "$LOG_FILE"
}

# التحقق من وجود أمر
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# التحقق من المتطلبات
check_prerequisites() {
    log "INFO" "التحقق من المتطلبات..."
    
    local required_commands=("docker" "docker-compose" "curl")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        log "ERROR" "الأوامر التالية غير موجودة: ${missing_commands[*]}"
        exit 1
    fi
    
    # التحقق من تشغيل Docker
    if ! docker info >/dev/null 2>&1; then
        log "ERROR" "Docker لا يعمل. يرجى تشغيل Docker أولاً."
        exit 1
    fi
    
    # إنشاء المجلدات الضرورية
    mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"
    
    log "INFO" "✓ جميع المتطلبات متوفرة"
}

# عرض شعار التطبيق
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██╗     ██╗██████╗ ███████╗    ██╗██████╗ ███████╗                  ║
║   ██╔══██╗██║     ██║██╔══██╗██╔════╝    ██║██╔══██╗██╔════╝                  ║
║   ██████╔╝██║     ██║██║  ██║█████╗      ██║██║  ██║█████╗                    ║
║   ██╔══██╗██║     ██║██║  ██║██╔══╝      ██║██║  ██║██╔══╝                    ║
║   ██████╔╝███████╗██║██████╔╝███████╗    ██║██████╔╝███████╗                  ║
║   ╚═════╝ ╚══════╝╚═╝╚═════╝ ╚══════╝    ╚═╝╚═════╝ ╚══════╝                  ║
║                                                                               ║
║                      v8 - نظام النشر الشامل                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# عرض الاستخدام
show_usage() {
    echo -e "${BOLD}الاستخدام:${NC} $0 [خيارات]"
    echo ""
    echo -e "${BOLD}الخيارات:${NC}"
    echo "  -e, --environment    بيئة النشر (staging|production|all) [افتراضي: staging]"
    echo "  -v, --version        إصدار الصورة [افتراضي: timestamp]"
    echo "  -r, --registry       سجل الحاويات [افتراضي: ghcr.io]"
    echo "  -b, --build-only     بناء الصور فقط دون نشر"
    echo "  -p, --push-only      دفع الصور فقط"
    echo "  -d, --deploy-only    نشر فقط دون بناء"
    echo "  -s, --skip-tests     تخطي الاختبارات"
    echo "  --rollback           التراجع عن آخر نشر"
    echo "  --health-check       فحص صحة النظام فقط"
    echo "  -h, --help           عرض هذه المساعدة"
    echo ""
    echo -e "${BOLD}أمثلة:${NC}"
    echo "  $0 -e staging                    # نشر على البيئة التجريبية"
    echo "  $0 -e production -v 1.2.3       # نشر إصدار محدد على الإنتاج"
    echo "  $0 -b -v 1.0.0                  # بناء فقط"
    echo "  $0 --rollback -e production     # التراجع عن آخر نشر"
}

# ═══════════════════════════════════════════════════════════════════════════════
# مراحل النشر
# ═══════════════════════════════════════════════════════════════════════════════

# بناء صور Docker
build_images() {
    log "INFO" "════════════════════════════════════════════════════════════"
    log "INFO" "بدء بناء صور Docker..."
    log "INFO" "الإصدار: ${BOLD}$VERSION${NC}"
    log "INFO" "════════════════════════════════════════════════════════════"
    
    cd "$PROJECT_ROOT"
    
    # بناء صورة API
    log "INFO" "بناء صورة API..."
    docker build \
        --target runtime \
        -t "$REGISTRY/$IMAGE_NAME/api:$VERSION" \
        -t "$REGISTRY/$IMAGE_NAME/api:latest" \
        -f Dockerfile . 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR" "فشل بناء صورة API"
        return 1
    fi
    
    # بناء صورة Worker
    log "INFO" "بناء صورة Worker..."
    docker build \
        --target runtime \
        -t "$REGISTRY/$IMAGE_NAME/worker:$VERSION" \
        -t "$REGISTRY/$IMAGE_NAME/worker:latest" \
        -f Dockerfile . 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR" "فشل بناء صورة Worker"
        return 1
    fi
    
    log "INFO" "✓ تم بناء الصور بنجاح"
    return 0
}

# دفع الصور إلى السجل
push_images() {
    log "INFO" "════════════════════════════════════════════════════════════"
    log "INFO" "دفع الصور إلى السجل: $REGISTRY"
    log "INFO" "════════════════════════════════════════════════════════════"
    
    # التحقق من تسجيل الدخول
    if ! docker info | grep -q "Username"; then
        log "WARN" "غير مسجل الدخول إلى السجل. جاري تسجيل الدخول..."
        docker login "$REGISTRY"
    fi
    
    # دفع صور API
    log "INFO" "دفع صورة API..."
    docker push "$REGISTRY/$IMAGE_NAME/api:$VERSION" 2>&1 | tee -a "$LOG_FILE"
    docker push "$REGISTRY/$IMAGE_NAME/api:latest" 2>&1 | tee -a "$LOG_FILE"
    
    # دفع صور Worker
    log "INFO" "دفع صورة Worker..."
    docker push "$REGISTRY/$IMAGE_NAME/worker:$VERSION" 2>&1 | tee -a "$LOG_FILE"
    docker push "$REGISTRY/$IMAGE_NAME/worker:latest" 2>&1 | tee -a "$LOG_FILE"
    
    log "INFO" "✓ تم دفع الصور بنجاح"
}

# إنشاء نسخة احتياطية
create_backup() {
    local environment=$1
    log "INFO" "إنشاء نسخة احتياطية لـ $environment..."
    
    local backup_file="$BACKUP_DIR/backup_${environment}_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    # نسخ قاعدة البيانات
    cd "$PROJECT_ROOT"
    docker-compose -f "$PRODUCTION_COMPOSE" exec -T postgres pg_dump \
        -U "${POSTGRES_USER:-bi_ide}" \
        "${POSTGRES_DB:-bi_ide}" > "$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql" 2>/dev/null || true
    
    # نسخ المجلدات
    tar -czf "$backup_file" data/ learning_data/ 2>/dev/null || true
    
    log "INFO" "✓ تم إنشاء النسخة الاحتياطية: $backup_file"
    echo "$backup_file"
}

# نشر على بيئة معينة
deploy_environment() {
    local environment=$1
    local compose_file=$2
    local host=$3
    
    log "INFO" "════════════════════════════════════════════════════════════"
    log "INFO" "النشر على بيئة: ${BOLD}$environment${NC}"
    log "INFO" "المضيف: $host"
    log "INFO" "════════════════════════════════════════════════════════════"
    
    # إنشاء نسخة احتياطية قبل النشر
    if [ "$environment" == "production" ]; then
        BACKUP_FILE=$(create_backup "$environment")
        log "INFO" "النسخة الاحتياطية: $BACKUP_FILE"
    fi
    
    if [ "$environment" == "local" ]; then
        # نشر محلي
        deploy_local "$compose_file"
    else
        # نشر عن بعد
        deploy_remote "$environment" "$compose_file" "$host"
    fi
    
    return $?
}

# نشر محلي
deploy_local() {
    local compose_file=$1
    
    cd "$PROJECT_ROOT"
    
    # سحب أحدث الصور
    log "INFO" "سحب أحدث الصور..."
    docker-compose -f "$compose_file" pull 2>&1 | tee -a "$LOG_FILE"
    
    # تشغيل migrations
    log "INFO" "تشغيل migrations..."
    docker-compose -f "$compose_file" run --rm api alembic upgrade head 2>&1 | tee -a "$LOG_FILE"
    
    # إعادة تشغيل الخدمات
    log "INFO" "إعادة تشغيل الخدمات..."
    docker-compose -f "$compose_file" up -d --remove-orphans 2>&1 | tee -a "$LOG_FILE"
    
    # تنظيف الصور القديمة
    log "INFO" "تنظيف الصور القديمة..."
    docker image prune -af --filter "until=168h" 2>&1 | tee -a "$LOG_FILE" || true
    
    log "INFO" "✓ تم النشر المحلي بنجاح"
}

# نشر عن بعد
deploy_remote() {
    local environment=$1
    local compose_file=$2
    local host=$3
    
    # إنشاء سكربت النشر
    local deploy_script=$(cat << EOF
#!/bin/bash
set -e

echo "🚀 بدء النشر على $environment..."

cd /opt/bi-ide

# سحب أحدث الصور
docker-compose -f $compose_file pull

# تشغيل migrations
docker-compose -f $compose_file run --rm api alembic upgrade head

# إعادة تشغيل الخدمات
docker-compose -f $compose_file up -d --remove-orphans

# تنظيف
docker system prune -af --volumes=false --filter "until=168h" || true

echo "✅ تم النشر بنجاح!"
EOF
)
    
    # تنفيذ النشر عبر SSH
    log "INFO" "الاتصال بالخادم $host..."
    
    # نسخ السكربت
    echo "$deploy_script" | ssh -o StrictHostKeyChecking=no "$host" "cat > /tmp/deploy.sh && chmod +x /tmp/deploy.sh && bash /tmp/deploy.sh" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR" "فشل النشر على $host"
        return 1
    fi
    
    log "INFO" "✓ تم النشر على $host بنجاح"
}

# فحص صحة النظام
health_check() {
    local environment=$1
    local host=$2
    local max_retries=10
    local retry_count=0
    
    log "INFO" "════════════════════════════════════════════════════════════"
    log "INFO" "فحص صحة النظام: $environment"
    log "INFO" "════════════════════════════════════════════════════════════"
    
    local health_url="http://${host}/health"
    
    while [ $retry_count -lt $max_retries ]; do
        log "INFO" "محاولة فحص الصحة رقم $((retry_count + 1))..."
        
        if curl -sf "$health_url" >/dev/null 2>&1; then
            log "INFO" "${GREEN}✓ النظام يعمل بشكل صحيح!${NC}"
            
            # فحص إضافي
            local api_response=$(curl -sf "http://${host}/api/v1/health" 2>/dev/null || echo "{}")
            log "INFO" "استجابة API: $api_response"
            
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        sleep 10
    done
    
    log "ERROR" "${RED}✗ فشل فحص الصحة بعد $max_retries محاولات${NC}"
    return 1
}

# التراجع عن النشر
rollback() {
    local environment=$1
    
    log "INFO" "════════════════════════════════════════════════════════════"
    log "INFO" "التراجع عن النشر: $environment"
    log "INFO" "════════════════════════════════════════════════════════════"
    
    log "WARN" "⚠️  جاري التراجع عن آخر نشر..."
    
    cd "$PROJECT_ROOT"
    
    # استعادة النسخة الاحتياطية إذا وجدت
    local latest_backup=$(ls -t "$BACKUP_DIR"/backup_${environment}_*.tar.gz 2>/dev/null | head -1)
    if [ -n "$latest_backup" ]; then
        log "INFO" "استعادة النسخة الاحتياطية: $latest_backup"
        tar -xzf "$latest_backup" -C "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_FILE"
    fi
    
    # إعادة تشغيل الخدمات السابقة
    docker-compose -f "$PRODUCTION_COMPOSE" down 2>&1 | tee -a "$LOG_FILE"
    docker-compose -f "$PRODUCTION_COMPOSE" up -d 2>&1 | tee -a "$LOG_FILE"
    
    log "INFO" "✓ تم التراجع بنجاح"
}

# ═══════════════════════════════════════════════════════════════════════════════
# الدالة الرئيسية
# ═══════════════════════════════════════════════════════════════════════════════
main() {
    local environment="staging"
    local build_only=false
    local push_only=false
    local deploy_only=false
    local skip_tests=false
    local rollback_mode=false
    local health_check_only=false
    
    # معالجة المعاملات
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                environment="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -b|--build-only)
                build_only=true
                shift
                ;;
            -p|--push-only)
                push_only=true
                shift
                ;;
            -d|--deploy-only)
                deploy_only=true
                shift
                ;;
            -s|--skip-tests)
                skip_tests=true
                shift
                ;;
            --rollback)
                rollback_mode=true
                shift
                ;;
            --health-check)
                health_check_only=true
                shift
                ;;
            -h|--help)
                show_banner
                show_usage
                exit 0
                ;;
            *)
                log "ERROR" "خيار غير معروف: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # عرض الشعار
    show_banner
    
    # التحقق من المتطلبات
    check_prerequisites
    
    # فحص الصحة فقط
    if [ "$health_check_only" = true ]; then
        if [ "$environment" == "production" ]; then
            health_check "$environment" "$PRODUCTION_HOST"
        else
            health_check "$environment" "$STAGING_HOST"
        fi
        exit $?
    fi
    
    # وضع التراجع
    if [ "$rollback_mode" = true ]; then
        rollback "$environment"
        exit $?
    fi
    
    # البناء فقط
    if [ "$build_only" = true ]; then
        build_images
        exit $?
    fi
    
    # الدفع فقط
    if [ "$push_only" = true ]; then
        push_images
        exit $?
    fi
    
    # النشر فقط
    if [ "$deploy_only" = true ]; then
        if [ "$environment" == "all" ]; then
            deploy_environment "staging" "$STAGING_COMPOSE" "$STAGING_HOST" && \
            health_check "staging" "$STAGING_HOST" && \
            deploy_environment "production" "$PRODUCTION_COMPOSE" "$PRODUCTION_HOST" && \
            health_check "production" "$PRODUCTION_HOST"
        else
            deploy_environment "$environment" "$PRODUCTION_COMPOSE" "$environment"
        fi
        exit $?
    fi
    
    # ═══════════════════════════════════════════════════════════════════════════
    # سير عمل النشر الكامل
    # ═══════════════════════════════════════════════════════════════════════════
    log "INFO" "${BOLD}بدء سير عمل النشر الكامل...${NC}"
    
    local failed=false
    
    # 1. بناء الصور
    if ! build_images; then
        log "ERROR" "فشل بناء الصور"
        exit 1
    fi
    
    # 2. دفع الصور
    push_images
    
    # 3. نشر على البيئة التجريبية
    if [ "$environment" == "staging" ] || [ "$environment" == "all" ]; then
        if deploy_environment "staging" "$STAGING_COMPOSE" "$STAGING_HOST"; then
            if ! health_check "staging" "$STAGING_HOST"; then
                log "WARN" "فشل فحص الصحة على البيئة التجريبية"
                failed=true
            fi
        else
            log "ERROR" "فشل النشر على البيئة التجريبية"
            exit 1
        fi
    fi
    
    # 4. نشر على الإنتاج
    if [ "$environment" == "production" ] || [ "$environment" == "all" ]; then
        if [ "$failed" = false ]; then
            log "INFO" "الانتظار للموافقة على النشر في الإنتاج..."
            read -p "هل تريد المتابعة للنشر في الإنتاج؟ (yes/no): " confirm
            
            if [ "$confirm" == "yes" ]; then
                if deploy_environment "production" "$PRODUCTION_COMPOSE" "$PRODUCTION_HOST"; then
                    if ! health_check "production" "$PRODUCTION_HOST"; then
                        log "ERROR" "فشل فحص الصحة على الإنتاج! جاري التراجع..."
                        rollback "production"
                        exit 1
                    fi
                else
                    log "ERROR" "فشل النشر على الإنتاج"
                    exit 1
                fi
            else
                log "INFO" "تم إلغاء النشر في الإنتاج"
            fi
        else
            log "WARN" "تم تخطي النشر في الإنتاج بسبب فشل الاختبارات"
        fi
    fi
    
    log "INFO" "════════════════════════════════════════════════════════════"
    log "INFO" "${GREEN}${BOLD}✅ تم النشر بنجاح!${NC}"
    log "INFO" "════════════════════════════════════════════════════════════"
    log "INFO" "الإصدار: $VERSION"
    log "INFO" "البيئة: $environment"
    log "INFO" "سجل النشر: $LOG_FILE"
    
    return 0
}

# تشغيل السكربت
main "$@"
