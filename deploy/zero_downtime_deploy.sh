#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Zero-Downtime Deployment Script for BI-IDE
# نشر بدون توقف - يستخدم Blue-Green Deployment
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
BLUE_ENV="bi-ide-blue"
GREEN_ENV="bi-ide-green"
TRAEFIK_NETWORK="traefik-proxy"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# التحقق من البيئة الحالية
get_current_env() {
    if docker ps --format "{{.Names}}" | grep -q "^${BLUE_ENV}"; then
        echo "blue"
    elif docker ps --format "{{.Names}}" | grep -q "^${GREEN_ENV}"; then
        echo "green"
    else
        echo "none"
    fi
}

# التحقق من صحة الخدمة
health_check() {
    local env=$1
    local max_attempts=30
    local attempt=1
    
    log_info "فحص صحة البيئة ${env}..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log_success "البيئة ${env} صحية!"
            return 0
        fi
        
        log_warning "محاولة ${attempt}/${max_attempts}... انتظر 2 ثانية"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "فشل فحص الصحة للبيئة ${env}"
    return 1
}

# نشر البيئة الجديدة
deploy_new_env() {
    local target_env=$1
    local compose_override="docker-compose.${target_env}.yml"
    
    log_info "نشر البيئة ${target_env}..."
    
    # بناء الصورة
    docker-compose -f ${COMPOSE_FILE} -f ${compose_override} build --no-cache
    
    # تشغيل الخدمات الجديدة
    docker-compose -f ${COMPOSE_FILE} -f ${compose_override} up -d
    
    # انتظار بدء الخدمات
    sleep 10
    
    log_success "تم نشر البيئة ${target_env}"
}

# تبديل حركة المرور
switch_traffic() {
    local target_env=$1
    
    log_info "تبديل حركة المرور إلى ${target_env}..."
    
    # تحديث Traefik أو Nginx للإشارة إلى البيئة الجديدة
    if [ -f "deploy/switch_traffic.sh" ]; then
        ./deploy/switch_traffic.sh ${target_env}
    else
        # تحديث Nginx upstream
        sed -i "s/server bi-ide-.*:8000/server ${target_env}:8000/g" deploy/nginx.conf
        docker exec bi-ide-nginx nginx -s reload
    fi
    
    log_success "تم تبديل حركة المرور إلى ${target_env}"
}

# إيقاف البيئة القديمة
stop_old_env() {
    local old_env=$1
    local compose_override="docker-compose.${old_env}.yml"
    
    log_info "إيقاف البيئة القديمة ${old_env}..."
    
    # الانتظار قليلاً للتأكد من اكتمال الطلبات الحالية
    sleep 5
    
    # إيقاف الخدمات القديمة
    docker-compose -f ${COMPOSE_FILE} -f ${compose_override} down
    
    log_success "تم إيقاف البيئة ${old_env}"
}

# النشر بدون توقف
zero_downtime_deploy() {
    log_info "بدء النشر بدون توقف..."
    
    # تحديد البيئة الحالية
    current_env=$(get_current_env)
    log_info "البيئة الحالية: ${current_env}"
    
    # تحديد البيئة الهدف
    if [ "$current_env" = "blue" ]; then
        target_env="green"
    else
        target_env="blue"
    fi
    
    log_info "البيئة الهدف: ${target_env}"
    
    # 1. نشر البيئة الجديدة
    deploy_new_env ${target_env}
    
    # 2. فحص صحة البيئة الجديدة
    if ! health_check ${target_env}; then
        log_error "فشل فحص الصحة! إلغاء النشر..."
        docker-compose -f ${COMPOSE_FILE} -f docker-compose.${target_env}.yml down
        exit 1
    fi
    
    # 3. تبديل حركة المرور
    switch_traffic ${target_env}
    
    # 4. إيقاف البيئة القديمة (إذا وجدت)
    if [ "$current_env" != "none" ]; then
        stop_old_env ${current_env}
    fi
    
    log_success "✅ تم النشر بنجاح بدون توقف!"
    log_info "البيئة النشطة الآن: ${target_env}"
}

# التراجع عن النشر
rollback() {
    log_warning "التراجع عن النشر..."
    
    current_env=$(get_current_env)
    
    if [ "$current_env" = "blue" ]; then
        rollback_env="green"
    else
        rollback_env="blue"
    fi
    
    # التأكد من وجود البيئة القديمة
    if docker ps --format "{{.Names}}" | grep -q "^${rollback_env}"; then
        switch_traffic ${rollback_env}
        stop_old_env ${current_env}
        log_success "✅ تم التراجع بنجاح!"
    else
        log_error "❌ لا توجد بيئة سابقة للتراجع إليها!"
        exit 1
    fi
}

# عرض الحالة
status() {
    log_info "حالة البيئات:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    current_env=$(get_current_env)
    echo -e "البيئة النشطة: ${GREEN}${current_env}${NC}"
    
    echo ""
    echo "الحاويات النشطة:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(bi-ide|blue|green)" || echo "لا توجد حاويات نشطة"
}

# الاستخدام
usage() {
    echo "Usage: $0 [deploy|rollback|status]"
    echo ""
    echo "Commands:"
    echo "  deploy    نشر بدون توقف"
    echo "  rollback  التراجع عن آخر نشر"
    echo "  status    عرض حالة البيئات"
    echo ""
    echo "Example:"
    echo "  $0 deploy"
}

# Main
main() {
    case "${1:-deploy}" in
        deploy)
            zero_downtime_deploy
            ;;
        rollback)
            rollback
            ;;
        status)
            status
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            log_error "أمر غير معروف: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
