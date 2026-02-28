#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BI-IDE v8 - RTX 5090 Deployment Script
# نص نشر BI-IDE على جهاز RTX 5090
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# الإعدادات / Settings
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/deploy_rtx_${TIMESTAMP}.log"
DRY_RUN=false

# إعدادات GPU
CUDA_VERSION="12.4"
MIN_DRIVER_VERSION="550.0"
GPU_MEMORY_GB=32

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
# معالجة الأخطاء / Error Handler
# ═══════════════════════════════════════════════════════════════════
handle_error() {
    local line=$1
    local error_code=$2
    log_error "Error occurred at line $line (exit code: $error_code)"
    exit $error_code
}

trap 'handle_error $LINENO $?' ERR

# ═══════════════════════════════════════════════════════════════════
# فحص GPU / Check GPU
# ═══════════════════════════════════════════════════════════════════
check_gpu() {
    log_step "Checking NVIDIA GPU..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. NVIDIA drivers not installed."
        log_info "Install drivers: sudo ubuntu-drivers autoinstall"
        return 1
    fi
    
    # الحصول على معلومات GPU
    local gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)
    
    if [ -z "$gpu_info" ]; then
        log_error "No NVIDIA GPU detected"
        return 1
    fi
    
    log_info "GPU Info: $gpu_info"
    
    # فحص نوع GPU
    local gpu_name=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
    if [[ ! "$gpu_name" =~ RTX.*5090 ]]; then
        log_warn "Expected RTX 5090, found: $gpu_name"
        log_info "Continuing anyway..."
    else
        log_success "RTX 5090 confirmed"
    fi
    
    # فحص إصدار السائق
    local driver_version=$(echo "$gpu_info" | cut -d',' -f2 | xargs | cut -d'.' -f1)
    if [ "$driver_version" -lt "${MIN_DRIVER_VERSION%%.*}" ]; then
        log_warn "Driver version $driver_version may be too old (recommended: >= $MIN_DRIVER_VERSION)"
    else
        log_success "Driver version: $driver_version"
    fi
    
    # فحص ذاكرة GPU
    local gpu_memory=$(echo "$gpu_info" | cut -d',' -f3 | xargs | grep -o '[0-9]*' | head -1)
    if [ "$gpu_memory" -lt "$GPU_MEMORY_GB"000 ]; then
        log_warn "GPU memory: ${gpu_memory}MB (expected >= ${GPU_MEMORY_GB}GB)"
    else
        log_success "GPU memory: ${gpu_memory}MB"
    fi
    
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# تثبيت/تحديث CUDA / Install/Update CUDA
# ═══════════════════════════════════════════════════════════════════
install_cuda() {
    log_step "Checking CUDA installation..."
    
    local cuda_installed=false
    local current_cuda=""
    
    if command -v nvcc &> /dev/null; then
        current_cuda=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        log_info "CUDA found: $current_cuda"
        cuda_installed=true
    fi
    
    if [ "$cuda_installed" = true ]; then
        local current_major=$(echo "$current_cuda" | cut -d'.' -f1)
        local required_major=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
        
        if [ "$current_major" -ge "$required_major" ]; then
            log_success "CUDA version $current_cuda is sufficient"
            return 0
        else
            log_warn "CUDA $current_cuda is too old. Required: >= $CUDA_VERSION"
        fi
    fi
    
    # تثبيت CUDA
    log_info "Installing CUDA $CUDA_VERSION..."
    
    execute_or_dry "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update" "Add CUDA repository"
    
    execute_or_dry "apt-get install -y cuda-toolkit-$CUDA_VERSION nvidia-driver-550" "Install CUDA toolkit"
    
    # إضافة CUDA إلى PATH
    if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
        execute_or_dry "echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc" "Add CUDA to PATH"
        execute_or_dry "echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc" "Add CUDA to LD_LIBRARY_PATH"
    fi
    
    log_success "CUDA $CUDA_VERSION installed"
    log_warn "Please restart the system or run: source ~/.bashrc"
}

# ═══════════════════════════════════════════════════════════════════
# تثبيت مكتبات Python GPU / Install GPU Python Libraries
# ═══════════════════════════════════════════════════════════════════
install_gpu_python_libs() {
    log_step "Installing GPU Python libraries..."
    
    local venv_path="$PROJECT_ROOT/.venv_rtx"
    
    # إنشاء virtual environment للـ GPU
    if [ ! -d "$venv_path" ]; then
        execute_or_dry "python3 -m venv '$venv_path'" "Create GPU virtual environment"
    fi
    
    # تثبيت PyTorch مع CUDA
    execute_or_dry "'$venv_path/bin/pip' install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" "Install PyTorch with CUDA"
    
    # تثبيت TensorFlow
    execute_or_dry "'$venv_path/bin/pip' install tensorflow[and-cuda]" "Install TensorFlow with CUDA"
    
    # تثبيت مكتبات إضافية
    execute_or_dry "'$venv_path/bin/pip' install transformers accelerate bitsandbytes xformers" "Install ML libraries"
    
    log_success "GPU Python libraries installed"
}

# ═══════════════════════════════════════════════════════════════════
# نشر خدمة GPU Trainer / Deploy GPU Trainer Service
# ═══════════════════════════════════════════════════════════════════
deploy_gpu_trainer() {
    log_step "Deploying GPU Trainer service..."
    
    local service_file="/etc/systemd/system/bi-ide-gpu-trainer.service"
    local venv_path="$PROJECT_ROOT/.venv_rtx"
    
    # إنشاء ملف الخدمة
    local service_content="[Unit]
Description=BI-IDE GPU Trainer (RTX 5090)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ExecStart=$venv_path/bin/python $PROJECT_ROOT/services/gpu_trainer.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"
    
    if [ "$DRY_RUN" = false ]; then
        echo "$service_content" | sudo tee "$service_file" > /dev/null
        log_success "Service file created: $service_file"
    else
        log_warn "[DRY RUN] Would create service file: $service_file"
    fi
    
    # إعادة تحميل systemd
    execute_or_dry "systemctl daemon-reload" "Reload systemd"
    
    # تمكين الخدمة
    execute_or_dry "systemctl enable bi-ide-gpu-trainer" "Enable GPU trainer service"
    
    # بدء الخدمة
    execute_or_dry "systemctl start bi-ide-gpu-trainer" "Start GPU trainer service"
    
    log_success "GPU Trainer service deployed"
}

# ═══════════════════════════════════════════════════════════════════
# تكوين إعدادات GPU / Configure GPU Settings
# ═══════════════════════════════════════════════════════════════════
configure_gpu_settings() {
    log_step "Configuring GPU settings..."
    
    # إنشاء ملف تكوين GPU
    local config_dir="$PROJECT_ROOT/config"
    local config_file="$config_dir/gpu_rtx5090.json"
    
    execute_or_dry "mkdir -p '$config_dir'" "Create config directory"
    
    local config_content='{
    "gpu": {
        "model": "RTX 5090",
        "memory_gb": 32,
        "cuda_cores": 21760,
        "tensor_cores": 680
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.0001,
        "mixed_precision": true,
        "gradient_checkpointing": true,
        "optimizer": "adamw_8bit",
        "max_memory_usage": 0.95
    },
    "inference": {
        "batch_size": 128,
        "fp16": true,
        "compile_model": true
    },
    "monitoring": {
        "temperature_threshold": 85,
        "memory_threshold": 0.95,
        "power_limit_watts": 450
    }
}'
    
    if [ "$DRY_RUN" = false ]; then
        echo "$config_content" > "$config_file"
        log_success "GPU config created: $config_file"
    else
        log_warn "[DRY RUN] Would create config: $config_file"
    fi
    
    # تكوين nvidia-persistenced للحفاظ على حالة GPU
    execute_or_dry "systemctl enable nvidia-persistenced" "Enable nvidia-persistenced"
    execute_or_dry "systemctl start nvidia-persistenced" "Start nvidia-persistenced"
    
    log_success "GPU settings configured"
}

# ═══════════════════════════════════════════════════════════════════
# إعداد المراقبة / Setup Monitoring
# ═══════════════════════════════════════════════════════════════════
setup_monitoring() {
    log_step "Setting up GPU monitoring..."
    
    # إنشاء سكربت مراقبة GPU
    local monitor_script="$PROJECT_ROOT/scripts/monitor_gpu.sh"
    
    local script_content='#!/bin/bash
# GPU Monitoring Script for RTX 5090

LOG_FILE="/var/log/bi-ide/gpu_monitor.log"
mkdir -p "$(dirname "$LOG_FILE")"

while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    # جمع البيانات
    GPU_DATA=$(nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits)
    
    # حفظ البيانات
    echo "$TIMESTAMP,$GPU_DATA" >> "$LOG_FILE"
    
    # فحص التنبيهات
    TEMP=$(echo "$GPU_DATA" | cut -d"," -f2 | xargs)
    if [ "$TEMP" -gt 85 ]; then
        logger -t bi-ide-gpu "WARNING: GPU temperature high: ${TEMP}°C"
    fi
    
    sleep 10
done
'
    
    if [ "$DRY_RUN" = false ]; then
        echo "$script_content" > "$monitor_script"
        chmod +x "$monitor_script"
        log_success "GPU monitor script created"
    else
        log_warn "[DRY RUN] Would create monitor script"
    fi
    
    # إنشاء خدمة المراقبة
    local monitor_service="/etc/systemd/system/bi-ide-gpu-monitor.service"
    local service_content="[Unit]
Description=BI-IDE GPU Monitor
After=network.target

[Service]
Type=simple
ExecStart=$monitor_script
Restart=always

[Install]
WantedBy=multi-user.target
"
    
    if [ "$DRY_RUN" = false ]; then
        echo "$service_content" | sudo tee "$monitor_service" > /dev/null
        sudo systemctl daemon-reload
        sudo systemctl enable bi-ide-gpu-monitor
        log_success "GPU monitor service configured"
    else
        log_warn "[DRY RUN] Would configure GPU monitor service"
    fi
    
    log_success "GPU monitoring setup complete"
}

# ═══════════════════════════════════════════════════════════════════
# اختبار CUDA / Test CUDA
# ═══════════════════════════════════════════════════════════════════
test_cuda() {
    log_step "Testing CUDA availability..."
    
    if [ "$DRY_RUN" = true ]; then
        log_warn "[DRY RUN] Skipping CUDA test"
        return 0
    fi
    
    # إنشاء سكربت اختبار Python
    local test_script="/tmp/test_cuda_$$.py"
    
    cat > "$test_script" << 'EOF'
import sys

def test_cuda():
    print("Testing CUDA availability...")
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"\n✓ TensorFlow: {tf.__version__}")
        print(f"  GPUs: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("\n✗ TensorFlow not installed")
    
    # Performance test
    if torch.cuda.is_available():
        print("\nRunning performance test...")
        import time
        
        # Matrix multiplication test
        size = 10000
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  Matrix multiplication ({size}x{size}): {elapsed:.3f}s")
        print(f"  TFLOPS: {(2 * size**3 / elapsed) / 1e12:.2f}")
    
    return True

if __name__ == "__main__":
    success = test_cuda()
    sys.exit(0 if success else 1)
EOF
    
    # تشغيل الاختبار
    local venv_path="$PROJECT_ROOT/.venv_rtx"
    if [ -f "$venv_path/bin/python" ]; then
        if "$venv_path/bin/python" "$test_script"; then
            log_success "CUDA test passed"
        else
            log_error "CUDA test failed"
            rm -f "$test_script"
            return 1
        fi
    else
        log_warn "Virtual environment not found, skipping CUDA test"
    fi
    
    rm -f "$test_script"
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# عرض الاستخدام / Show Usage
# ═══════════════════════════════════════════════════════════════════
show_usage() {
    cat << EOF
${CYAN}BI-IDE v8 - RTX 5090 Deployment Script${NC}
${MAGENTA}======================================${NC}

Usage: $0 [OPTIONS]

Options:
    -d, --dry-run        Run in dry-run mode (no actual changes)
    --skip-cuda          Skip CUDA installation check
    -h, --help           Show this help message

Examples:
    # نشر كامل على RTX 5090
    $0

    # وضع التشغيل الجاف
    $0 --dry-run

    # تخطي تثبيت CUDA
    $0 --skip-cuda

EOF
}

# ═══════════════════════════════════════════════════════════════════
# الدالة الرئيسية / Main Function
# ═══════════════════════════════════════════════════════════════════
main() {
    local skip_cuda=false
    
    # معالجة المعاملات
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-cuda)
                skip_cuda=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # إنشاء دليل السجلات
    mkdir -p "$LOG_DIR"
    touch "$LOG_FILE"
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   BI-IDE v8 - RTX 5090 Deployment${NC}"
    echo -e "${CYAN}   Mode: $([ "$DRY_RUN" = true ] && echo 'DRY-RUN' || echo 'LIVE')${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        log_warn "Running in DRY-RUN mode"
    fi
    
    # فحوصات ما قبل النشر
    check_gpu
    
    if [ "$skip_cuda" = false ]; then
        install_cuda
    fi
    
    # تثبيت المكتبات
    install_gpu_python_libs
    
    # النشر
    configure_gpu_settings
    deploy_gpu_trainer
    setup_monitoring
    
    # اختبار
    test_cuda
    
    # ملخص
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   RTX 5090 DEPLOYMENT COMPLETE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Log file:${NC} $LOG_FILE"
    echo -e "  ${CYAN}GPU:${NC} $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo -e "  ${CYAN}CUDA:${NC} $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'N/A')"
    echo ""
    echo -e "  ${CYAN}Services:${NC}"
    echo -e "    bi-ide-gpu-trainer    - GPU training service"
    echo -e "    bi-ide-gpu-monitor    - GPU monitoring service"
    echo ""
    echo -e "  ${CYAN}Commands:${NC}"
    echo -e "    Check GPU:     nvidia-smi"
    echo -e "    Service logs:  journalctl -u bi-ide-gpu-trainer -f"
    echo -e "    Monitor GPU:   watch -n 1 nvidia-smi"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
}

# تشغيل الدالة الرئيسية
main "$@"
