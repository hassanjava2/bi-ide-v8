#!/bin/bash
# ═══════════════════════════════════════════════════
# BI-IDE Emergency Data Downloader
# ═══════════════════════════════════════════════════
#
# يحمل بيانات تدريب ضخمة للعمل بدون إنترنت
# الاستخدام:
#   bash download_emergency_data.sh /path/to/4tb/drive
#
# ═══════════════════════════════════════════════════

set -e

DATA_ROOT="${1:-/data/emergency}"
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║   BI-IDE Emergency Data Downloader v1.0   ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "📁 Target: $DATA_ROOT"
echo ""

mkdir -p "$DATA_ROOT"

# === وظائف مساعدة ===
download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "   ⏭️  Already exists: $(basename $dest)"
        return 0
    fi
    echo "   📥 Downloading: $(basename $dest)..."
    curl -L --progress-bar "$url" -o "$dest"
}

# ═══════════════════════════════════════════════════
# 1. Wikipedia (عربي + إنجليزي)
# ═══════════════════════════════════════════════════
echo "📚 [1/6] Wikipedia dumps..."
mkdir -p "$DATA_ROOT/wikipedia"

# عربي — أحدث dump
download_if_missing \
    "https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2" \
    "$DATA_ROOT/wikipedia/arwiki-latest.xml.bz2"

# إنجليزي — أحدث dump  
download_if_missing \
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" \
    "$DATA_ROOT/wikipedia/enwiki-latest.xml.bz2"

echo "   ✅ Wikipedia ready"

# ═══════════════════════════════════════════════════
# 2. Stack Overflow
# ═══════════════════════════════════════════════════
echo ""
echo "💬 [2/6] Stack Overflow dump..."
mkdir -p "$DATA_ROOT/stackoverflow"

download_if_missing \
    "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z" \
    "$DATA_ROOT/stackoverflow/posts.7z"

echo "   ✅ Stack Overflow ready"

# ═══════════════════════════════════════════════════
# 3. Linux Kernel
# ═══════════════════════════════════════════════════
echo ""
echo "🐧 [3/6] Linux kernel source..."
mkdir -p "$DATA_ROOT/linux"

if [ ! -d "$DATA_ROOT/linux/linux-src" ]; then
    echo "   📥 Cloning Linux kernel (shallow)..."
    git clone --depth 1 https://github.com/torvalds/linux.git "$DATA_ROOT/linux/linux-src"
else
    echo "   ⏭️  Already exists: linux-src"
fi

echo "   ✅ Linux kernel ready"

# ═══════════════════════════════════════════════════
# 4. AOSP (Android Open Source Project)
# ═══════════════════════════════════════════════════
echo ""
echo "🤖 [4/6] Android AOSP (framework only)..."
mkdir -p "$DATA_ROOT/aosp"

if [ ! -d "$DATA_ROOT/aosp/frameworks-base" ]; then
    echo "   📥 Cloning Android framework..."
    git clone --depth 1 https://github.com/niccolli/aosp-platform-frameworks-base.git "$DATA_ROOT/aosp/frameworks-base"
else
    echo "   ⏭️  Already exists"
fi

echo "   ✅ AOSP framework ready"

# ═══════════════════════════════════════════════════
# 5. Security Data (CVE + Exploits)
# ═══════════════════════════════════════════════════
echo ""
echo "🔒 [5/6] Security databases..."
mkdir -p "$DATA_ROOT/security"

# CVE database
if [ ! -d "$DATA_ROOT/security/cve-list" ]; then
    echo "   📥 Cloning CVE database..."
    git clone --depth 1 https://github.com/CVEProject/cvelistV5.git "$DATA_ROOT/security/cve-list"
else
    echo "   ⏭️  Already exists"
fi

# Exploit database
if [ ! -d "$DATA_ROOT/security/exploitdb" ]; then
    echo "   📥 Cloning Exploit-DB..."
    git clone --depth 1 https://gitlab.com/exploit-database/exploitdb.git "$DATA_ROOT/security/exploitdb"
else
    echo "   ⏭️  Already exists"
fi

echo "   ✅ Security data ready"

# ═══════════════════════════════════════════════════
# 6. تقنية وبرمجة
# ═══════════════════════════════════════════════════
echo ""
echo "📖 [6/6] Programming resources..."
mkdir -p "$DATA_ROOT/code"

# Rust compiler (مفتوح المصدر)
if [ ! -d "$DATA_ROOT/code/rust" ]; then
    git clone --depth 1 https://github.com/rust-lang/rust.git "$DATA_ROOT/code/rust"
fi

# Swift (أبل — مفتوح)
if [ ! -d "$DATA_ROOT/code/swift" ]; then
    git clone --depth 1 https://github.com/apple/swift.git "$DATA_ROOT/code/swift"
fi

# Chromium (جزء صغير — WebKit)
if [ ! -d "$DATA_ROOT/code/webkit" ]; then
    git clone --depth 1 https://github.com/niccolli/WebKit.git "$DATA_ROOT/code/webkit" 2>/dev/null || echo "   ⚠️ WebKit skipped"
fi

echo "   ✅ Code resources ready"

# ═══════════════════════════════════════════════════
# ملخص
# ═══════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║           ✅ Download Complete!            ║"
echo "╠═══════════════════════════════════════════╣"
TOTAL_SIZE=$(du -sh "$DATA_ROOT" 2>/dev/null | awk '{print $1}')
echo "║  Total: $TOTAL_SIZE"
echo "║  Path:  $DATA_ROOT"
echo "╠═══════════════════════════════════════════╣"
echo "║  Contents:                                ║"
echo "║  📚 Wikipedia (AR + EN)                   ║"
echo "║  💬 Stack Overflow                        ║"
echo "║  🐧 Linux kernel                         ║"
echo "║  🤖 Android framework                    ║"
echo "║  🔒 CVE + Exploit-DB                     ║"
echo "║  📖 Rust + Swift source                   ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "الكشافة تكدر تستخدم هاي البيانات تلقائياً."
echo "بس أضف المسار لـ InternalScout:"
echo "  EMERGENCY_DATA=$DATA_ROOT"
echo ""
