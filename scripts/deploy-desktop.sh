#!/bin/bash
# BI-IDE Desktop — Build & Deploy Script
# يبني التطبيق لكل الأنظمة ويرفعه للسيرفر
# Usage: ./deploy.sh [macos|linux|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESKTOP_DIR="$PROJECT_ROOT/apps/desktop-tauri"
RELEASES_DIR="$PROJECT_ROOT/releases/installers"
HOSTINGER="root@76.13.154.123"
HOSTINGER_PATH="/opt/bi-iq-app/releases/installers"
RTX5090_USER="bi"
RTX5090_HOST="localhost"
RTX5090_PORT="2222"
VERSION=$(grep '"version"' "$DESKTOP_DIR/src-tauri/tauri.conf.json" | head -1 | sed 's/.*"version": *"\([^"]*\)".*/\1/')

echo "🚀 BI-IDE Desktop Build & Deploy v${VERSION}"
echo "==========================================="

build_macos() {
    echo "🍎 Building macOS (arm64)..."
    cd "$DESKTOP_DIR"
    source "$HOME/.cargo/env" 2>/dev/null || true
    npm run tauri:build
    
    # Create DMG
    BUNDLE_DIR="$PROJECT_ROOT/target/release/bundle/macos"
    if [ -d "$BUNDLE_DIR/BI-IDE Desktop.app" ]; then
        hdiutil create -volname "BI-IDE Desktop" \
            -srcfolder "$BUNDLE_DIR/BI-IDE Desktop.app" \
            -ov -format UDZO \
            "$BUNDLE_DIR/BI-IDE_Desktop_${VERSION}_arm64.dmg"
        echo "✅ macOS DMG created: BI-IDE_Desktop_${VERSION}_arm64.dmg"
    fi
}

build_linux() {
    echo "🐧 Building Linux on RTX 5090..."
    cd "$PROJECT_ROOT"
    
    # Create tarball
    tar czf /tmp/desktop-tauri-src.tar.gz \
        --exclude='node_modules' --exclude='target' --exclude='.git' \
        --exclude='data' --exclude='.venv' \
        apps/desktop-tauri/ libs/protocol/ agents/desktop-agent-rs/ \
        services/sync-service/ Cargo.toml Cargo.lock
    
    # Upload to Hostinger -> RTX 5090
    scp -o StrictHostKeyChecking=no /tmp/desktop-tauri-src.tar.gz "$HOSTINGER:/tmp/"
    ssh -o StrictHostKeyChecking=no "$HOSTINGER" \
        "sshpass -p 353631 scp -o StrictHostKeyChecking=no -P $RTX5090_PORT /tmp/desktop-tauri-src.tar.gz $RTX5090_USER@$RTX5090_HOST:/tmp/"
    
    # Build on RTX 5090
    ssh -o StrictHostKeyChecking=no "$HOSTINGER" \
        "sshpass -p 353631 ssh -o StrictHostKeyChecking=no -p $RTX5090_PORT $RTX5090_USER@$RTX5090_HOST \
        'cd /tmp && rm -rf bi-ide-build && mkdir bi-ide-build && cd bi-ide-build && \
         tar xzf /tmp/desktop-tauri-src.tar.gz 2>/dev/null && \
         cd apps/desktop-tauri && npm install --silent && npx tauri build'"
    
    echo "✅ Linux build complete"
}

upload_to_server() {
    echo "📤 Uploading installers to server..."
    ssh -o StrictHostKeyChecking=no "$HOSTINGER" "mkdir -p $HOSTINGER_PATH"
    
    # Upload macOS DMG
    DMG_PATH="$PROJECT_ROOT/target/release/bundle/macos/BI-IDE_Desktop_${VERSION}_arm64.dmg"
    if [ -f "$DMG_PATH" ]; then
        scp -o StrictHostKeyChecking=no "$DMG_PATH" "$HOSTINGER:$HOSTINGER_PATH/"
        echo "  ✅ macOS DMG uploaded"
    fi
    
    # Fetch Linux builds from RTX 5090
    ssh -o StrictHostKeyChecking=no "$HOSTINGER" \
        "sshpass -p 353631 scp -o StrictHostKeyChecking=no -P $RTX5090_PORT \
        '$RTX5090_USER@$RTX5090_HOST:/tmp/bi-ide-build/target/release/bundle/deb/*.deb' \
        $HOSTINGER_PATH/ 2>/dev/null && echo '  ✅ Linux .deb uploaded' || echo '  ⚠️ No Linux .deb found'"
    
    ssh -o StrictHostKeyChecking=no "$HOSTINGER" \
        "sshpass -p 353631 scp -o StrictHostKeyChecking=no -P $RTX5090_PORT \
        '$RTX5090_USER@$RTX5090_HOST:/tmp/bi-ide-build/target/release/bundle/appimage/*.AppImage' \
        $HOSTINGER_PATH/ 2>/dev/null && echo '  ✅ Linux AppImage uploaded' || echo '  ⚠️ No Linux AppImage found'"
    
    echo ""
    echo "📋 Files on server:"
    ssh -o StrictHostKeyChecking=no "$HOSTINGER" "ls -lh $HOSTINGER_PATH/"
}

case "${1:-all}" in
    macos)
        build_macos
        upload_to_server
        ;;
    linux)
        build_linux
        upload_to_server
        ;;
    all)
        build_macos
        build_linux
        upload_to_server
        ;;
    upload)
        upload_to_server
        ;;
    *)
        echo "Usage: $0 [macos|linux|all|upload]"
        exit 1
        ;;
esac

echo ""
echo "🎉 Done! Check https://bi-iq.com/downloads"
