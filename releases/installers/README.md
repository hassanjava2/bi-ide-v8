# Installers Directory

ضع هنا ملفات التثبيت النهائية التي تريد إتاحتها عبر صفحة الويب بعد تسجيل الدخول.

## Supported file types
- Windows: `.msi`, `.exe`
- macOS: `.dmg`
- Linux: `.deb`, `.rpm`, `.AppImage`
- Other: `.zip`, `.tar.gz`

## Example names
- `bi-ide-desktop-0.1.0-windows-x64.msi`
- `bi-ide-desktop-0.1.0-macos-arm64.dmg`
- `bi-ide-desktop-0.1.0-linux-amd64.deb`

## API endpoints
- List installers: `GET /api/v1/downloads/installers`
- Download installer: `GET /api/v1/downloads/installers/{installer_id}`

> ملاحظة: هذه الروابط تتطلب تسجيل دخول (Bearer token).
