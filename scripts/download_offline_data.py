#!/usr/bin/env python3
"""
سكربت تحميل البيانات للعمل Offline
تنزيل النماذج الجاهزة وتخزينها محلياً

القائمة الإلزامية:
| البيانات | الحجم التقريبي | الأولوية |
|----------|----------------|----------|
| Llama 3.1 70B Q4_K_M | 40GB | 🔴 فوري |
| Mistral 7B Q4_K_M | 4GB | 🔴 فوري |
| Qwen2.5 72B Q4_K_M | 40GB | 🔴 فوري |
| Wikipedia Arabic | 2GB | 🔴 فوري |
| Wikipedia English | 22GB | 🟡 مهم |
| arXiv papers (STEM) | 50GB | 🟡 مهم |
| OpenTextbook Library | 10GB | 🟡 مهم |
| Stack Overflow dump | 15GB | 🟢 مفيد |
| Wikibooks | 1GB | 🟢 مفيد |
| PubMed (طب) | 30GB | 🟢 مفيد |

⚠️ Deadline: أسبوعين من الآن — لا تأخير!
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OfflineDataDownloader:
    """محمل البيانات للعمل Offline"""
    
    def __init__(self, base_dir: str = "offline_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Define downloads
        self.downloads = {
            "critical": {
                "llama_3.1_70b": {
                    "url": "https://huggingface.co/TheBloke/Llama-3.1-70B-GGUF/resolve/main/llama-3.1-70b.Q4_K_M.gguf",
                    "size_gb": 40,
                    "path": self.base_dir / "models" / "llama-3.1-70b.Q4_K_M.gguf"
                },
                "mistral_7b": {
                    "url": "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf",
                    "size_gb": 4,
                    "path": self.base_dir / "models" / "mistral-7b.Q4_K_M.gguf"
                },
                "qwen_72b": {
                    "url": "https://huggingface.co/TheBloke/Qwen2.5-72B-GGUF/resolve/main/qwen2.5-72b.Q4_K_M.gguf",
                    "size_gb": 40,
                    "path": self.base_dir / "models" / "qwen-72b.Q4_K_M.gguf"
                },
                "wikipedia_ar": {
                    "url": "https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2",
                    "size_gb": 2,
                    "path": self.base_dir / "wikipedia" / "arwiki-latest.xml.bz2"
                }
            },
            "important": {
                "wikipedia_en": {
                    "url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
                    "size_gb": 22,
                    "path": self.base_dir / "wikipedia" / "enwiki-latest.xml.bz2"
                },
                "arxiv_stem": {
                    "url": "https://arxiv.org/help/bulk_data_s3",
                    "size_gb": 50,
                    "path": self.base_dir / "arxiv" / "manifest.txt",
                    "note": "Use S3 sync: aws s3 sync s3://arxiv/src arxiv_src/"
                },
                "opentextbook": {
                    "url": "https://open.umn.edu/opentextbooks",
                    "size_gb": 10,
                    "path": self.base_dir / "textbooks",
                    "note": "Manual download from website"
                }
            },
            "useful": {
                "stack_overflow": {
                    "url": "https://archive.org/download/stackexchange/stackexchange_archive.torrent",
                    "size_gb": 15,
                    "path": self.base_dir / "stackexchange",
                    "note": "Use torrent client"
                },
                "wikibooks": {
                    "url": "https://dumps.wikimedia.org/wikibooks/latest/",
                    "size_gb": 1,
                    "path": self.base_dir / "wikibooks"
                },
                "pubmed": {
                    "url": "https://ftp.ncbi.nlm.nih.gov/pubmed/",
                    "size_gb": 30,
                    "path": self.base_dir / "pubmed"
                }
            }
        }
    
    def check_space(self, required_gb: float) -> bool:
        """التحقق من المساحة المتوفرة"""
        stat = os.statvfs(self.base_dir)
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        if available_gb < required_gb:
            logger.error(f"❌ Insufficient space: {available_gb:.1f}GB available, {required_gb:.1f}GB required")
            return False
        
        logger.info(f"✅ Space check passed: {available_gb:.1f}GB available")
        return True
    
    def download_with_progress(self, url: str, destination: Path, chunk_size: int = 8192):
        """تحميل مع عرض التقدم"""
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"⬇️  Downloading: {url}")
        logger.info(f"💾 Destination: {destination}")
        
        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(destination, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r📊 Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
                
                print()  # New line after progress
                logger.info(f"✅ Downloaded: {destination}")
                return True
        
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def download_huggingface(self, repo_id: str, filename: str, destination: Path):
        """تحميل من Hugging Face"""
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info(f"⬇️  Downloading from Hugging Face: {repo_id}/{filename}")
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=destination.parent,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"✅ Downloaded: {downloaded_path}")
            return True
        
        except ImportError:
            logger.error("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"❌ HuggingFace download failed: {e}")
            return False
    
    def download_critical(self):
        """تحميل البيانات الحرجة"""
        logger.info("=" * 60)
        logger.info("🔴 DOWNLOADING CRITICAL DATA")
        logger.info("=" * 60)
        
        total_size = sum(d["size_gb"] for d in self.downloads["critical"].values())
        
        if not self.check_space(total_size + 10):  # 10GB buffer
            return False
        
        success_count = 0
        
        for name, info in self.downloads["critical"].items():
            if info["path"].exists():
                logger.info(f"⏭️  {name} already exists, skipping")
                success_count += 1
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📦 Downloading: {name} ({info['size_gb']}GB)")
            logger.info(f"{'='*60}")
            
            # Use huggingface_hub for HuggingFace URLs
            if "huggingface.co" in info["url"]:
                # Parse repo_id and filename from URL
                parts = info["url"].split("/resolve/main/")
                if len(parts) == 2:
                    repo_path = parts[0].split("huggingface.co/")[-1]
                    filename = parts[1]
                    if self.download_huggingface(repo_path, filename, info["path"]):
                        success_count += 1
            else:
                if self.download_with_progress(info["url"], info["path"]):
                    success_count += 1
        
        logger.info(f"\n✅ Critical downloads: {success_count}/{len(self.downloads['critical'])}")
        return success_count == len(self.downloads["critical"])
    
    def download_important(self):
        """تحميل البيانات المهمة"""
        logger.info("\n" + "=" * 60)
        logger.info("🟡 DOWNLOADING IMPORTANT DATA")
        logger.info("=" * 60)
        
        for name, info in self.downloads["important"].items():
            logger.info(f"\n📦 {name} ({info['size_gb']}GB)")
            if "note" in info:
                logger.info(f"📝 Note: {info['note']}")
            logger.info(f"🔗 URL: {info['url']}")
    
    def download_useful(self):
        """تحميل البيانات المفيدة"""
        logger.info("\n" + "=" * 60)
        logger.info("🟢 DOWNLOADING USEFUL DATA")
        logger.info("=" * 60)
        
        for name, info in self.downloads["useful"].items():
            logger.info(f"\n📦 {name} ({info['size_gb']}GB)")
            if "note" in info:
                logger.info(f"📝 Note: {info['note']}")
            logger.info(f"🔗 URL: {info['url']}")
    
    def create_verification_script(self):
        """إنشاء سكربت للتحقق"""
        script_path = self.base_dir / "verify_offline_data.py"
        
        script_content = '''#!/usr/bin/env python3
"""
التحقق من اكتمال البيانات Offline
"""

from pathlib import Path
import sys

def verify_offline_data(base_dir: str = "offline_data"):
    base = Path(base_dir)
    
    critical_files = {
        "Llama 3.1 70B": base / "models" / "llama-3.1-70b.Q4_K_M.gguf",
        "Mistral 7B": base / "models" / "mistral-7b.Q4_K_M.gguf",
        "Qwen 72B": base / "models" / "qwen-72b.Q4_K_M.gguf",
        "Wikipedia AR": base / "wikipedia" / "arwiki-latest.xml.bz2"
    }
    
    print("🔍 Verifying offline data...")
    print("=" * 60)
    
    all_present = True
    for name, path in critical_files.items():
        status = "✅" if path.exists() else "❌"
        size = f"({path.stat().st_size / (1024**3):.1f} GB)" if path.exists() else "(missing)"
        print(f"{status} {name}: {size}")
        if not path.exists():
            all_present = False
    
    print("=" * 60)
    if all_present:
        print("✅ All critical data present!")
        return 0
    else:
        print("❌ Some critical data missing!")
        return 1

if __name__ == "__main__":
    sys.exit(verify_offline_data())
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        logger.info(f"✅ Created verification script: {script_path}")
    
    def generate_report(self):
        """تقرير شامل"""
        report = {
            "timestamp": str(datetime.now()),
            "base_directory": str(self.base_dir.absolute()),
            "downloads": {}
        }
        
        for category, items in self.downloads.items():
            report["downloads"][category] = {}
            for name, info in items.items():
                path = info["path"]
                report["downloads"][category][name] = {
                    "size_gb": info["size_gb"],
                    "exists": path.exists(),
                    "actual_size_gb": path.stat().st_size / (1024**3) if path.exists() else 0
                }
        
        report_path = self.base_dir / "download_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Report saved: {report_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="Download offline data for BI-IDE")
    parser.add_argument("--base-dir", default="offline_data", help="Base directory for downloads")
    parser.add_argument("--critical-only", action="store_true", help="Download only critical data")
    parser.add_argument("--verify", action="store_true", help="Verify existing data")
    
    args = parser.parse_args()
    
    downloader = OfflineDataDownloader(args.base_dir)
    
    if args.verify:
        # Run verification
        verify_script = Path(args.base_dir) / "verify_offline_data.py"
        if verify_script.exists():
            subprocess.run([sys.executable, str(verify_script)])
        else:
            logger.error("Verification script not found. Run download first.")
        return
    
    # Download
    logger.info("🚀 BI-IDE Offline Data Downloader")
    logger.info("⚠️  This will download large files. Ensure you have sufficient space and bandwidth.")
    
    # Download critical first
    downloader.download_critical()
    
    if not args.critical_only:
        downloader.download_important()
        downloader.download_useful()
    
    # Create verification script
    downloader.create_verification_script()
    
    # Generate report
    downloader.generate_report()
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Critical downloads attempted")
    logger.info(f"📁 Data location: {downloader.base_dir.absolute()}")
    logger.info(f"🔍 Run verification: python {downloader.base_dir}/verify_offline_data.py")
    logger.info("\n⚠️  IMPORTANT: Copy this data to the RTX 5090 machine!")
    logger.info("   rsync -av --progress offline_data/ bi@192.168.1.164:/home/bi/offline_data/")


if __name__ == "__main__":
    from datetime import datetime
    main()
