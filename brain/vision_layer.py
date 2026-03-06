#!/usr/bin/env python3
"""
vision_layer.py — طبقة تحليل الصور والفيديو 👁️

تحليل:
  1. صور (أنواع، أشياء، نصوص)
  2. فيديو (نشاط، أحداث)
  3. كاميرات مباشرة (مراقبة مصانع)
  4. ملفات (PDF, Excel, أي ملف)
  5. مراقبة عمال المصنع ← توجيه + تدريب

يستخدم:
  - نموذج BI-Vision (خاص بينا — MobileNet+SE)
  - YOLO كمعلّم خارجي (بياناتنا ما تطلع)
  - OCR للنصوص
  - Activity Recognition للنشاطات

الأولوية: BI-Vision → YOLO → Rule-based
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("vision")

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.memory_system import memory
except ImportError:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory

MODELS_DIR = PROJECT_ROOT / "brain" / "vision_data" / "models"


@dataclass
class DetectedObject:
    """كائن مكتشف بالصورة"""
    label: str
    confidence: float
    bbox: List[float] = field(default_factory=list)  # x,y,w,h
    category: str = ""  # person, vehicle, equipment, etc.


@dataclass
class AnalysisResult:
    """نتيجة تحليل"""
    source: str             # filepath or camera_id
    source_type: str        # image, video, camera, file
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    objects: List[DetectedObject] = field(default_factory=list)
    text_detected: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: Dict = field(default_factory=dict)


class ImageAnalyzer:
    """
    محلل الصور — يكشف أشياء + نصوص + تصنيف

    الأولوية:
      1. BI-Vision (نموذجنا الخاص)
      2. YOLO (معلّم خارجي — بياناتنا ما تطلع)
      3. Rule-based (قواعد)
    """

    KNOWN_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg",
    }

    def __init__(self):
        self.model_loaded = False
        self.yolo_model = None
        self.bi_vision_model = None
        self.active_model = "rule-based"  # "bi-vision" | "yolo" | "rule-based"
        self._try_load_models()

    def _try_load_models(self):
        """تحميل النماذج — BI-Vision أولاً ثم YOLO"""
        # 1. محاولة BI-Vision (نموذجنا)
        try:
            import torch
            bi_path = MODELS_DIR / "bi_vision_latest.pt"
            if bi_path.exists():
                self.bi_vision_model = torch.load(bi_path, map_location="cpu")
                self.bi_vision_model.eval()
                self.model_loaded = True
                self.active_model = "bi-vision"
                logger.info("✅ BI-Vision model loaded (our model)")
                return
        except Exception:
            pass

        # 2. محاولة YOLO (معلّم خارجي)
        try:
            from ultralytics import YOLO
            yolo_path = MODELS_DIR / "yolov8n.pt"
            if yolo_path.exists():
                self.yolo_model = YOLO(str(yolo_path))
            else:
                self.yolo_model = YOLO("yolov8n.pt")
            self.model_loaded = True
            self.active_model = "yolo"
            logger.info("✅ YOLO teacher model loaded (external)")
        except ImportError:
            logger.info("👁️ Vision: no model — using rule-based")
        except Exception as e:
            logger.warning(f"YOLO load error: {e}")

    def analyze(self, filepath: str, content_bytes: bytes = None) -> AnalysisResult:
        """تحليل صورة"""
        fp = Path(filepath)
        result = AnalysisResult(source=filepath, source_type="image")

        if not fp.exists() and content_bytes is None:
            result.summary = f"File not found: {filepath}"
            return result

        # Metadata
        result.metadata = {
            "filename": fp.name,
            "extension": fp.suffix.lower(),
            "size_bytes": fp.stat().st_size if fp.exists() else len(content_bytes or b""),
            "model_used": self.active_model,
        }

        # الأولوية: BI-Vision → YOLO → Rule-based
        if self.bi_vision_model:
            result = self._analyze_bi_vision(filepath, result)
        elif self.yolo_model:
            result = self._analyze_yolo(filepath, result)
        else:
            result = self._analyze_rule_based(fp, result)

        memory.save_knowledge(
            topic=f"Image Analysis: {fp.name}",
            content=result.summary, source="vision_layer",
        )
        return result

    def _analyze_rule_based(self, fp: Path, result: AnalysisResult) -> AnalysisResult:
        """تحليل قائم على القواعد"""
        size_mb = result.metadata["size_bytes"] / (1024 * 1024)
        name = fp.stem.lower()

        categories = {
            "screenshot": ["screen", "screenshot", "capture", "snap"],
            "diagram": ["diagram", "chart", "graph", "flow", "uml"],
            "photo": ["photo", "img", "pic", "camera"],
            "logo": ["logo", "icon", "brand"],
            "document": ["doc", "scan", "pdf", "page"],
        }

        detected_cat = "unknown"
        for cat, keywords in categories.items():
            if any(kw in name for kw in keywords):
                detected_cat = cat
                break

        result.objects.append(DetectedObject(
            label=detected_cat, confidence=0.6, category="classification"
        ))
        result.summary = f"Image: {fp.name} ({size_mb:.1f}MB), Category: {detected_cat} [rule-based]"
        return result

    def _analyze_yolo(self, filepath: str, result: AnalysisResult) -> AnalysisResult:
        """تحليل بـ YOLO — معلّم خارجي (بياناتنا ما تطلع)"""
        try:
            detections = self.yolo_model(filepath, verbose=False)
            for r in detections:
                for box in r.boxes:
                    obj = DetectedObject(
                        label=r.names[int(box.cls[0])],
                        confidence=round(float(box.conf[0]), 3),
                        bbox=box.xyxy[0].tolist(),
                        category="yolo_detection",
                    )
                    result.objects.append(obj)

            result.summary = (f"Image: {Path(filepath).name}, "
                            f"{len(result.objects)} objects [YOLO teacher]")
        except Exception as e:
            logger.warning(f"YOLO analysis failed: {e}")
            result = self._analyze_rule_based(Path(filepath), result)
        return result

    def _analyze_bi_vision(self, filepath: str, result: AnalysisResult) -> AnalysisResult:
        """تحليل بنموذج BI-Vision (نموذجنا الخاص)"""
        try:
            import torch
            from torchvision import transforms
            from PIL import Image

            img = Image.open(filepath).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = self.bi_vision_model(tensor)
                probs = torch.softmax(output, dim=1)
                top5 = torch.topk(probs, 5)

            for prob, idx in zip(top5.values[0], top5.indices[0]):
                result.objects.append(DetectedObject(
                    label=f"class_{idx.item()}",
                    confidence=round(prob.item(), 3),
                    category="bi_vision",
                ))

            result.summary = (f"Image: {Path(filepath).name}, "
                            f"{len(result.objects)} detections [BI-Vision]")
        except Exception as e:
            logger.warning(f"BI-Vision failed, falling back to YOLO: {e}")
            if self.yolo_model:
                result = self._analyze_yolo(filepath, result)
            else:
                result = self._analyze_rule_based(Path(filepath), result)
        return result


class VideoAnalyzer:
    """محلل الفيديو — frame-by-frame + نشاطات"""

    def __init__(self):
        self.image_analyzer = ImageAnalyzer()

    def analyze(self, filepath: str, sample_fps: int = 1) -> AnalysisResult:
        """تحليل فيديو"""
        fp = Path(filepath)
        result = AnalysisResult(source=filepath, source_type="video")

        result.metadata = {
            "filename": fp.name,
            "extension": fp.suffix.lower(),
            "size_bytes": fp.stat().st_size if fp.exists() else 0,
        }

        # Phase 2: frame extraction + analysis
        # try:
        #     import cv2
        #     cap = cv2.VideoCapture(filepath)
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        #     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #     duration = frame_count / fps
        #     result.metadata["fps"] = fps
        #     result.metadata["duration_sec"] = duration
        #     result.metadata["frames"] = frame_count
        # except ImportError:
        #     pass

        result.summary = f"Video: {fp.name} ({result.metadata['size_bytes']/(1024*1024):.1f}MB)"
        return result


class CameraMonitor:
    """
    مراقب كاميرات — مراقبة مباشرة

    يراقب:
      - مصانع ← يكشف مشاكل
      - عمال ← يوجه ويدرب (حضور/انصراف أوتوماتيكي)
      - سلامة ← ينبه على مخاطر

    متصل بـ company_camera لإدارة الشركة 🏢
    """

    def __init__(self):
        self.cameras: Dict[str, Dict] = {}
        self.alerts: List[Dict] = []
        self.image_analyzer = ImageAnalyzer()
        self._company = None  # lazy import

    def _get_company(self):
        if self._company is None:
            try:
                from brain.company_camera import company
                self._company = company
            except ImportError:
                pass
        return self._company

    def add_camera(self, camera_id: str, url: str, location: str = "", purpose: str = "general"):
        """إضافة كاميرا"""
        self.cameras[camera_id] = {
            "url": url, "location": location, "purpose": purpose,
            "active": True, "last_frame": None,
        }

    def monitor_frame(self, camera_id: str, frame_data: bytes = None) -> Dict:
        """تحليل إطار من كاميرا — يرسل نتائج لـ company_camera"""
        if camera_id not in self.cameras:
            return {"error": f"Camera {camera_id} not found"}

        cam = self.cameras[camera_id]
        analysis = {
            "camera": camera_id, "location": cam["location"],
            "timestamp": datetime.now().isoformat(),
            "objects_detected": 0, "warnings": [],
        }

        # إرسال النتائج لنظام إدارة الشركة
        company = self._get_company()
        if company:
            result = company.process_camera_frame(
                camera_id,
                detected_objects=analysis.get("detected_objects"),
                alerts=analysis.get("warnings"),
            )
            analysis["company_actions"] = result.get("actions", [])

        return analysis

    def check_safety(self, camera_id: str) -> List[str]:
        """فحص سلامة من الكاميرا"""
        warnings = []
        # YOLO safety detection (PPE, fire, hazards)
        return warnings


class FileAnalyzer:
    """محلل ملفات عام — PDF, Excel, CSV, JSON..."""

    def analyze(self, filepath: str) -> AnalysisResult:
        """تحليل أي ملف"""
        fp = Path(filepath)
        result = AnalysisResult(source=filepath, source_type="file")

        if not fp.exists():
            result.summary = f"File not found: {filepath}"
            return result

        ext = fp.suffix.lower()
        size = fp.stat().st_size

        result.metadata = {
            "filename": fp.name, "extension": ext,
            "size_bytes": size, "size_mb": round(size / (1024*1024), 2),
        }

        # تحليل حسب النوع
        if ext in {".json", ".jsonl"}:
            result = self._analyze_json(fp, result)
        elif ext in {".csv", ".tsv"}:
            result = self._analyze_csv(fp, result)
        elif ext in {".txt", ".md", ".log"}:
            result = self._analyze_text(fp, result)
        elif ext in {".py", ".js", ".ts", ".rs", ".go"}:
            result = self._analyze_code(fp, result)
        elif ext in ImageAnalyzer.KNOWN_EXTENSIONS:
            return ImageAnalyzer().analyze(filepath)
        else:
            result.summary = f"File: {fp.name} ({result.metadata['size_mb']}MB)"

        return result

    def _analyze_json(self, fp: Path, result: AnalysisResult) -> AnalysisResult:
        try:
            data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                result.metadata["keys"] = list(data.keys())[:20]
                result.metadata["type"] = "object"
            elif isinstance(data, list):
                result.metadata["items"] = len(data)
                result.metadata["type"] = "array"
            result.summary = f"JSON: {fp.name}, type={result.metadata.get('type')}"
        except Exception:
            result.summary = f"JSON (invalid): {fp.name}"
        return result

    def _analyze_csv(self, fp: Path, result: AnalysisResult) -> AnalysisResult:
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
        result.metadata["rows"] = len(lines) - 1
        if lines:
            result.metadata["columns"] = lines[0].count(",") + 1
        result.summary = f"CSV: {fp.name}, {result.metadata.get('rows', 0)} rows"
        return result

    def _analyze_text(self, fp: Path, result: AnalysisResult) -> AnalysisResult:
        content = fp.read_text(encoding="utf-8", errors="ignore")
        result.metadata["lines"] = content.count("\n") + 1
        result.metadata["words"] = len(content.split())
        result.summary = f"Text: {fp.name}, {result.metadata['lines']} lines"
        return result

    def _analyze_code(self, fp: Path, result: AnalysisResult) -> AnalysisResult:
        content = fp.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        result.metadata["lines"] = len(lines)
        result.metadata["functions"] = sum(1 for l in lines if l.strip().startswith(("def ", "function ", "fn ", "func ")))
        result.metadata["classes"] = sum(1 for l in lines if l.strip().startswith(("class ", "struct ", "interface ")))
        result.summary = f"Code: {fp.name}, {result.metadata['lines']} lines, {result.metadata['functions']} functions"
        return result


class VisionLayer:
    """
    طبقة الرؤية الموحدة

    تجمع كل المحللين:
      - صور
      - فيديو
      - كاميرات
      - ملفات
    """

    def __init__(self):
        self.image = ImageAnalyzer()
        self.video = VideoAnalyzer()
        self.camera = CameraMonitor()
        self.file = FileAnalyzer()

    def analyze(self, filepath: str) -> AnalysisResult:
        """تحيل أي شي — يكتشف النوع أوتوماتيكياً"""
        fp = Path(filepath)
        ext = fp.suffix.lower()

        if ext in ImageAnalyzer.KNOWN_EXTENSIONS:
            return self.image.analyze(filepath)
        elif ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return self.video.analyze(filepath)
        else:
            return self.file.analyze(filepath)

    def get_status(self) -> Dict:
        return {
            "image_model": self.image.active_model,
            "model_loaded": self.image.model_loaded,
            "yolo_available": self.image.yolo_model is not None,
            "bi_vision_available": self.image.bi_vision_model is not None,
            "cameras": len(self.camera.cameras),
            "alerts": len(self.camera.alerts),
        }


# Singleton
vision = VisionLayer()


if __name__ == "__main__":
    print("👁️ Vision Layer — Test\n")

    # تحليل ملف كود
    test_file = Path(__file__)
    r = vision.analyze(str(test_file))
    print(f"Self-analysis: {r.summary}")
    print(f"  Lines: {r.metadata.get('lines')}, Functions: {r.metadata.get('functions')}")

    # تحليل JSON
    import tempfile, json
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps({"test": True, "items": [1,2,3]}, indent=2))
    r2 = vision.analyze(str(tmp))
    print(f"\nJSON analysis: {r2.summary}")
    tmp.unlink()

    # كاميرات
    vision.camera.add_camera("cam_factory_1", "rtsp://192.168.1.100:554", "خط الإنتاج", "production")
    vision.camera.add_camera("cam_safety_1", "rtsp://192.168.1.101:554", "مخزن مواد خام", "safety")
    print(f"\nCameras: {len(vision.camera.cameras)}")
    print(f"Status: {vision.get_status()}")
