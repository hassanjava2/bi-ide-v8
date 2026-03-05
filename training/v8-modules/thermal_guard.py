"""
Thermal Guard - الحماية الحرارية

قوانين RTX5090:
- WARNING: 90°C
- THROTTLE: 94°C  
- EMERGENCY STOP: 97°C
- RESUME: 86°C
"""

import os
import time
import threading
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum


class ThermalState(Enum):
    """حالة الحرارة"""
    NORMAL = "normal"
    WARNING = "warning"
    THROTTLE = "throttle"
    EMERGENCY = "emergency"


@dataclass
class ThermalReading:
    """قراءة حرارية"""
    cpu_temp: float
    gpu_temp: Optional[float]
    state: ThermalState
    timestamp: float


class ThermalGuard:
    """
    حارس حراري - يتبع قوانين RTX5090 بدقة
    """
    
    # حدود RTX5090 (ثابتة - لا تتغير)
    WARNING_TEMP = 90.0      # °C
    THROTTLE_TEMP = 94.0     # °C
    EMERGENCY_TEMP = 97.0    # °C - STOP ALL
    RESUME_TEMP = 86.0       # °C
    
    CHECK_INTERVAL = 5.0     # ثواني
    
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_state = ThermalState.NORMAL
        self._cpu_temp = 0.0
        self._gpu_temp: Optional[float] = None
        self._callbacks: list[Callable[[ThermalState, ThermalReading], None]] = []
        self._lock = threading.RLock()
        
        # إحصائيات
        self._emergency_count = 0
        self._throttle_count = 0
        self._warning_count = 0
        
        print(f"🌡️ Thermal Guard initialized")
        print(f"   Warning: {self.WARNING_TEMP}°C")
        print(f"   Throttle: {self.THROTTLE_TEMP}°C")
        print(f"   Emergency: {self.EMERGENCY_TEMP}°C")
    
    def _read_cpu_temp(self) -> float:
        """قراءة حرارة CPU"""
        max_temp = 0.0
        
        try:
            # Linux hwmon
            hwmon_path = Path("/sys/class/hwmon")
            if hwmon_path.exists():
                for hwmon in hwmon_path.glob("hwmon*"):
                    name_file = hwmon / "name"
                    if name_file.exists():
                        name = name_file.read_text().strip()
                        if name in ("coretemp", "k10temp", "zenpower"):
                            for t_file in hwmon.glob("temp*_input"):
                                try:
                                    t = int(t_file.read_text().strip()) / 1000
                                    if 0 < t < 120:  # تصفية القيم الغير منطقية
                                        max_temp = max(max_temp, t)
                                except:
                                    pass
            
            # thermal_zone fallback
            if max_temp == 0:
                thermal_path = Path("/sys/class/thermal")
                if thermal_path.exists():
                    for zone in thermal_path.glob("thermal_zone*"):
                        try:
                            t = int((zone / "temp").read_text().strip()) / 1000
                            if 0 < t < 120:
                                max_temp = max(max_temp, t)
                        except:
                            pass
            
            # macOS
            if max_temp == 0 and os.uname().sysname == "Darwin":
                import subprocess
                result = subprocess.run(
                    ["osx-cpu-temp"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    max_temp = float(result.stdout.strip().replace("°C", ""))
                    
        except Exception as e:
            pass
        
        return max_temp
    
    def _read_gpu_temp(self) -> Optional[float]:
        """قراءة حرارة GPU"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return None
    
    def _determine_state(self, cpu_temp: float, gpu_temp: Optional[float]) -> ThermalState:
        """تحديد الحالة بناءً على الحرارة"""
        max_temp = max(cpu_temp, gpu_temp or 0)
        
        if max_temp >= self.EMERGENCY_TEMP:
            return ThermalState.EMERGENCY
        elif max_temp >= self.THROTTLE_TEMP:
            return ThermalState.THROTTLE
        elif max_temp >= self.WARNING_TEMP:
            return ThermalState.WARNING
        return ThermalState.NORMAL
    
    def _monitoring_loop(self):
        """حلقة المراقبة"""
        print("🌡️ Thermal monitoring started")
        
        while self._running:
            try:
                # قراءة الحرارة
                cpu_temp = self._read_cpu_temp()
                gpu_temp = self._read_gpu_temp()
                
                with self._lock:
                    self._cpu_temp = cpu_temp
                    self._gpu_temp = gpu_temp
                    
                    old_state = self._current_state
                    new_state = self._determine_state(cpu_temp, gpu_temp)
                    
                    # تحديث الإحصائيات
                    if new_state == ThermalState.EMERGENCY:
                        self._emergency_count += 1
                    elif new_state == ThermalState.THROTTLE:
                        self._throttle_count += 1
                    elif new_state == ThermalState.WARNING:
                        self._warning_count += 1
                    
                    self._current_state = new_state
                    
                    # إشعار التغيير
                    if new_state != old_state:
                        reading = ThermalReading(
                            cpu_temp=cpu_temp,
                            gpu_temp=gpu_temp,
                            state=new_state,
                            timestamp=time.time()
                        )
                        self._notify_state_change(new_state, reading)
                
                time.sleep(self.CHECK_INTERVAL)
                
            except Exception as e:
                print(f"⚠️ Thermal monitor error: {e}")
                time.sleep(self.CHECK_INTERVAL)
        
        print("🌡️ Thermal monitoring stopped")
    
    def _notify_state_change(self, state: ThermalState, reading: ThermalReading):
        """إشعار تغيير الحالة"""
        icons = {
            ThermalState.NORMAL: "✅",
            ThermalState.WARNING: "⚠️",
            ThermalState.THROTTLE: "🔥",
            ThermalState.EMERGENCY: "🚨"
        }
        
        msgs = {
            ThermalState.NORMAL: f"CPU: {reading.cpu_temp:.1f}°C - Normal",
            ThermalState.WARNING: f"CPU: {reading.cpu_temp:.1f}°C - Warning",
            ThermalState.THROTTLE: f"CPU: {reading.cpu_temp:.1f}°C - Throttling",
            ThermalState.EMERGENCY: f"CPU: {reading.cpu_temp:.1f}°C - EMERGENCY STOP"
        }
        
        print(f"{icons.get(state, '❓')} {msgs.get(state)}")
        
        # استدعاء المعالجات
        for callback in self._callbacks:
            try:
                callback(state, reading)
            except Exception as e:
                print(f"⚠️ Thermal callback error: {e}")
    
    def start(self):
        """بدء المراقبة"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """إيقاف المراقبة"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def get_state(self) -> ThermalState:
        """الحصول على الحالة الحالية"""
        with self._lock:
            return self._current_state
    
    def get_reading(self) -> ThermalReading:
        """الحصول على آخر قراءة"""
        with self._lock:
            return ThermalReading(
                cpu_temp=self._cpu_temp,
                gpu_temp=self._gpu_temp,
                state=self._current_state,
                timestamp=time.time()
            )
    
    def is_safe_to_train(self) -> bool:
        """هل من الآمن التدريب؟"""
        with self._lock:
            return self._current_state != ThermalState.EMERGENCY
    
    def should_throttle(self) -> bool:
        """هل يجب خنق الأداء؟"""
        with self._lock:
            return self._current_state in (ThermalState.THROTTLE, ThermalState.EMERGENCY)
    
    def get_stats(self) -> dict:
        """إحصائيات الحرارة"""
        with self._lock:
            return {
                "current_cpu_temp": round(self._cpu_temp, 1),
                "current_gpu_temp": round(self._gpu_temp, 1) if self._gpu_temp else None,
                "state": self._current_state.value,
                "emergency_count": self._emergency_count,
                "throttle_count": self._throttle_count,
                "warning_count": self._warning_count,
                "limits": {
                    "warning": self.WARNING_TEMP,
                    "throttle": self.THROTTLE_TEMP,
                    "emergency": self.EMERGENCY_TEMP,
                    "resume": self.RESUME_TEMP,
                }
            }
    
    def on_state_change(self, callback: Callable[[ThermalState, ThermalReading], None]):
        """تسجيل معالج لتغيير الحالة"""
        self._callbacks.append(callback)


# Singleton
_thermal_guard: Optional[ThermalGuard] = None


def get_thermal_guard() -> ThermalGuard:
    """الحصول على الحارس الحراري الموحد"""
    global _thermal_guard
    if _thermal_guard is None:
        _thermal_guard = ThermalGuard()
    return _thermal_guard
