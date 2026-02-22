"""
🌐 REAL API LEARNING SYSTEM - نظام التعلم الحقيقي من APIs
كل بيانات حقيقية - لا محاكاة
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import threading
import time
import json
import requests
import os
from datetime import datetime
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RealAPIDataFetcher:
    """جالب بيانات حقيقية من APIs مفتوحة"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BI-IDE-Training-System/1.0'
        })
        self.cache = {}
        self.last_fetch = {}
    
    def fetch_github_trending(self):
        """جلب مستودعات GitHub الشائعة - حقيقي 100%"""
        try:
            # GitHub Search API - مجاني
            url = "https://api.github.com/search/repositories"
            params = {
                "q": "language:python stars:>10000",
                "sort": "stars",
                "order": "desc",
                "per_page": 5
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                repos = [item["full_name"] for item in data.get("items", [])]
                return {"type": "code", "source": "GitHub API", "repos": repos, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"    ⚠️ GitHub API Error: {e}")
        return None
    
    def fetch_crypto_prices(self):
        """جلب أسعار العملات الرقمية - حقيقي من CoinGecko"""
        try:
            # CoinGecko API - مجاني
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": "bitcoin,ethereum,cardano,solana",
                "vs_currencies": "usd",
                "include_24hr_change": "true"
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                markets = {}
                for coin, info in data.items():
                    markets[coin.upper()] = {
                        "price": info["usd"],
                        "change_24h": info.get("usd_24h_change", 0)
                    }
                return {"type": "markets", "source": "CoinGecko API", "markets": markets, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"    ⚠️ CoinGecko API Error: {e}")
        return None
    
    def fetch_hackernews(self):
        """جلب أخبار تقنية من HackerNews - حقيقي"""
        try:
            # HackerNews API - مجاني
            top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = self.session.get(top_stories_url, timeout=10)
            if response.status_code == 200:
                story_ids = response.json()[:3]
                stories = []
                for story_id in story_ids:
                    story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    story_resp = self.session.get(story_url, timeout=10)
                    if story_resp.status_code == 200:
                        story = story_resp.json()
                        stories.append(story.get("title", "No title"))
                return {"type": "tech", "source": "HackerNews API", "stories": stories, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"    ⚠️ HackerNews API Error: {e}")
        return None
    
    def fetch_reddit_programming(self):
        """جلب نقاشات برمجية من Reddit - حقيقي"""
        try:
            # Reddit JSON API - عام
            url = "https://www.reddit.com/r/programming/top.json"
            params = {"limit": 3, "t": "day"}
            headers = {"User-Agent": "BI-IDE-Training/1.0"}
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                posts = [post["data"]["title"] for post in data["data"]["children"]]
                return {"type": "discussion", "source": "Reddit API", "posts": posts, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"    ⚠️ Reddit API Error: {e}")
        return None
    
    def fetch_stackoverflow_questions(self):
        """جلب أسئلة StackOverflow - حقيقي"""
        try:
            # StackExchange API - مجاني
            url = "https://api.stackexchange.com/2.3/questions"
            params = {
                "order": "desc",
                "sort": "votes",
                "tagged": "python;pytorch",
                "site": "stackoverflow",
                "pagesize": 3
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                questions = [item["title"] for item in data.get("items", [])]
                return {"type": "qa", "source": "StackOverflow API", "questions": questions, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"    ⚠️ StackOverflow API Error: {e}")
        return None
    
    def fetch_cve_security(self):
        """جلب ثغرات أمنية من CVE - حقيقي"""
        try:
            # CVE API - عام
            url = "https://cve.circl.lu/api/last"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                cves = []
                for item in data[:3]:
                    cves.append(f"{item['id']}: {item.get('summary', 'No summary')[:100]}")
                return {"type": "security", "source": "CVE API", "cves": cves, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"    ⚠️ CVE API Error: {e}")
        return None
    
    def fetch_arxiv_papers(self):
        """جلب أبحاث من arXiv - حقيقي"""
        try:
            # arXiv API - مجاني
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": "cat:cs.AI",
                "start": 0,
                "max_results": 3,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                papers = []
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                for entry in root.findall('atom:entry', ns):
                    title = entry.find('atom:title', ns)
                    if title is not None:
                        papers.append(title.text.strip().replace('\n', ' '))
                return {"type": "research", "source": "arXiv API", "papers": papers, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"    ⚠️ arXiv API Error: {e}")
        return None
    
    def fetch_for_layer(self, layer_name):
        """جلب بيانات حقيقية حسب اختصاص الطبقة"""
        # Rate limiting - لا نجلب كل مرة
        current_time = time.time()
        if layer_name in self.last_fetch:
            if current_time - self.last_fetch[layer_name] < 60:  # دقيقة واحدة
                return self.cache.get(layer_name)
        
        result = None
        source = "Unknown"
        
        if "code" in layer_name or "builder" in layer_name or "ide" in layer_name:
            result = self.fetch_github_trending() or self.fetch_stackoverflow_questions()
            source = "GitHub/StackOverflow"
        
        elif "erp" in layer_name or "business" in layer_name or "council" in layer_name or "president" in layer_name:
            result = self.fetch_crypto_prices() or self.fetch_hackernews()
            source = "CoinGecko/HackerNews"
        
        elif "guardian" in layer_name or "security" in layer_name:
            result = self.fetch_cve_security()
            source = "CVE Database"
        
        elif "scouts" in layer_name or "research" in layer_name or "experts" in layer_name or "learning" in layer_name:
            result = self.fetch_arxiv_papers()
            source = "arXiv"
        
        elif "meta" in layer_name or "execution" in layer_name:
            result = self.fetch_reddit_programming() or self.fetch_hackernews()
            source = "Reddit/HackerNews"
        
        else:
            result = self.fetch_hackernews()
            source = "HackerNews"
        
        # تخزين في cache
        if result:
            self.cache[layer_name] = result
            self.last_fetch[layer_name] = current_time
            print(f"    ✅ Fetched REAL data from {source}")
        
        return result

class RealDataset(Dataset):
    """Dataset يستخدم بيانات حقيقية من APIs"""
    
    def __init__(self, layer_name, fetcher):
        self.layer_name = layer_name
        self.fetcher = fetcher
        self.seq_length = 128
        self.data = []
        self.vocab = {}
        self.fetch_count = 0
        self.refresh()
    
    def tokenize(self, text):
        """تحويل نص لـ tokens"""
        tokens = []
        for word in str(text).lower().split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) % 10000
            tokens.append(self.vocab[word])
        # padding
        tokens += [0] * (self.seq_length - len(tokens))
        return tokens[:self.seq_length]
    
    def refresh(self):
        """جلب بيانات حقيقية جديدة - وتحرير الذاكرة بعد الاستخدام"""
        print(f"🌐 [{self.layer_name}] Fetching REAL data from Internet APIs...")
        
        data = self.fetcher.fetch_for_layer(self.layer_name)
        samples = []
        
        if data:
            self.fetch_count += 1
            
            if data.get("type") == "code":
                for repo in data.get("repos", [])[:2]:
                    text = f"Repository {repo} is trending with high quality code for machine learning"
                    samples.append(self.tokenize(text))
                for qa in data.get("questions", []):
                    samples.append(self.tokenize(qa))
            
            elif data.get("type") == "markets":
                for asset, info in data.get("markets", {}).items():
                    text = f"Asset {asset} trading at ${info['price']:.2f} with {info['change_24h']:.2f}% change"
                    samples.append(self.tokenize(text))
            
            elif data.get("type") == "tech" or data.get("type") == "discussion":
                for story in data.get("stories", []) or data.get("posts", []):
                    samples.append(self.tokenize(story))
            
            elif data.get("type") == "security":
                for cve in data.get("cves", []):
                    samples.append(self.tokenize(cve))
            
            elif data.get("type") == "research":
                for paper in data.get("papers", []):
                    samples.append(self.tokenize(paper))
            
            elif data.get("type") == "qa":
                for q in data.get("questions", []):
                    samples.append(self.tokenize(q))
        
        # إذا فشلت كل APIs
        if not samples:
            print(f"    ⚠️ All APIs failed, using fallback")
            samples.append(self.tokenize(f"Training data for {self.layer_name}"))
        
        # تكرار للوصول لحجم كافي
        while len(samples) < 50:
            samples.extend(samples[:5])
        
        self.data = samples[:50]
        print(f"    ✅ Got {len(self.data)} samples (Fetch #{self.fetch_count})")
        
        # 🧹 تنظيف: حدد حجم الـ vocab (احتفظ بأحدث 5000 كلمة فقط)
        if len(self.vocab) > 5000:
            # احتفظ بالكلمات الأكثر استخداماً (أو أحدثها)
            self.vocab = dict(list(self.vocab.items())[-5000:])
            print(f"    🧹 Cleaned vocab: {len(self.vocab)} words")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx][:-1])
        y = torch.LongTensor(self.data[idx][1:])
        return x, y

class SmartTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 256
        self.embedding = nn.Embedding(10000, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=4)
        self.fc = nn.Linear(d_model, 10000)
        self.d_model = d_model
    
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        x = self.transformer(x)
        return self.fc(x)

class RealAPITrainer:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        print(f"\n🎯 [{name}]")
        print(f"    Purpose: {desc}")
        
        self.fetcher = RealAPIDataFetcher()
        self.model = SmartTransformer().to(DEVICE)
        self.dataset = RealDataset(name, self.fetcher)
        self.loader = DataLoader(self.dataset, batch_size=16, shuffle=True, num_workers=0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95)
        self.criterion = nn.CrossEntropyLoss()
        
        self.epoch = 0
        self.training = False
        self.stats = {
            "loss": 0.0,
            "accuracy": 0.0,
            "samples": 0,
            "api_fetches": 0
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path("learning_data/checkpoints") / name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoint if available
        self._load_checkpoint()
    
    def train_epoch(self):
        """تدريب epoch واحد"""
        # تحديث البيانات كل 5 epochs
        if self.epoch % 5 == 0:
            self.dataset.refresh()
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in self.loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.view(-1, 10000), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, pred = output.max(2)
            correct += pred.eq(target).sum().item()
            total += target.numel()
            self.stats["samples"] += data.size(0)
        
        self.stats["loss"] = total_loss / len(self.loader)
        self.stats["accuracy"] = 100.0 * correct / total
        self.stats["api_fetches"] = self.dataset.fetch_count
        self.epoch += 1
        self.scheduler.step()
        
        if self.epoch % 10 == 0:
            print(f"  📊 [{self.name}] Epoch {self.epoch:3d} | Loss: {self.stats['loss']:.4f} | Acc: {self.stats['accuracy']:.1f}% | Fetches: {self.stats['api_fetches']}")
        
        # Save checkpoint every 50 epochs
        if self.epoch % 50 == 0 and self.epoch > 0:
            self._save_checkpoint()
    
    def training_loop(self):
        print(f"🔥 [{self.name}] Training with REAL API data...")
        while self.training:
            self.train_epoch()
            time.sleep(0.1)
    
    def start(self):
        if not self.training:
            self.training = True
            threading.Thread(target=self.training_loop, daemon=True).start()
    
    def stop(self):
        self.training = False
        self._save_checkpoint()
    
    def _save_checkpoint(self):
        """حفظ حالة التدريب - مع rotation"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'stats': {
                'loss': self.stats['loss'],
                'accuracy': self.stats['accuracy'],
                'samples': self.stats['samples'],
                'api_fetches': self.stats['api_fetches']
            },
            'vocab_size': len(self.dataset.vocab),
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. حفظ latest.pt (دائماً نحتفظ بي)
        path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, path)
        print(f"    💾 Saved checkpoint: {path} (Epoch {self.epoch})")
        
        # 2. حفظ best model (إذا دقة جيدة)
        if self.stats['accuracy'] > 90:
            best_path = self.checkpoint_dir / f"best_acc{self.stats['accuracy']:.1f}_ep{self.epoch}.pt"
            torch.save(checkpoint, best_path)
        
        # 3. 🧹 ROTATION: احذف القديم (احتفظ بآخر 3 best فقط)
        self._cleanup_old_checkpoints()
        
        # 4. Auto-sync to Windows
        self._sync_to_windows(path)
    
    def _cleanup_old_checkpoints(self):
        """حذف القديم - احتفظ بـ latest.pt + آخر 2 best"""
        best_files = sorted(self.checkpoint_dir.glob("best_acc*.pt"), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        # احذف القديم (أكثر من 2)
        for old_file in best_files[2:]:
            old_file.unlink()
            print(f"    🗑️ Removed old checkpoint: {old_file.name}")
    
    def _sync_to_windows(self, checkpoint_path):
        """مزامنة تلقائية للـ Windows"""
        try:
            import requests
            windows_api = os.getenv("WINDOWS_API", "http://192.168.68.109:8000")
            
            with open(checkpoint_path, "rb") as f:
                files = {"file": (checkpoint_path.name, f, "application/octet-stream")}
                response = requests.post(
                    f"{windows_api}/api/v1/checkpoints/upload/{self.name}",
                    files=files,
                    timeout=30
                )
                if response.status_code == 200:
                    print(f"    🔄 Auto-synced to Windows: {self.name}/{checkpoint_path.name}")
                else:
                    print(f"    ⚠️ Auto-sync failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"    ⚠️ Auto-sync error: {e}")
    
    def _load_checkpoint(self):
        """استعادة حالة التدريب"""
        path = self.checkpoint_dir / "latest.pt"
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                self.epoch = checkpoint['epoch']
                # استعادة الـ stats
                saved_stats = checkpoint.get('stats', {})
                self.stats['loss'] = saved_stats.get('loss', 0.0)
                self.stats['accuracy'] = saved_stats.get('accuracy', 0.0)
                self.stats['samples'] = saved_stats.get('samples', 0)
                self.stats['api_fetches'] = saved_stats.get('api_fetches', 0)
                print(f"    📂 Loaded checkpoint: Epoch {self.epoch}, Acc: {self.stats['accuracy']:.1f}%")
            except Exception as e:
                print(f"    ⚠️ Failed to load checkpoint: {e}")
        else:
            print(f"    🆕 Starting fresh training")

class RealAPILearningSystem:
    def __init__(self):
        self.layers = {}
        self.running = False
        
        configs = [
            ("president", "Strategic Decisions - GitHub + Markets"),
            ("seventh_dimension", "Future Planning - HackerNews"),
            ("high_council", "Collective Wisdom - Reddit Discussions"),
            ("shadow_light", "Risk Analysis - CVE + Markets"),
            ("scouts", "Intelligence - arXiv Research"),
            ("meta_team", "Optimization - StackOverflow"),
            ("domain_experts", "Expert Knowledge - arXiv"),
            ("execution", "Task Execution - GitHub + Reddit"),
            ("meta_architect", "Architecture - HackerNews + GitHub"),
            ("builder_council", "Software Dev - GitHub + StackOverflow"),
            ("executive_controller", "Control - Markets + Reddit"),
            ("guardian", "Cybersecurity - CVE Database"),
            ("cosmic_bridge", "API Integration - All Sources"),
            ("eternity", "Knowledge - arXiv + HackerNews"),
            ("learning_core", "Self-Improvement - StackOverflow")
        ]
        
        print("="*80)
        print("🌐 REAL API LEARNING SYSTEM - 15 LAYERS")
        print("="*80)
        print("📡 APIs:")
        print("   • GitHub API - Real trending repositories")
        print("   • CoinGecko API - Live crypto prices")
        print("   • HackerNews API - Tech news")
        print("   • Reddit API - Programming discussions")
        print("   • StackOverflow API - Real questions")
        print("   • CVE API - Security vulnerabilities")
        print("   • arXiv API - Research papers")
        print("="*80)
        
        for i, (name, desc) in enumerate(configs, 1):
            print(f"\n[{i:2d}/15] {name}")
            self.layers[name] = RealAPITrainer(name, desc)
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"      💾 VRAM: {vram:.2f} GB")
        
        total_params = sum(sum(p.numel() for p in t.model.parameters()) for t in self.layers.values())
        print(f"\n{'='*80}")
        print(f"✅ Total: {total_params/1e9:.2f}B Parameters")
        print(f"🌐 All layers fetch REAL data from Internet APIs")
        print(f"{'='*80}\n")
    
    def start_all(self):
        print("🚀 STARTING REAL API TRAINING...")
        print("🌐 Fetching live data from GitHub, CoinGecko, HackerNews, Reddit, arXiv, CVE...\n")
        self.running = True
        for name, trainer in self.layers.items():
            trainer.start()
            time.sleep(0.5)
        print("\n🔥 ALL LAYERS LEARNING FROM REAL APIs!")
    
    def stop_all(self):
        self.running = False
        for t in self.layers.values():
            t.stop()
    
    def get_status(self):
        gpu = {}
        if torch.cuda.is_available():
            try:
                gpu = {
                    "name": torch.cuda.get_device_name(0),
                    "utilization": torch.cuda.utilization(),
                    "memory_used": torch.cuda.memory_allocated() / 1e9,
                    "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9
                }
            except:
                pass
        
        # Count saved checkpoints
        total_checkpoints = 0
        for t in self.layers.values():
            if t.checkpoint_dir.exists():
                total_checkpoints += len(list(t.checkpoint_dir.glob("*.pt")))
        
        return {
            "is_training": self.running,
            "device": str(DEVICE),
            "mode": "REAL_API_LEARNING",
            "apis": ["GitHub", "CoinGecko", "HackerNews", "Reddit", "StackOverflow", "CVE", "arXiv"],
            "gpu": gpu,
            "checkpoints": {
                "saved_total": total_checkpoints,
                "location": "learning_data/checkpoints/",
                "auto_save_every": 50
            },
            "layers": {
                name: {
                    "epoch": t.epoch,
                    "loss": t.stats["loss"],
                    "accuracy": t.stats["accuracy"],
                    "samples": t.stats["samples"],
                    "api_fetches": t.stats["api_fetches"],
                    "specialization": t.desc
                }
                for name, t in self.layers.items()
            }
        }

real_api_learning_system = RealAPILearningSystem()

if __name__ == "__main__":
    real_api_learning_system.start_all()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        real_api_learning_system.stop_all()
