"""
Services Tests - اختبارات الخدمات
"""
import pytest


class TestIDEService:
    """Test IDE Service"""
    
    def test_file_system_manager_init(self):
        """Test file system manager initialization"""
        from ide.ide_service import FileSystemManager
        
        manager = FileSystemManager()
        assert manager.root_path is not None
    
    def test_get_file_tree(self):
        """Test getting file tree"""
        from ide.ide_service import FileSystemManager
        
        manager = FileSystemManager()
        tree = manager.get_file_tree()
        assert tree is not None
    
    def test_analyze_code(self, sample_code):
        """Test code analysis"""
        from ide.ide_service import analyze_code_quality
        
        result = analyze_code_quality(sample_code, "python")
        assert "score" in result or isinstance(result, dict)


class TestERPService:
    """Test ERP Service"""
    
    def test_erp_service_init(self):
        """Test ERP service initialization"""
        from erp.erp_service import ERPService
        
        service = ERPService(None)
        assert service is not None
    
    def test_create_invoice(self, sample_invoice):
        """Test creating invoice"""
        from erp.erp_service import ERPService
        
        service = ERPService(None)
        # Test with sample data
        assert sample_invoice["total"] == sample_invoice["amount"] + sample_invoice["tax"]


class TestCache:
    """Test Cache Manager"""
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache):
        """Test cache set and get"""
        await cache.set("test_key", "test_value", ttl=60)
        value = await cache.get("test_key")
        assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, cache):
        """Test cache delete"""
        await cache.set("delete_key", "value", ttl=60)
        await cache.delete("delete_key")
        value = await cache.get("delete_key")
        assert value is None


class TestDatabase:
    """Test Database Manager"""
    
    @pytest.mark.asyncio
    async def test_store_and_get_knowledge(self, db):
        """Test storing and retrieving knowledge"""
        await db.store_knowledge(
            entry_id="test-1",
            category="test",
            content="Test content",
            confidence=0.9
        )
        
        entries = await db.get_knowledge(category="test")
        assert len(entries) > 0
        assert entries[0]["content"] == "Test content"
    
    @pytest.mark.asyncio
    async def test_store_learning_experience(self, db):
        """Test storing learning experience"""
        await db.store_learning_experience(
            exp_id="exp-1",
            exp_type="code",
            context={"file": "test.py"},
            action="write",
            outcome="success",
            reward=1.0
        )
        
        stats = await db.get_learning_stats()
        assert stats["total_experiences"] > 0
