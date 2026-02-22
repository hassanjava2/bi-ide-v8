"""
توصيل النظام الهرمي بالخدمات
Connects AI Hierarchy to IDE + ERP for Observational Learning
"""

from .autonomous_learning import get_learning_system


def connect_ide_to_learning(ide_service, hierarchy):
    """
    توصيل الـ IDE بنظام التعلم
    """
    learning = get_learning_system(hierarchy)
    
    # ربط events الـ IDE
    original_save = ide_service.fs.save_file if hasattr(ide_service.fs, 'save_file') else None
    
    def save_and_learn(file_id, content):
        """يحفظ الملف ويتعلم"""
        # الحفظ الأصلي
        if original_save:
            result = original_save(file_id, content)
        else:
            result = True
        
        # التعلم
        if learning:
            language = "python" if file_id.endswith('.py') else "javascript"
            learning.observe_code_written(file_id, content, language)
        
        return result
    
    # استبدال الدالة
    if hasattr(ide_service.fs, 'save_file'):
        ide_service.fs.save_file = save_and_learn
    
    print("💻 IDE connected to Learning System")
    return ide_service


def connect_erp_to_learning(erp_service, hierarchy):
    """
    توصيل الـ ERP بنظام التعلم
    """
    learning = get_learning_system(hierarchy)
    
    # ربط إنشاء الفواتير
    original_create_invoice = erp_service.accounting.create_invoice if hasattr(erp_service.accounting, 'create_invoice') else None
    
    def create_and_learn(data):
        """ينشئ فاتورة ويتعلم"""
        # الإنشاء الأصلي
        if original_create_invoice:
            invoice = original_create_invoice(data)
        else:
            invoice = None
        
        # التعلم
        if learning and invoice:
            learning.observe_erp_transaction(
                "create_invoice",
                {"amount": invoice.amount, "customer": invoice.customer_name},
                "success"
            )
        
        return invoice
    
    # استبدال الدالة
    if hasattr(erp_service.accounting, 'create_invoice'):
        erp_service.accounting.create_invoice = create_and_learn
    
    print("🏢 ERP connected to Learning System")
    return erp_service


def connect_api_to_learning(api_app, hierarchy):
    """
    توصيل الـ API بنظام التعلم
    """
    learning = get_learning_system(hierarchy)
    
    # ربط تنفيذ الأوامر
    if hasattr(api_app, 'execute_command'):
        original_execute = api_app.execute_command
        
        async def execute_and_learn(request):
            """ينفذ الأمر ويتعلم"""
            result = await original_execute(request)
            
            if learning:
                learning.observe_command(
                    request.command,
                    request.alert_level,
                    result
                )
            
            return result
        
        api_app.execute_command = execute_and_learn
    
    print("🌐 API connected to Learning System")
    return api_app


def connect_all_services(hierarchy, ide_service=None, erp_service=None, api_app=None):
    """
    توصيل كل الخدمات
    """
    print("\n🔗 Connecting Services to Learning System...")
    
    if ide_service:
        connect_ide_to_learning(ide_service, hierarchy)
    
    if erp_service:
        connect_erp_to_learning(erp_service, hierarchy)
    
    if api_app:
        connect_api_to_learning(api_app, hierarchy)
    
    print("✅ All services connected!")
    print("💡 The system is now learning from your work automatically\n")
