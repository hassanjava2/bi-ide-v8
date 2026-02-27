# CRM Module - وحدة إدارة العملاء
"""
Customer Relationship Management Module - وحدة إدارة علاقات العملاء

المميزات:
- Customer management
- Sales pipeline
- Support tickets
"""

from .customers import Customer, CustomerManager, CustomerClassification
from .sales_pipeline import SalesPipeline, Deal, PipelineStage
from .support_tickets import SupportTicket, TicketManager, TicketStatus, TicketPriority

__all__ = [
    'Customer', 'CustomerManager', 'CustomerClassification',
    'SalesPipeline', 'Deal', 'PipelineStage',
    'SupportTicket', 'TicketManager', 'TicketStatus', 'TicketPriority',
]
