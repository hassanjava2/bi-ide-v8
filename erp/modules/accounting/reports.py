"""
Financial Reports Generator - مولد التقارير المالية

تقارير مالية متوافقة مع GAAP:
- Trial Balance (ميزان المراجعة)
- Profit & Loss (قائمة الدخل)
- Cash Flow (التدفق النقدي)
- VAT/Tax Reports (تقارير الضرائب)

Exports: PDF, Excel
"""

import uuid
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass, field
import json


class ReportFormat(Enum):
    """صيغ التقرير"""
    JSON = "json"
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    HTML = "html"


class ReportType(Enum):
    """أنواع التقارير"""
    TRIAL_BALANCE = "trial_balance"
    PROFIT_LOSS = "profit_loss"
    CASH_FLOW = "cash_flow"
    BALANCE_SHEET = "balance_sheet"
    VAT_REPORT = "vat_report"
    GENERAL_LEDGER = "general_ledger"


@dataclass
class ReportPeriod:
    """فترة التقرير"""
    start_date: date
    end_date: date
    
    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "start": self.start_date.isoformat(),
            "end": self.end_date.isoformat(),
            "days": self.days
        }


class FinancialReportGenerator:
    """
    مولد التقارير المالية
    """
    
    def __init__(self, ledger=None):
        self.ledger = ledger
        self.reports_cache: Dict[str, Any] = {}
    
    def _get_period(self, start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   period: str = "month") -> ReportPeriod:
        """تحديد فترة التقرير"""
        if end_date is None:
            end_date = date.today()
        
        if start_date is None:
            if period == "month":
                start_date = end_date.replace(day=1)
            elif period == "quarter":
                quarter = (end_date.month - 1) // 3
                start_date = end_date.replace(month=quarter * 3 + 1, day=1)
            elif period == "year":
                start_date = end_date.replace(month=1, day=1)
            else:
                start_date = end_date - timedelta(days=30)
        
        return ReportPeriod(start_date, end_date)
    
    def generate_trial_balance(self, as_of: Optional[date] = None,
                               format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        """
        ميزان المراجعة (Trial Balance)
        التحقق من توازن الدفتر العام
        """
        if as_of is None:
            as_of = date.today()
        
        if self.ledger:
            trial_balance = self.ledger.get_trial_balance()
        else:
            trial_balance = {"accounts": [], "total_debits": 0, "total_credits": 0, "balanced": True}
        
        report = {
            "report_type": "Trial Balance",
            "report_type_ar": "ميزان المراجعة",
            "as_of": as_of.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": trial_balance
        }
        
        if format == ReportFormat.JSON:
            return report
        elif format == ReportFormat.CSV:
            return self._to_csv(report)
        elif format == ReportFormat.EXCEL:
            return self._to_excel(report)
        else:
            return report
    
    def generate_profit_loss(self, start_date: Optional[date] = None,
                            end_date: Optional[date] = None,
                            period: str = "month",
                            format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        """
        قائمة الدخل (Profit & Loss / Income Statement)
        """
        report_period = self._get_period(start_date, end_date, period)
        
        if self.ledger:
            income_stmt = self.ledger.get_income_statement(
                report_period.start_date, 
                report_period.end_date
            )
        else:
            income_stmt = {
                "revenue": {"total": 0, "accounts": []},
                "expenses": {"total": 0, "accounts": []},
                "net_income": 0,
                "gross_profit": 0,
                "operating_income": 0
            }
        
        report = {
            "report_type": "Profit & Loss",
            "report_type_ar": "قائمة الدخل",
            "period": report_period.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": income_stmt
        }
        
        return self._format_report(report, format)
    
    def generate_cash_flow(self, start_date: Optional[date] = None,
                          end_date: Optional[date] = None,
                          period: str = "month",
                          format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        """
        قائمة التدفقات النقدية (Cash Flow Statement)
        
        تقسيم النشاطات إلى:
        - Operating Activities (الأنشطة التشغيلية)
        - Investing Activities (الأنشطة الاستثمارية)
        - Financing Activities (الأنشطة التمويلية)
        """
        report_period = self._get_period(start_date, end_date, period)
        
        # Cash Flow Statement Structure
        cash_flow = {
            "operating_activities": {
                "net_income": 0,
                "adjustments": {
                    "depreciation": 0,
                    "changes_in_working_capital": {
                        "accounts_receivable": 0,
                        "inventory": 0,
                        "accounts_payable": 0,
                        "accrued_expenses": 0
                    }
                },
                "net_cash_from_operations": 0
            },
            "investing_activities": {
                "purchase_of_equipment": 0,
                "sale_of_assets": 0,
                "net_cash_from_investing": 0
            },
            "financing_activities": {
                "proceeds_from_loans": 0,
                "repayment_of_loans": 0,
                "capital_contributions": 0,
                "dividends_paid": 0,
                "net_cash_from_financing": 0
            },
            "net_change_in_cash": 0,
            "beginning_cash": 0,
            "ending_cash": 0
        }
        
        # Calculate from ledger if available
        if self.ledger:
            cash_account = self.ledger.get_account_balance("1110")
            cash_flow["ending_cash"] = float(cash_account)
            cash_flow["beginning_cash"] = float(cash_account)  # Simplified
            
            # Get transactions for the period
            transactions = [
                t for t in self.ledger.transactions
                if report_period.start_date <= t.date.date() <= report_period.end_date
            ]
            
            # Classify transactions
            for t in transactions:
                amount = float(t.amount)
                
                # Operating - linked to revenue/expense accounts
                if t.debit_account.startswith("4") or t.credit_account.startswith("5"):
                    cash_flow["operating_activities"]["net_cash_from_operations"] += amount
                
                # Investing - linked to fixed assets
                elif t.debit_account.startswith("12") or t.credit_account.startswith("12"):
                    cash_flow["investing_activities"]["net_cash_from_investing"] -= amount
                
                # Financing - linked to loans/equity
                elif t.debit_account.startswith("2") or t.credit_account.startswith("3"):
                    cash_flow["financing_activities"]["net_cash_from_financing"] += amount
        
        # Calculate totals
        cash_flow["net_change_in_cash"] = (
            cash_flow["operating_activities"]["net_cash_from_operations"] +
            cash_flow["investing_activities"]["net_cash_from_investing"] +
            cash_flow["financing_activities"]["net_cash_from_financing"]
        )
        
        report = {
            "report_type": "Cash Flow Statement",
            "report_type_ar": "قائمة التدفقات النقدية",
            "period": report_period.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": cash_flow
        }
        
        return self._format_report(report, format)
    
    def generate_vat_report(self, start_date: Optional[date] = None,
                           end_date: Optional[date] = None,
                           period: str = "quarter",
                           format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        """
        تقرير ضريبة القيمة المضافة (VAT Report)
        
        حساب:
        - Output VAT (ضريبة المخرجات - على المبيعات)
        - Input VAT (ضريبة المدخلات - على المشتريات)
        - Net VAT Payable/Receivable (الصافي)
        """
        report_period = self._get_period(start_date, end_date, period)
        
        # VAT rates (Saudi Arabia)
        VAT_RATE_STANDARD = Decimal("0.15")  # 15%
        VAT_RATE_ZERO = Decimal("0")
        VAT_RATE_EXEMPT = None
        
        vat_report = {
            "vat_period": report_period.to_dict(),
            "output_vat": {
                "standard_rated_sales": Decimal('0'),
                "vat_amount": Decimal('0'),
                "details": []
            },
            "input_vat": {
                "standard_rated_purchases": Decimal('0'),
                "vat_amount": Decimal('0'),
                "details": []
            },
            "net_vat": Decimal('0'),
            "status": "payable"  # or "refundable"
        }
        
        # Calculate from ledger transactions
        if self.ledger:
            for t in self.ledger.transactions:
                if report_period.start_date <= t.date.date() <= report_period.end_date:
                    # Sales (Revenue accounts)
                    if t.credit_account.startswith("4"):
                        amount = t.amount
                        vat = amount * VAT_RATE_STANDARD
                        vat_report["output_vat"]["standard_rated_sales"] += amount
                        vat_report["output_vat"]["vat_amount"] += vat
                        vat_report["output_vat"]["details"].append({
                            "date": t.date.isoformat(),
                            "reference": t.reference,
                            "amount": float(amount),
                            "vat": float(vat)
                        })
                    
                    # Purchases (Expense/Asset accounts)
                    elif t.debit_account.startswith("5") or t.debit_account.startswith("1"):
                        amount = t.amount
                        vat = amount * VAT_RATE_STANDARD
                        vat_report["input_vat"]["standard_rated_purchases"] += amount
                        vat_report["input_vat"]["vat_amount"] += vat
                        vat_report["input_vat"]["details"].append({
                            "date": t.date.isoformat(),
                            "reference": t.reference,
                            "amount": float(amount),
                            "vat": float(vat)
                        })
        
        # Calculate net VAT
        output_vat = vat_report["output_vat"]["vat_amount"]
        input_vat = vat_report["input_vat"]["vat_amount"]
        vat_report["net_vat"] = output_vat - input_vat
        vat_report["status"] = "payable" if vat_report["net_vat"] > 0 else "refundable"
        
        report = {
            "report_type": "VAT Report",
            "report_type_ar": "تقرير ضريبة القيمة المضافة",
            "period": report_period.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": {
                "output_vat": {
                    "standard_rated_sales": float(vat_report["output_vat"]["standard_rated_sales"]),
                    "vat_amount": float(vat_report["output_vat"]["vat_amount"]),
                    "details": vat_report["output_vat"]["details"]
                },
                "input_vat": {
                    "standard_rated_purchases": float(vat_report["input_vat"]["standard_rated_purchases"]),
                    "vat_amount": float(vat_report["input_vat"]["vat_amount"]),
                    "details": vat_report["input_vat"]["details"]
                },
                "net_vat": float(vat_report["net_vat"]),
                "status": vat_report["status"]
            }
        }
        
        return self._format_report(report, format)
    
    def generate_balance_sheet(self, as_of: Optional[date] = None,
                              format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        """
        قائمة المركز المالي (Balance Sheet)
        """
        if as_of is None:
            as_of = date.today()
        
        if self.ledger:
            balance_sheet = self.ledger.get_balance_sheet(as_of)
        else:
            balance_sheet = {
                "assets": {"total": 0, "current_assets": {"total": 0}, "fixed_assets": {"total": 0}},
                "liabilities": {"total": 0, "current_liabilities": {"total": 0}},
                "equity": {"total": 0},
                "balanced": True
            }
        
        report = {
            "report_type": "Balance Sheet",
            "report_type_ar": "قائمة المركز المالي",
            "as_of": as_of.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": balance_sheet
        }
        
        return self._format_report(report, format)
    
    def _format_report(self, report: Dict[str, Any], 
                      format: ReportFormat) -> Dict[str, Any]:
        """تنسيق التقرير حسب الصيغة المطلوبة"""
        if format == ReportFormat.JSON:
            return report
        elif format == ReportFormat.CSV:
            return self._to_csv(report)
        elif format == ReportFormat.EXCEL:
            return self._to_excel(report)
        elif format == ReportFormat.HTML:
            return self._to_html(report)
        elif format == ReportFormat.PDF:
            return self._to_pdf(report)
        return report
    
    def _to_csv(self, report: Dict[str, Any]) -> str:
        """تحويل التقرير إلى CSV"""
        lines = []
        lines.append(f"Report: {report['report_type']}")
        lines.append(f"Period: {report.get('period', {}).get('start', 'N/A')} to {report.get('period', {}).get('end', 'N/A')}")
        lines.append("")
        
        # Convert data to CSV format
        data = report.get("data", {})
        
        def flatten_dict(d, prefix=""):
            items = []
            for k, v in d.items():
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, f"{prefix}{k}_"))
                elif isinstance(v, (int, float, str)):
                    items.append(f"{prefix}{k},{v}")
            return items
        
        lines.extend(flatten_dict(data))
        return "\n".join(lines)
    
    def _to_excel(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """تحضير بيانات Excel (للتصدير الفعلي يحتاج مكتبة openpyxl)"""
        # Return structured data that can be converted to Excel
        return {
            "format": "excel",
            "filename": f"{report['report_type'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "sheets": {
                "Report": report,
                "Data": report.get("data", {})
            }
        }
    
    def _to_html(self, report: Dict[str, Any]) -> str:
        """تحويل التقرير إلى HTML"""
        html = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <title>{report['report_type']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .header {{ margin-bottom: 30px; }}
                .total {{ font-weight: bold; background-color: #e7f3fe; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report['report_type_ar']} ({report['report_type']})</h1>
                <p>Generated: {report['generated_at']}</p>
            </div>
            <pre>{json.dumps(report.get('data', {}), indent=2, ensure_ascii=False)}</pre>
        </body>
        </html>
        """
        return html
    
    def _to_pdf(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """تحضير بيانات PDF (للتصدير الفعلي يحتاج مكتبة مثل ReportLab)"""
        return {
            "format": "pdf",
            "filename": f"{report['report_type'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
            "html_content": self._to_html(report),
            "data": report
        }
    
    def generate_comprehensive_report(self, start_date: Optional[date] = None,
                                     end_date: Optional[date] = None,
                                     period: str = "month") -> Dict[str, Any]:
        """
        تقرير شامل يتضمن جميع التقارير المالية
        """
        return {
            "report_title": "Comprehensive Financial Report",
            "report_title_ar": "التقرير المالي الشامل",
            "period": self._get_period(start_date, end_date, period).to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reports": {
                "trial_balance": self.generate_trial_balance(format=ReportFormat.JSON),
                "profit_loss": self.generate_profit_loss(start_date, end_date, period, ReportFormat.JSON),
                "cash_flow": self.generate_cash_flow(start_date, end_date, period, ReportFormat.JSON),
                "balance_sheet": self.generate_balance_sheet(end_date, ReportFormat.JSON),
                "vat_report": self.generate_vat_report(start_date, end_date, period, ReportFormat.JSON)
            }
        }
