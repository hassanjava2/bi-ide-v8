"""
Email Service - خدمة الإيميل
Handles password reset and notification emails
"""
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional


class EmailService:
    """Email service for sending notifications and password resets"""
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("SMTP_FROM", "noreply@bi-ide.com")
        self.enabled = all([self.smtp_user, self.smtp_password])
    
    async def send_password_reset_email(self, to_email: str, token: str) -> bool:
        """
        Send password reset email with reset link
        
        Args:
            to_email: User's email address
            token: Reset token
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            # Development mode: log the link
            reset_link = f"http://localhost:8000/reset-password?token={token}"
            print(f"\n{'='*60}")
            print(f"📧 DEVELOPMENT MODE - Password Reset Email")
            print(f"{'='*60}")
            print(f"To: {to_email}")
            print(f"Reset Link: {reset_link}")
            print(f"{'='*60}\n")
            return True
        
        try:
            # Production: Send actual email
            subject = "Password Reset Request - BI-IDE v8"
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email
            
            # HTML content
            reset_link = f"https://your-domain.com/reset-password?token={token}"
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2>Password Reset Request</h2>
                <p>You requested a password reset for your BI-IDE v8 account.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href="{reset_link}" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                <p>Or copy this URL: {reset_link}</p>
                <p><strong>This link expires in 1 hour.</strong></p>
                <p>If you didn't request this, please ignore this email.</p>
                <hr>
                <p style="color: #666; font-size: 12px;">BI-IDE v8 - AI-Powered Enterprise Platform</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, "html"))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, to_email, msg.as_string())
            
            print(f"✅ Password reset email sent to {to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email to {to_email}: {e}")
            return False


# Global instance
email_service = EmailService()


async def send_password_reset_email(email: str, token: str) -> bool:
    """Convenience function to send password reset email"""
    return await email_service.send_password_reset_email(email, token)
