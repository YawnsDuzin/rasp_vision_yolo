"""
Notification System
알림 시스템 - 텔레그램, 이메일 등
"""

import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage


class TelegramNotifier:
    """텔레그램 봇을 통한 알림"""

    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier

        Args:
            bot_token: 텔레그램 봇 토큰
            chat_id: 메시지를 받을 채팅 ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = logging.getLogger(__name__)

    def send_message(self, message: str) -> bool:
        """
        텍스트 메시지 전송

        Args:
            message: 전송할 메시지

        Returns:
            성공 여부
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }

            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()

            self.logger.info("텔레그램 메시지 전송 성공")
            return True

        except Exception as e:
            self.logger.error(f"텔레그램 메시지 전송 실패: {e}")
            return False

    def send_photo(
        self,
        image_path: str,
        caption: Optional[str] = None
    ) -> bool:
        """
        이미지와 함께 메시지 전송

        Args:
            image_path: 이미지 파일 경로
            caption: 이미지 설명

        Returns:
            성공 여부
        """
        try:
            url = f"{self.base_url}/sendPhoto"

            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption or "",
                    "parse_mode": "HTML"
                }

                response = requests.post(url, data=data, files=files, timeout=30)
                response.raise_for_status()

            self.logger.info("텔레그램 이미지 전송 성공")
            return True

        except Exception as e:
            self.logger.error(f"텔레그램 이미지 전송 실패: {e}")
            return False

    def send_alert(
        self,
        alert_type: str,
        message: str,
        image_path: Optional[str] = None
    ) -> bool:
        """
        알림 전송 (이모지 포함)

        Args:
            alert_type: 알림 유형 (intrusion, violation, count 등)
            message: 알림 메시지
            image_path: 첨부할 이미지 (선택)

        Returns:
            성공 여부
        """
        # 알림 유형별 이모지
        emoji_map = {
            'intrusion': '🚨',
            'violation': '⚠️',
            'count': '📊',
            'alert': '🔔',
            'warning': '⚡',
            'info': 'ℹ️'
        }

        emoji = emoji_map.get(alert_type, '📢')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        formatted_message = (
            f"{emoji} <b>{alert_type.upper()}</b>\n\n"
            f"{message}\n\n"
            f"🕒 {timestamp}"
        )

        if image_path and Path(image_path).exists():
            return self.send_photo(image_path, formatted_message)
        else:
            return self.send_message(formatted_message)


class EmailNotifier:
    """이메일을 통한 알림"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str
    ):
        """
        Initialize email notifier

        Args:
            smtp_server: SMTP 서버 주소 (예: smtp.gmail.com)
            smtp_port: SMTP 포트 (예: 587)
            sender_email: 발신자 이메일
            sender_password: 발신자 비밀번호 (또는 앱 비밀번호)
            recipient_email: 수신자 이메일
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.logger = logging.getLogger(__name__)

    def send_email(
        self,
        subject: str,
        body: str,
        image_path: Optional[str] = None
    ) -> bool:
        """
        이메일 전송

        Args:
            subject: 이메일 제목
            body: 이메일 본문
            image_path: 첨부할 이미지 (선택)

        Returns:
            성공 여부
        """
        try:
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            # 본문 추가
            msg.attach(MIMEText(body, 'html'))

            # 이미지 첨부
            if image_path and Path(image_path).exists():
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-ID', '<image1>')
                    img.add_header(
                        'Content-Disposition',
                        'attachment',
                        filename=Path(image_path).name
                    )
                    msg.attach(img)

            # SMTP 연결 및 전송
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            self.logger.info("이메일 전송 성공")
            return True

        except Exception as e:
            self.logger.error(f"이메일 전송 실패: {e}")
            return False

    def send_alert(
        self,
        alert_type: str,
        message: str,
        image_path: Optional[str] = None
    ) -> bool:
        """
        알림 이메일 전송

        Args:
            alert_type: 알림 유형
            message: 알림 메시지
            image_path: 첨부할 이미지

        Returns:
            성공 여부
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        subject = f"[{alert_type.upper()}] Vision System Alert - {timestamp}"

        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #ff4444; color: white; padding: 10px; }}
                .content {{ padding: 20px; }}
                .footer {{ color: #666; font-size: 12px; padding: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 {alert_type.upper()} Alert</h2>
            </div>
            <div class="content">
                <p>{message}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                {'<p><img src="cid:image1" alt="Detection Image" style="max-width:600px;"></p>' if image_path else ''}
            </div>
            <div class="footer">
                <p>This is an automated message from Raspberry Pi YOLO Vision System.</p>
            </div>
        </body>
        </html>
        """

        return self.send_email(subject, body, image_path)


class MultiNotifier:
    """여러 알림 채널을 통합 관리"""

    def __init__(self):
        """Initialize multi-channel notifier"""
        self.notifiers = []
        self.logger = logging.getLogger(__name__)

    def add_telegram(self, bot_token: str, chat_id: str):
        """텔레그램 알림 추가"""
        notifier = TelegramNotifier(bot_token, chat_id)
        self.notifiers.append(('telegram', notifier))
        self.logger.info("텔레그램 알림 채널 추가됨")

    def add_email(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str
    ):
        """이메일 알림 추가"""
        notifier = EmailNotifier(
            smtp_server,
            smtp_port,
            sender_email,
            sender_password,
            recipient_email
        )
        self.notifiers.append(('email', notifier))
        self.logger.info("이메일 알림 채널 추가됨")

    def send_alert(
        self,
        alert_type: str,
        message: str,
        image_path: Optional[str] = None,
        channels: Optional[list] = None
    ) -> dict:
        """
        모든 활성화된 채널로 알림 전송

        Args:
            alert_type: 알림 유형
            message: 알림 메시지
            image_path: 첨부 이미지
            channels: 특정 채널만 사용 (None이면 모든 채널)

        Returns:
            채널별 전송 결과
        """
        results = {}

        for channel_type, notifier in self.notifiers:
            # 특정 채널만 사용하는 경우
            if channels and channel_type not in channels:
                continue

            try:
                success = notifier.send_alert(alert_type, message, image_path)
                results[channel_type] = success
            except Exception as e:
                self.logger.error(f"{channel_type} 알림 전송 실패: {e}")
                results[channel_type] = False

        return results


# 사용 예시
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # 환경 변수에서 설정 로드
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        # 텔레그램 알림 테스트
        notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

        # 텍스트 메시지
        notifier.send_message("테스트 메시지입니다.")

        # 알림 전송
        notifier.send_alert(
            'intrusion',
            '침입자가 감지되었습니다!\n위치: 정문\n시간: 14:30'
        )

        print("텔레그램 알림 전송 완료")
    else:
        print("환경 변수 설정이 필요합니다:")
        print("TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
