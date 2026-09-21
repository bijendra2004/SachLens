from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any


logger = logging.getLogger("sachlens.mailer")


class EmailDeliveryError(RuntimeError):
    pass


_OTP_HTML_TEMPLATE = """
<div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 20px;">
    <h2 style="color: #333;">🔐 Your SachLens OTP Code</h2>
    <p style="font-size: 16px; color: #555;">Use the following code to verify your identity:</p>
    <div style="background: #f4f4f4; padding: 20px; text-align: center; border-radius: 8px; margin: 20px 0;">
        <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #222;">{otp}</span>
    </div>
    <p style="font-size: 14px; color: #777;">This code expires in {expires_minutes} minutes.</p>
    <p style="font-size: 14px; color: #777;">If you did not request this, you can safely ignore this email.</p>
    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
    <p style="font-size: 12px; color: #aaa;">— {sender_name}</p>
</div>
"""


def send_otp_email(email: str, otp: str, expires_minutes: int = 10) -> None:
    """Send OTP email with fast automatic failover across configured providers.

    Configurable via EMAIL_PROVIDER (e.g. 'resend', 'brevo', 'smtp').
    """
    sender_name = os.getenv("SMTP_FROM_NAME", "SachLens").strip() or "SachLens"
    preferred_provider = os.getenv("EMAIL_PROVIDER", "").strip().lower()

    brevo_api_key = os.getenv("BREVO_API_KEY", "").strip()
    resend_api_key = os.getenv("RESEND_API_KEY", "").strip()
    smtp_host = os.getenv("SMTP_HOST", "").strip()

    providers_sequence: list[str] = []
    if preferred_provider in ("resend", "brevo", "smtp"):
        providers_sequence.append(preferred_provider)

    for p in ["resend" if preferred_provider == "resend" else "brevo", "resend", "smtp"]:
        if p not in providers_sequence:
            providers_sequence.append(p)

    attempted_providers: list[str] = []
    errors: list[str] = []

    for provider in providers_sequence:
        if provider == "brevo" and brevo_api_key:
            attempted_providers.append("Brevo")
            try:
                _send_via_brevo(email, otp, expires_minutes, brevo_api_key, sender_name)
                logger.info("OTP email delivered successfully via Brevo to %s", email)
                return
            except Exception as error:
                logger.warning("Brevo delivery failed for %s: %s; trying next provider", email, error)
                errors.append(f"Brevo: {error}")

        elif provider == "resend" and resend_api_key:
            attempted_providers.append("Resend")
            try:
                _send_via_resend(email, otp, expires_minutes, resend_api_key, sender_name)
                logger.info("OTP email delivered successfully via Resend to %s", email)
                return
            except Exception as error:
                logger.warning("Resend delivery failed for %s: %s; trying next provider", email, error)
                errors.append(f"Resend: {error}")

        elif provider == "smtp" and smtp_host:
            attempted_providers.append("SMTP")
            try:
                _send_via_smtp(email, otp, expires_minutes)
                logger.info("OTP email delivered successfully via SMTP to %s", email)
                return
            except Exception as error:
                logger.warning("SMTP delivery failed for %s: %s", email, error)
                errors.append(f"SMTP: {error}")

    # If no providers configured or all failed
    if not attempted_providers:
        if os.getenv("APP_ENV", "development").lower() == "production":
            raise EmailDeliveryError("Email delivery is not configured. Set BREVO_API_KEY, RESEND_API_KEY, or SMTP_HOST.")
        logger.info("Email not configured; OTP email skipped for %s in non-production", email)
        return

    raise EmailDeliveryError(f"All email delivery providers failed for {email}: {'; '.join(errors)}")


def _send_via_brevo(email: str, otp: str, expires_minutes: int, api_key: str, sender_name: str) -> None:
    """Send OTP email via Brevo HTTP API with detailed HTTP error extraction."""
    from_email = (
        os.getenv("BREVO_FROM_EMAIL")
        or os.getenv("SMTP_FROM_EMAIL")
        or "sachlensuserauth@gmail.com"
    ).strip()
    subject = os.getenv("OTP_EMAIL_SUBJECT", "Your SachLens OTP Code")

    html_body = _OTP_HTML_TEMPLATE.format(otp=otp, expires_minutes=expires_minutes, sender_name=sender_name)

    payload = json.dumps({
        "sender": {"name": sender_name, "email": from_email},
        "to": [{"email": email}],
        "subject": subject,
        "htmlContent": html_body,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=payload,
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SachLens/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body = resp.read().decode("utf-8")
            logger.info("Brevo response status=%s body=%s", resp.status, resp_body)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        logger.error("Brevo API HTTP %d error: %s (sender=%s)", exc.code, err_body, from_email)
        raise EmailDeliveryError(f"Brevo HTTP {exc.code}: {err_body}") from exc
    except Exception as exc:
        logger.error("Brevo request failed: %s", exc)
        raise EmailDeliveryError(f"Brevo request failed: {exc}") from exc


def _send_via_resend(email: str, otp: str, expires_minutes: int, api_key: str, sender_name: str) -> None:
    """Send OTP email via Resend HTTP API with detailed HTTP error extraction."""
    from_email = (
        os.getenv("RESEND_FROM_EMAIL")
        or "onboarding@resend.dev"
    ).strip()
    subject = os.getenv("OTP_EMAIL_SUBJECT", "Your SachLens OTP Code")

    html_body = _OTP_HTML_TEMPLATE.format(otp=otp, expires_minutes=expires_minutes, sender_name=sender_name)

    payload = json.dumps({
        "from": f"{sender_name} <{from_email}>",
        "to": [email],
        "subject": subject,
        "html": html_body,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SachLens/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body = resp.read().decode("utf-8")
            logger.info("Resend response status=%s body=%s", resp.status, resp_body)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        logger.error("Resend API HTTP %d error: %s (sender=%s)", exc.code, err_body, from_email)
        raise EmailDeliveryError(f"Resend HTTP {exc.code}: {err_body}") from exc
    except Exception as exc:
        logger.error("Resend request failed: %s", exc)
        raise EmailDeliveryError(f"Resend request failed: {exc}") from exc


def _send_via_smtp(email: str, otp: str, expires_minutes: int) -> None:
    """Send OTP email via traditional SMTP with short connection timeout."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_use_tls = _parse_bool(os.getenv("SMTP_USE_TLS", "true"))
    smtp_use_ssl = _parse_bool(os.getenv("SMTP_USE_SSL", "false"))

    sender_email = os.getenv("SMTP_FROM_EMAIL", smtp_user).strip()
    sender_name = os.getenv("SMTP_FROM_NAME", "SachLens").strip() or "SachLens"
    if not sender_email:
        raise EmailDeliveryError("SMTP_FROM_EMAIL or SMTP_USERNAME must be configured")

    subject = os.getenv("OTP_EMAIL_SUBJECT", "Your SachLens OTP Code")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{sender_name} <{sender_email}>"
    message["To"] = email
    message.set_content(
        (
            f"Your SachLens OTP is: {otp}\n\n"
            f"This OTP expires in {expires_minutes} minutes.\n"
            "If you did not request this, you can ignore this email.\n"
        )
    )

    if smtp_use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=6, context=ssl.create_default_context()) as smtp:
            if smtp_user:
                smtp.login(smtp_user, smtp_password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=6) as smtp:
            smtp.ehlo()
            if smtp_use_tls:
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            if smtp_user:
                smtp.login(smtp_user, smtp_password)
            smtp.send_message(message)


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}