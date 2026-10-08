"""Deterministic text rendering of source objects.

Every system reads `payload["text"]`, produced here from the object's fields, so the raw text a
zero system retrieves, the text HoldMem's writer quotes and the text the checker verifies spans
against are the same bytes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _when(s: str) -> str:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").strftime("%A, %Y-%m-%d %H:%M")


def render_text(source: str, p: dict[str, Any]) -> str:
    if source == "email":
        return (
            f"Email\nFrom: {p['from_name']} <{p['from_addr']}>\nTo: {p['to']}\n"
            f"Date: {p['sent']}\nSubject: {p['subject']}\n\n{p['body']}"
        )
    if source == "messages":
        return (
            f"Chat message (conversation with {p['conversation_with']})\n"
            f"From: {p['sender']}\nDate: {p['sent']}\n{p['body']}"
        )
    if source == "notes":
        return f"Note: {p['title']}\nLast edited: {p.get('updated', '')}\n{p['body']}"
    if source == "contacts":
        return (
            f"Contact card: {p['name']}\nEmail: {p['email']}\nPhone: {p['phone']}\n"
            f"Company: {p['employer']}\nCity: {p['city']}\nRelationship: {p['relation']}\n"
            f"Starred: {'yes' if p.get('starred') else 'no'}"
        )
    if source == "tasks":
        return (
            f"Task: {p['title']}\nProject: {p['project']}\nDue: {p['due']}\n"
            f"Status: {p['status']}\nPriority: {p['priority']}\n"
            f"Last updated: {p.get('updated', '')}"
        )
    if source == "calendar":
        kind = p.get("kind")
        if kind == "settings":
            return f"Calendar settings\nTime zone: {p['timezone']}"
        if kind == "series":
            return (
                f"Recurring calendar event: {p['title']}\n"
                f"Repeats: every {p['weekday']} at {p['time']} ({p['duration_min']} min)\n"
                f"Attendees: {', '.join(p['attendees'])}\nCategory: {p['category']}"
            )
        if kind == "appointment":
            return (
                f"Calendar event: {p['title']}\nWhen: {_when(p['start'])} "
                f"({p['duration_min']} min)\nType: {p['appt_type']} appointment"
            )
        if kind == "meeting":
            end = datetime.strptime(p["end"], "%Y-%m-%d %H:%M").strftime("%H:%M")
            return (
                f"Calendar event: {p['title']}\nWhen: {_when(p['start'])}-{end}\n"
                f"Attendees: {', '.join(p['attendees'])}\nCategory: {p['category']}"
            )
    raise ValueError(f"cannot render {source} / {p.get('kind')}")
