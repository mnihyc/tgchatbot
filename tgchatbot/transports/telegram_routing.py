"""Destination metadata for raw Telegram sends outside Message shortcuts."""
from __future__ import annotations


def topic_arguments(message) -> dict[str, int]:
    arguments = {}
    thread_id = getattr(message, 'message_thread_id', None)
    if thread_id is not None:
        arguments['message_thread_id'] = thread_id
    direct_topic = getattr(message, 'direct_messages_topic', None)
    if direct_topic is not None:
        arguments['direct_messages_topic_id'] = direct_topic.topic_id
    return arguments
