import logging


def error(msg, *args, **kwargs):
    """
    Log a message with severity 'ERROR' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    logging.error(msg, *args, **kwargs)
    """
    后续增加切面
    """


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    logging.info(msg, *args, **kwargs)
    """
    后续增加切面
    """


def warning(msg, *args, **kwargs):
    """
    Log a message with severity 'WARNING' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    logging.warning(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    warning(msg, *args, **kwargs)
