# utils/helpers.py
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neo_stats")




def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.exception("Error in safe_call: %s", e)
        return None