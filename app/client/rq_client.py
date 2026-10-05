import os

from redis import Redis
from rq import Queue


redis_connection = Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

queue = Queue("rag", connection=redis_connection)
