import redis
import os

redis_client = redis.Redis.from_url(os.getenv('REDIS_URL'))