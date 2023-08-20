import json
import logging
import os

import redis


class RedisHelper:
    def __init__(self, host=None, port=None, db=None, pool_size=None, password=None, config_file=None):
        self.config_file = '../config.json'
        self.host = host
        self.port = port
        self.db = db
        self.pool_size = pool_size
        self.password = password

        self.initConfig()

        # 创建连接池
        self.redis_pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            max_connections=self.pool_size,
            password=self.password
        )
        self.redis_conn = redis.StrictRedis(connection_pool=self.redis_pool,
                                            decode_responses=True
                                            )

    # String 操作
    def set(self, key, value, expire=None):
        """设置键值对到 Redis"""
        self.redis_conn.set(key, value, ex=expire)

    def get(self, key):
        """从 Redis 获取键对应的值"""
        return self.redis_conn.get(key)

    # Hash 操作
    def hset(self, name, key, value):
        """设置 Hash 中的字段值"""
        self.redis_conn.hset(name, key, value)

    def hget(self, name, key):
        """获取 Hash 中字段的值"""
        return self.redis_conn.hget(name, key)

    # 集合操作
    def sadd(self, name, *values):
        """向集合添加成员"""
        self.redis_conn.sadd(name, *values)

    def smembers(self, name):
        """获取集合中的所有成员"""
        return self.redis_conn.smembers(name)

    # 事务
    def execute_transaction(self, func):
        """执行一个事务"""
        with self.redis_conn.pipeline(transaction=True) as pipe:
            while True:
                try:
                    pipe.watch("key_to_watch")
                    pipe.multi()
                    func(pipe)
                    pipe.execute()
                    break
                except redis.WatchError:
                    continue

    # 发布订阅
    def publish(self, channel, message):
        """发布消息到指定频道"""
        self.redis_conn.publish(channel, message)

    def subscribe(self, channel, callback):
        """订阅指定频道并处理消息"""
        pubsub = self.redis_conn.pubsub()
        pubsub.subscribe(channel)
        for message in pubsub.listen():
            if message['type'] == 'message':
                callback(message['channel'], message['data'])

    def initConfig(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as config_file:
                config = json.load(config_file)
            redis_config = config['redis']

            host = None
            port = None
            db = None
            pool_size = None
            password = None

            if 'host' in redis_config.keys():
                host = redis_config['host']
            if 'port' in redis_config.keys():
                port = redis_config['port']
            if 'db' in redis_config.keys():
                db = redis_config['db']
            if 'pool_size' in redis_config.keys():
                pool_size = redis_config['pool_size']
            if 'password' in redis_config.keys():
                password = redis_config['password']

            if host is not None and host != "" and self.host is None:
                logging.info("set redis host as " + host)
                self.host = host
            if port is not None and port != -1 and self.port is None:
                logging.info("set redis port as " + str(port))
                self.port = port
            if db is not None and db != -1 and self.db is None:
                self.db = db
                logging.info("set redis db as " + str(db))
            if pool_size is not None and pool_size != -1 and self.pool_size is None:
                logging.info("set redis pool_size as " + str(pool_size))
                self.pool_size = pool_size
            if password is not None and password != "" and self.password is None:
                logging.info("set redis password as " + password)
                self.password = password


if __name__ == "__main__":
    redis_helper = RedisHelper()

    # 使用例子
    redis_helper.set("name", "Alice")
    print(redis_helper.get("name"))

    redis_helper.hset("user:1", "name", "Bob")
    print(redis_helper.hget("user:1", "name"))

    redis_helper.sadd("fruits", "apple", "banana", "orange")
    print(redis_helper.smembers("fruits"))

    def transaction_example(pipe):
        pipe.incr("counter")
        pipe.decr("balance")
    redis_helper.execute_transaction(transaction_example)

    def message_handler(channel, message):
        print(f"Received message from channel {channel}: {message}")
    redis_helper.subscribe("notifications", message_handler)

    redis_helper.publish("notifications", "New message!")

    # 注意：这个示例中的订阅示例是阻塞的，你可能需要在另一个终端中执行发布操作
