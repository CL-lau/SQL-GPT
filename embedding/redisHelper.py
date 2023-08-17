import redis

class RedisHelper:
    def __init__(self, host='localhost', port=6379, db=0, pool_size=10):
        # 创建连接池
        self.redis_pool = redis.ConnectionPool(host=host, port=port, db=db, max_connections=pool_size)
        self.redis_conn = redis.StrictRedis(connection_pool=self.redis_pool, decode_responses=True)

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
