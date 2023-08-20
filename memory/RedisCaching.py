import json
import logging
import os
from collections import defaultdict

import redis
import numpy as np


class RedisCaching:
    def __init__(self, host=None, port=None, db=None, pool_size=None, password=None, config_file=None, th=0.5):
        if config_file is None:
            self.config_file = '../config.json'
        self.host = host
        self.port = port
        self.db = db
        self.pool_size = pool_size
        self.password = password
        self.th = th

        self.initConfig()

        self.r = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=self.pool_size
                             )

    def calculate_similarity(self, vec1, vec2):
        # 假设这里用余弦相似度计算嵌入向量的相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def query_cache(self, query: str = None, query_vec=None, k=3):
        results = []
        similarity_scores = defaultdict(float)

        # 查询第一层缓存
        cached_queries = self.r.keys('query:*')
        for cached_query in cached_queries:
            cached_query_str = self.r.hget(cached_query, 'query_str')
            cached_query_vec = np.fromstring(self.r.hget(cached_query, 'query_vec'), dtype=float)
            similarity = self.calculate_similarity(query_vec, cached_query_vec)

            if similarity >= self.th:
                cached_result = self.r.hget(cached_query, 'result')
                # results.append((cached_query_str, cached_result))
                # similarity_scores[cached_result] += similarity
                logging.info('命中第一层缓存。cached_result:', cached_query_str, cached_query_vec, cached_result)
                # return cached_result

        # 查询第二层缓存
        cached_results = self.r.zrangebyscore('cached_results', 0, float('inf'), withscores=True)
        for cached_result, score in cached_results:
            cached_result_str = cached_result.decode('utf-8')
            cached_result_vec = np.fromstring(self.r.hget(f'result:{cached_result_str}', 'result_vec'),
                                              dtype=float)
            similarity = self.calculate_similarity(query_vec, cached_result_vec)
            if similarity >= self.th:
                results.append(cached_result_str)
                # similarity_scores[cached_result_str] += similarity
                if similarity_scores[cached_result_str] >= self.th:  # 根据实际情况判断是否递增出现次数
                    self.r.zincrby(name='cached_results', value=cached_result_str, amount=1)  # 增加出现次数
                if len(results) >= k:
                    logging.info('命中第二层缓存: ', results)
                    return results

        return None

        if not results:
            # 从数据库中查询结果
            db_result = self.query_database(query)
            self.update_cache(query, query_vec, db_result)
            return db_result
        results.sort(key=lambda x: similarity_scores[x[1]], reverse=True)
        return results[:k]

    def query_database(self, query):
        logging.info('查询数据库')
        # 这里用伪代码代表从向量数据库查询结果的操作
        # 在实际应用中，你需要将查询向量和数据库中的向量进行比较
        db_result = [("Result 1", np.random.rand(10)), ("Result 2", np.random.rand(10))]
        return db_result

    def update_cache(self, query: str, query_vec, results):
        logging.info('更新缓存。')
        # 更新第一层缓存
        self.r.hset(f'query:{query}', 'query_str', query)
        self.r.hset(f'query:{query}', 'query_vec', query_vec.tobytes())
        results_string = [result[0] for result in results]
        self.r.hset(f'query:{query}', 'result', '___'.join(results_string))

        # 更新第二层缓存和相似节点计数
        for result in results:
            result_str, result_vec = result
            self.r.hset(f'result:{result_str}', 'result_str', result_str)
            self.r.hset(f'result:{result_str}', 'result_vec', result_vec.tobytes())
            self.r.zadd('cached_results', {result_str: 0})  # 初始化相似节点计数

    def clear_cache(self):
        # 清空缓存
        self.r.flushall()

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

if __name__ == '__main__':

    import numpy as np

    # 假设有一个查询向量和查询字符串
    query_vec = np.random.rand(10)
    query = "Query string"

    # 模拟数据库查询结果
    db_result = [("Result 1", np.random.rand(10)), ("Result 2", np.random.rand(10))]

    # 创建 RedisCaching 实例
    redis_caching = RedisCaching()

    # 查询缓存，首次查询应该命中数据库并更新缓存
    cached_results = redis_caching.query_cache(query, query_vec)
    print("Cached Results (1st Query):", cached_results)

    # 再次查询，这次应该命中缓存
    cached_results = redis_caching.query_cache(query, query_vec)
    print("Cached Results (2nd Query):", cached_results)

    # 清空缓存
    redis_caching.clear_cache()

    # 再次查询，现在应该再次命中数据库并更新缓存
    cached_results = redis_caching.query_cache(query, query_vec)
    print("Cached Results (After Cache Clear):", cached_results)
