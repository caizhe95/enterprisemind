import asyncio 
 
from cache.cache_manager import CacheStats, RedisPersistentCache 
from graph.agents.search import search_node 
from tools.postgres_sql_tool import sql_query_async 
from tools.tavily_tool import tavily_search_async 
 
 
def test_tavily_search_async_uses_httpx(monkeypatch): 
    captured = {} 
 
    class FakeResponse: 
        def raise_for_status(self): 
            return None 
 
        def json(self): 
            return {'answer': 'test-answer', 'results': [{'title': 'Result', 'content': 'Body', 'url': 'https://example.com'}]} 
 
    class FakeAsyncClient: 
        def __init__(self, timeout): 
            captured['timeout'] = timeout 
        async def __aenter__(self): 
            return self 
        async def __aexit__(self, exc_type, exc, tb): 
            return None 
        async def post(self, url, json): 
            captured['url'] = url 
            captured['json'] = json 
            return FakeResponse() 
 
    monkeypatch.setenv('TAVILY_API_KEY', 'test-key') 
    monkeypatch.setattr('tools.tavily_tool.httpx.AsyncClient', FakeAsyncClient) 
    result = asyncio.run(tavily_search_async('iphone 15 price', search_depth='advanced', max_results=3)) 
    assert result['answer'] == 'test-answer' 
    assert captured['json']['search_depth'] == 'advanced' 
 
 
def test_search_node_fanout_merges_async_results(monkeypatch): 
    async def fake_tavily_search_async(query, search_depth='basic', max_results=5): 
        return {'answer': f'{search_depth}-answer', 'results': [{'title': f'{search_depth}-title', 'content': f'{search_depth}-content', 'url': f'https://example.com/{search_depth}'}]} 
 
    class FakeEvaluator: 
        def evaluate_retrieval(self, question, docs): 
            return {'details': docs} 
 
    monkeypatch.setattr('graph.agents.search.tavily_search_async', fake_tavily_search_async) 
    monkeypatch.setattr('graph.agents.search.get_self_rag_evaluator', lambda: FakeEvaluator()) 
    result = search_node({'question': 'latest phone price', 'worker_input': 'latest phone price'}) 
    assert len(result['retrieved_docs']) >= 4  
    assert result['tool_results'][0]['result']['basic']['answer'] == 'basic-answer' 
    assert result['tool_results'][0]['result']['advanced']['answer'] == 'advanced-answer' 
 
 
def test_sql_query_async_uses_asyncpg(monkeypatch): 
    class FakeConn: 
        async def fetch(self, sql): 
            return [{'product_name': 'demo', 'sales': 390429}] 
        async def close(self): 
            return None 
 
    class FakeLLMResponse: 
        def __init__(self, content): 
            self.content = content 
 
    class FakeLLM: 
        async def ainvoke(self, prompt): 
            return FakeLLMResponse('async sql summary') 
 
    async def fake_connect(_url): 
        return FakeConn() 
    async def fake_generate_sql_with_examples_async(_question): 
        return 'SELECT product_name, sales FROM sales LIMIT 1;' 
 
    monkeypatch.setattr('tools.postgres_sql_tool.asyncpg.connect', fake_connect) 
    monkeypatch.setattr('tools.postgres_sql_tool.generate_sql_with_examples_async', fake_generate_sql_with_examples_async) 
    monkeypatch.setattr('tools.postgres_sql_tool.get_sql_llm', lambda: FakeLLM()) 
    result = asyncio.run(sql_query_async('top sales row')) 
    assert result['summary'] == 'async sql summary' 
    assert result['count'] == 1  
 
 
def test_redis_persistent_cache_async_round_trip(): 
    class FakeAsyncPipeline: 
        def __init__(self, store): 
            self.store = store 
        async def set(self, key, value, ex=None): 
            self.store[key] = value 
            return self 
        async def hset(self, key, mapping): 
            self.store[key] = mapping 
            return self 
        async def expire(self, key, ttl): 
            return self 
        async def execute(self): 
            return [] 
 
    class FakeAsyncRedisClient: 
        def __init__(self): 
            self.store = {} 
        async def get(self, key): 
            return self.store.get(key) 
        async def hincrby(self, key, field, amount): 
            meta = self.store.setdefault(key, {}) 
            meta[field] = int(meta.get(field, 0)) + amount 
        def pipeline(self): 
            return FakeAsyncPipeline(self.store) 
        async def keys(self, pattern): 
            prefix = pattern[:-1] 
            return [key for key in self.store if key.startswith(prefix)] 
        async def delete(self, *keys): 
            for key in keys: 
                self.store.pop(key, None) 
 
    cache = RedisPersistentCache.__new__(RedisPersistentCache) 
    cache.backend_name = 'redis' 
    cache.default_ttl = 60 
    cache.prefix = 'persistent_cache:' 
    cache.client = None 
    cache.async_client = FakeAsyncRedisClient() 
    cache.stats = CacheStats() 
    asyncio.run(cache.aset('demo', {'value': 'ok'}, ttl=30)) 
    result = asyncio.run(cache.aget('demo')) 
    asyncio.run(cache.aclear()) 
