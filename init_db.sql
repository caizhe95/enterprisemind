-- init_db.sql - 实习项目一致版本（对应 products.md / sales.md）
-- 如需重置：
-- DROP TABLE IF EXISTS sales;
-- DROP TABLE IF EXISTS products;

-- 产品表（与知识源 products.md 对齐）
CREATE TABLE IF NOT EXISTS products (
    sku VARCHAR(20) PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    category VARCHAR(40) NOT NULL,
    brand VARCHAR(40) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    launch_date DATE NOT NULL,
    warranty VARCHAR(30) NOT NULL,
    highlights TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 销售表（与知识源 sales.md 对齐）
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    sale_date DATE NOT NULL,
    sku VARCHAR(20) NOT NULL REFERENCES products(sku),
    product_name VARCHAR(120) NOT NULL,
    category VARCHAR(40) NOT NULL,
    region VARCHAR(20) NOT NULL,
    channel VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    amount DECIMAL(14,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 清空旧数据（按当前结构重建）
TRUNCATE TABLE sales, products RESTART IDENTITY;

-- 插入120个产品（8个品类循环）
WITH cat_cfg AS (
    SELECT * FROM (
        VALUES
        (1, '手机', '星澜', 1499, 6999),
        (2, '笔记本', '智核', 3599, 10999),
        (3, '平板', '星澜', 1499, 5999),
        (4, '智能手表', '云途', 699, 2999),
        (5, '蓝牙耳机', '锐动', 199, 1999),
        (6, '电视', '巨幕', 2299, 12999),
        (7, '空调', '清风', 1999, 7999),
        (8, '冰箱', '冷峰', 2499, 9999)
    ) AS t(idx, category, brand, min_price, max_price)
),
seq AS (
    SELECT generate_series(1, 120) AS n
)
INSERT INTO products (sku, name, category, brand, price, launch_date, warranty, highlights)
SELECT
    CASE c.category
        WHEN '手机' THEN '手机-' || lpad(s.n::text, 4, '0')
        WHEN '笔记本' THEN '笔记-' || lpad(s.n::text, 4, '0')
        WHEN '平板' THEN '平板-' || lpad(s.n::text, 4, '0')
        WHEN '智能手表' THEN '智能-' || lpad(s.n::text, 4, '0')
        WHEN '蓝牙耳机' THEN '蓝牙-' || lpad(s.n::text, 4, '0')
        WHEN '电视' THEN '电视-' || lpad(s.n::text, 4, '0')
        WHEN '空调' THEN '空调-' || lpad(s.n::text, 4, '0')
        ELSE '冰箱-' || lpad(s.n::text, 4, '0')
    END AS sku,
    c.brand || c.category || ((s.n - 1) % 18 + 1)::text || '代' AS name,
    c.category,
    c.brand,
    (c.min_price + ((s.n * 173) % (c.max_price - c.min_price + 1)))::numeric(10,2) AS price,
    make_date(2023 + ((s.n - 1) % 3), ((s.n - 1) % 12) + 1, ((s.n - 1) % 27) + 1) AS launch_date,
    '整机1年' AS warranty,
    CASE c.category
        WHEN '手机' THEN '影像稳定, 电池耐用, 信号稳定'
        WHEN '笔记本' THEN '轻薄便携, 长续航, 高性能'
        WHEN '平板' THEN '学习友好, 护眼屏, 影音体验好'
        WHEN '智能手表' THEN '健康监测, 运动识别, 防水'
        WHEN '蓝牙耳机' THEN '降噪清晰, 通话稳定, 佩戴舒适'
        WHEN '电视' THEN '高亮度, 色彩准确, 系统流畅'
        WHEN '空调' THEN '节能省电, 静音运行, 制冷快速'
        ELSE '保鲜持久, 分区清晰, 低噪音'
    END AS highlights
FROM seq s
JOIN cat_cfg c ON c.idx = ((s.n - 1) % 8) + 1;

-- 插入360条销售（与实习知识源规模一致）
WITH seq AS (
    SELECT generate_series(1, 360) AS n
),
picked AS (
    SELECT
        s.n,
        p.sku,
        p.name AS product_name,
        p.category,
        p.price,
        date '2024-01-01' + ((s.n - 1) % 60) AS sale_date,
        (ARRAY['华东', '华北', '华南', '西部', '华中'])[ ((s.n - 1) % 5) + 1 ] AS region,
        (ARRAY['线上', '线下'])[ ((s.n - 1) % 2) + 1 ] AS channel,
        ((s.n * 7) % 76 + 5) AS quantity
    FROM seq s
    JOIN products p ON p.sku = (
        SELECT sku FROM products ORDER BY sku LIMIT 1 OFFSET ((s.n - 1) % 120)
    )
)
INSERT INTO sales (sale_date, sku, product_name, category, region, channel, quantity, amount)
SELECT
    sale_date,
    sku,
    product_name,
    category,
    region,
    channel,
    quantity,
    (price * quantity)::numeric(14,2) AS amount
FROM picked;

-- 索引
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region);
CREATE INDEX IF NOT EXISTS idx_sales_channel ON sales(channel);
CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales(sku);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

-- 验证
SELECT '产品数量' AS metric, COUNT(*)::text AS value FROM products
UNION ALL
SELECT '销售记录数', COUNT(*)::text FROM sales
UNION ALL
SELECT '总销售额', ROUND(SUM(amount)::numeric, 2)::text FROM sales
UNION ALL
SELECT '覆盖品类数', COUNT(DISTINCT category)::text FROM products
UNION ALL
SELECT '覆盖地区数', COUNT(DISTINCT region)::text FROM sales;
