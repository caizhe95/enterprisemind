-- init_db.sql - 1000条模拟数据版本
-- 清空旧数据（如需重置）
-- TRUNCATE TABLE sales, products RESTART IDENTITY;

-- 创建产品表
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10,2),
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建销售表
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    sale_date DATE NOT NULL,
    region VARCHAR(20),
    salesperson VARCHAR(50),
    status VARCHAR(20) DEFAULT 'completed'
);

-- 插入产品数据
INSERT INTO products (name, category, price, stock_quantity) VALUES
('智能手表Pro', '电子产品', 1299.00, 500),
('无线耳机Max', '电子产品', 899.00, 800),
('人体工学椅', '办公家具', 1599.00, 200),
('机械键盘', '电子产品', 499.00, 1000),
('4K显示器', '电子产品', 2199.00, 300),
('智能手环Lite', '电子产品', 299.00, 1500),
('降噪耳机Pro', '电子产品', 1299.00, 600),
('升降办公桌', '办公家具', 2499.00, 150),
('曲面显示器', '电子产品', 1899.00, 400),
('机械鼠标', '电子产品', 299.00, 800),
('平板电脑支架', '办公配件', 159.00, 2000),
('USB扩展坞', '电子配件', 399.00, 1200),
('无线充电器', '电子配件', 199.00, 1500),
('蓝牙音箱', '电子产品', 599.00, 700),
('智能台灯', '智能家居', 349.00, 900)
ON CONFLICT DO NOTHING;

-- 生成1000条销售记录（使用递归CTE）
WITH RECURSIVE
sales_data AS (
    SELECT
        generate_series(1, 1000) as id,
        (random() * 14 + 1)::int as product_id,  -- 15个产品
        (random() * 50 + 1)::int as quantity,
        CURRENT_DATE - (random() * 365)::int as sale_date,
        (ARRAY['华东', '华北', '华南', '西部', '华中'])[floor(random() * 5 + 1)] as region,
        (ARRAY['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'])[floor(random() * 8 + 1)] as salesperson
)
INSERT INTO sales (product_id, quantity, amount, sale_date, region, salesperson)
SELECT
    s.product_id,
    s.quantity,
    p.price * s.quantity as amount,
    s.sale_date,
    s.region,
    s.salesperson
FROM sales_data s
JOIN products p ON s.product_id = p.id
WHERE NOT EXISTS (SELECT 1 FROM sales WHERE id = s.id);

-- 创建索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region);
CREATE INDEX IF NOT EXISTS idx_sales_person ON sales(salesperson);
CREATE INDEX IF NOT EXISTS idx_sales_product ON sales(product_id);

-- 验证数据
SELECT '产品数量' as metric, COUNT(*) as value FROM products
UNION ALL
SELECT '销售记录数', COUNT(*) FROM sales
UNION ALL
SELECT '总销售额', ROUND(SUM(amount)::numeric, 2) FROM sales
UNION ALL
SELECT '销售员人数', COUNT(DISTINCT salesperson) FROM sales
UNION ALL
SELECT '覆盖地区数', COUNT(DISTINCT region) FROM sales;