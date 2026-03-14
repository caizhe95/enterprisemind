-- init_db.sql - MySQL 8 初始化脚本

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS products;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS sales (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sale_date DATE NOT NULL,
    sku VARCHAR(20) NOT NULL,
    product_name VARCHAR(120) NOT NULL,
    category VARCHAR(40) NOT NULL,
    region VARCHAR(20) NOT NULL,
    channel VARCHAR(10) NOT NULL,
    quantity INT NOT NULL,
    amount DECIMAL(14,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_sales_product FOREIGN KEY (sku) REFERENCES products(sku)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO products (sku, name, category, brand, price, launch_date, warranty, highlights)
WITH RECURSIVE seq AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 120
)
SELECT
    CASE MOD(n - 1, 8) + 1
        WHEN 1 THEN CONCAT('手机-', LPAD(n, 4, '0'))
        WHEN 2 THEN CONCAT('笔记-', LPAD(n, 4, '0'))
        WHEN 3 THEN CONCAT('平板-', LPAD(n, 4, '0'))
        WHEN 4 THEN CONCAT('智能-', LPAD(n, 4, '0'))
        WHEN 5 THEN CONCAT('蓝牙-', LPAD(n, 4, '0'))
        WHEN 6 THEN CONCAT('电视-', LPAD(n, 4, '0'))
        WHEN 7 THEN CONCAT('空调-', LPAD(n, 4, '0'))
        ELSE CONCAT('冰箱-', LPAD(n, 4, '0'))
    END AS sku,
    CONCAT(
        CASE MOD(n - 1, 8) + 1
            WHEN 1 THEN '星澜'
            WHEN 2 THEN '智核'
            WHEN 3 THEN '星澜'
            WHEN 4 THEN '云途'
            WHEN 5 THEN '锐动'
            WHEN 6 THEN '巨幕'
            WHEN 7 THEN '清风'
            ELSE '冷峰'
        END,
        CASE MOD(n - 1, 8) + 1
            WHEN 1 THEN '手机'
            WHEN 2 THEN '笔记本'
            WHEN 3 THEN '平板'
            WHEN 4 THEN '智能手表'
            WHEN 5 THEN '蓝牙耳机'
            WHEN 6 THEN '电视'
            WHEN 7 THEN '空调'
            ELSE '冰箱'
        END,
        MOD(n - 1, 18) + 1,
        '代'
    ) AS name,
    CASE MOD(n - 1, 8) + 1
        WHEN 1 THEN '手机'
        WHEN 2 THEN '笔记本'
        WHEN 3 THEN '平板'
        WHEN 4 THEN '智能手表'
        WHEN 5 THEN '蓝牙耳机'
        WHEN 6 THEN '电视'
        WHEN 7 THEN '空调'
        ELSE '冰箱'
    END AS category,
    CASE MOD(n - 1, 8) + 1
        WHEN 1 THEN '星澜'
        WHEN 2 THEN '智核'
        WHEN 3 THEN '星澜'
        WHEN 4 THEN '云途'
        WHEN 5 THEN '锐动'
        WHEN 6 THEN '巨幕'
        WHEN 7 THEN '清风'
        ELSE '冷峰'
    END AS brand,
    CAST(
        CASE MOD(n - 1, 8) + 1
            WHEN 1 THEN 1499 + MOD(n * 173, 6999 - 1499 + 1)
            WHEN 2 THEN 3599 + MOD(n * 173, 10999 - 3599 + 1)
            WHEN 3 THEN 1499 + MOD(n * 173, 5999 - 1499 + 1)
            WHEN 4 THEN 699 + MOD(n * 173, 2999 - 699 + 1)
            WHEN 5 THEN 199 + MOD(n * 173, 1999 - 199 + 1)
            WHEN 6 THEN 2299 + MOD(n * 173, 12999 - 2299 + 1)
            WHEN 7 THEN 1999 + MOD(n * 173, 7999 - 1999 + 1)
            ELSE 2499 + MOD(n * 173, 9999 - 2499 + 1)
        END AS DECIMAL(10,2)
    ) AS price,
    MAKEDATE(2023 + MOD(n - 1, 3), 1) + INTERVAL MOD(n - 1, 12) MONTH + INTERVAL MOD(n - 1, 27) DAY AS launch_date,
    '整机1年' AS warranty,
    CASE MOD(n - 1, 8) + 1
        WHEN 1 THEN '影像稳定, 电池耐用, 信号稳定'
        WHEN 2 THEN '轻薄便携, 长续航, 高性能'
        WHEN 3 THEN '学习友好, 护眼屏, 影音体验好'
        WHEN 4 THEN '健康监测, 运动识别, 防水'
        WHEN 5 THEN '降噪清晰, 通话稳定, 佩戴舒适'
        WHEN 6 THEN '高亮度, 色彩准确, 系统流畅'
        WHEN 7 THEN '节能省电, 静音运行, 制冷快速'
        ELSE '保鲜持久, 分区清晰, 低噪音'
    END AS highlights
FROM seq;

INSERT INTO sales (sale_date, sku, product_name, category, region, channel, quantity, amount)
WITH RECURSIVE seq AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 360
)
SELECT
    DATE_ADD('2024-01-01', INTERVAL MOD(n - 1, 60) DAY) AS sale_date,
    p.sku,
    p.name AS product_name,
    p.category,
    CASE MOD(n - 1, 5) + 1
        WHEN 1 THEN '华东'
        WHEN 2 THEN '华北'
        WHEN 3 THEN '华南'
        WHEN 4 THEN '西部'
        ELSE '华中'
    END AS region,
    CASE MOD(n - 1, 2) + 1
        WHEN 1 THEN '线上'
        ELSE '线下'
    END AS channel,
    MOD(n * 7, 76) + 5 AS quantity,
    CAST(p.price * (MOD(n * 7, 76) + 5) AS DECIMAL(14,2)) AS amount
FROM seq
JOIN products p
    ON p.sku = CASE MOD(n - 1, 8) + 1
        WHEN 1 THEN CONCAT('手机-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
        WHEN 2 THEN CONCAT('笔记-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
        WHEN 3 THEN CONCAT('平板-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
        WHEN 4 THEN CONCAT('智能-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
        WHEN 5 THEN CONCAT('蓝牙-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
        WHEN 6 THEN CONCAT('电视-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
        WHEN 7 THEN CONCAT('空调-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
        ELSE CONCAT('冰箱-', LPAD(MOD(n - 1, 120) + 1, 4, '0'))
    END;

CREATE INDEX idx_sales_date ON sales(sale_date);
CREATE INDEX idx_sales_region ON sales(region);
CREATE INDEX idx_sales_channel ON sales(channel);
CREATE INDEX idx_sales_sku ON sales(sku);
CREATE INDEX idx_products_category ON products(category);

SET FOREIGN_KEY_CHECKS = 1;

SELECT '产品数量' AS metric, CAST(COUNT(*) AS CHAR) AS value FROM products
UNION ALL
SELECT '销售记录数', CAST(COUNT(*) AS CHAR) FROM sales
UNION ALL
SELECT '总销售额', CAST(ROUND(SUM(amount), 2) AS CHAR) FROM sales
UNION ALL
SELECT '覆盖品类数', CAST(COUNT(DISTINCT category) AS CHAR) FROM products
UNION ALL
SELECT '覆盖地区数', CAST(COUNT(DISTINCT region) AS CHAR) FROM sales;
