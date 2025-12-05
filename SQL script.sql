-- =============================================
-- 1. DATABASE & TABLE CREATION
-- =============================================
CREATE DATABASE IF NOT EXISTS SupplyChainDB;
USE SupplyChainDB;

DROP TABLE IF EXISTS supply_chain;
CREATE TABLE supply_chain (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_type VARCHAR(50),
    sku VARCHAR(20),
    price DECIMAL(10,2),
    availability INT,
    number_of_products_sold INT,
    revenue_generated DECIMAL(12,2),
    customer_demographics VARCHAR(20),
    stock_levels INT,
    lead_times INT, -- note: sometimes duplicated as "lead time"
    order_quantities INT,
    shipping_times INT,
    shipping_carriers VARCHAR(20),
    shipping_costs DECIMAL(10,2),
    supplier_name VARCHAR(50),
    location VARCHAR(50),
    lead_time INT, -- main target column
    production_volumes INT,
    manufacturing_lead_time INT,
    manufacturing_costs DECIMAL(12,2),
    inspection_results VARCHAR(20),
    defect_rates DECIMAL(5,2),
    transportation_modes VARCHAR(20),
    routes VARCHAR(20),
    costs DECIMAL(12,2)
);

-- =============================================
-- 2. BASIC DATA EXPLORATION
-- =============================================
SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT sku) AS unique_skus,
    COUNT(DISTINCT product_type) AS unique_product_types,
    COUNT(DISTINCT location) AS unique_locations,
    COUNT(DISTINCT supplier_name) AS unique_suppliers
FROM supply_chain;

SELECT 
    product_type,
    COUNT(*) AS count,
    AVG(lead_time) AS avg_lead_time,
    STDDEV(lead_time) AS std_lead_time
FROM supply_chain
GROUP BY product_type
ORDER BY avg_lead_time DESC;

-- =============================================
-- 3. DATA QUALITY CHECK
-- =============================================
SELECT 
    SUM(CASE WHEN lead_time IS NULL THEN 1 ELSE 0 END) AS missing_lead_time,
    SUM(CASE WHEN defect_rates IS NULL THEN 1 ELSE 0 END) AS missing_defect_rates,
    SUM(CASE WHEN production_volumes IS NULL THEN 1 ELSE 0 END) AS missing_prod_vol
FROM supply_chain;

SELECT 
    MIN(lead_time) AS min_lead_time,
    MAX(lead_time) AS max_lead_time,
    AVG(lead_time) AS avg_lead_time,
    STDDEV(lead_time) AS std_lead_time
FROM supply_chain;

-- =============================================
-- 4. LEAD TIME DISTRIBUTION ANALYSIS
-- =============================================
SELECT 
    FLOOR(lead_time/5)*5 AS lead_time_bin,
    COUNT(*) AS frequency
FROM supply_chain
GROUP BY lead_time_bin
ORDER BY lead_time_bin;

-- =============================================
-- 5. CORRELATION ANALYSIS (NUMERICAL FEATURES)
-- =============================================
-- Calculate Pearson correlation between lead_time and other numeric features
SELECT 
    (COUNT(*) * SUM(lead_time * defect_rates) - SUM(lead_time) * SUM(defect_rates)) 
    / (SQRT((COUNT(*) * SUM(lead_time*lead_time) - POW(SUM(lead_time),2)) 
    * (COUNT(*) * SUM(defect_rates*defect_rates) - POW(SUM(defect_rates),2)))) 
    AS corr_defect_rates,
    
    (COUNT(*) * SUM(lead_time * production_volumes) - SUM(lead_time) * SUM(production_volumes)) 
    / (SQRT((COUNT(*) * SUM(lead_time*lead_time) - POW(SUM(lead_time),2)) 
    * (COUNT(*) * SUM(production_volumes*production_volumes) - POW(SUM(production_volumes),2)))) 
    AS corr_production_volumes,
    
    (COUNT(*) * SUM(lead_time * price) - SUM(lead_time) * SUM(price)) 
    / (SQRT((COUNT(*) * SUM(lead_time*lead_time) - POW(SUM(lead_time),2)) 
    * (COUNT(*) * SUM(price*price) - POW(SUM(price),2)))) 
    AS corr_price
FROM supply_chain;

-- =============================================
-- 6. LEAD TIME BY CATEGORICAL GROUPS
-- =============================================
-- By Product Type
SELECT 
    product_type,
    COUNT(*) AS record_count,
    AVG(lead_time) AS avg_lead_time,
    MIN(lead_time) AS min_lead_time,
    MAX(lead_time) AS max_lead_time,
    STDDEV(lead_time) AS std_lead_time
FROM supply_chain
GROUP BY product_type
ORDER BY avg_lead_time DESC;

-- By Location
SELECT 
    location,
    COUNT(*) AS record_count,
    AVG(lead_time) AS avg_lead_time
FROM supply_chain
GROUP BY location
ORDER BY avg_lead_time DESC;

-- By Transportation Mode
SELECT 
    transportation_modes,
    COUNT(*) AS record_count,
    AVG(lead_time) AS avg_lead_time
FROM supply_chain
GROUP BY transportation_modes
ORDER BY avg_lead_time;

-- =============================================
-- 7. TARGET ENCODING FOR MACHINE LEARNING
-- =============================================
-- Create a view with mean lead time by product_type
CREATE OR REPLACE VIEW product_type_encoding AS
SELECT 
    product_type,
    AVG(lead_time) AS product_type_lead_time_mean
FROM supply_chain
GROUP BY product_type;

-- Create a view with mean lead time by location
CREATE OR REPLACE VIEW location_encoding AS
SELECT 
    location,
    AVG(lead_time) AS location_lead_time_mean
FROM supply_chain
GROUP BY location;

-- Create a view with mean lead time by SKU
CREATE OR REPLACE VIEW sku_encoding AS
SELECT 
    sku,
    AVG(lead_time) AS sku_lead_time_mean
FROM supply_chain
GROUP BY sku;

-- =============================================
-- 8. JOIN ENCODINGS BACK TO MAIN TABLE
-- =============================================
CREATE OR REPLACE VIEW supply_chain_encoded AS
SELECT 
    sc.*,
    pte.product_type_lead_time_mean,
    le.location_lead_time_mean,
    se.sku_lead_time_mean
FROM supply_chain sc
LEFT JOIN product_type_encoding pte ON sc.product_type = pte.product_type
LEFT JOIN location_encoding le ON sc.location = le.location
LEFT JOIN sku_encoding se ON sc.sku = se.sku;

-- =============================================
-- 9. HIGH-LEAD TIME IDENTIFICATION
-- =============================================
-- Identify SKUs with lead time > 20 days
SELECT 
    sku,
    product_type,
    lead_time,
    defect_rates,
    production_volumes,
    price
FROM supply_chain
WHERE lead_time > 20
ORDER BY lead_time DESC;

-- =============================================
-- 10. SUPPLIER PERFORMANCE ANALYSIS
-- =============================================
SELECT 
    supplier_name,
    COUNT(*) AS total_orders,
    AVG(lead_time) AS avg_lead_time,
    AVG(defect_rates) AS avg_defect_rate,
    AVG(production_volumes) AS avg_production_volume
FROM supply_chain
GROUP BY supplier_name
HAVING COUNT(*) > 5
ORDER BY avg_lead_time DESC;

-- =============================================
-- 11. TRANSPORTATION EFFICIENCY
-- =============================================
SELECT 
    transportation_modes,
    routes,
    COUNT(*) AS shipments,
    AVG(lead_time) AS avg_lead_time,
    AVG(shipping_costs) AS avg_shipping_cost
FROM supply_chain
GROUP BY transportation_modes, routes
ORDER BY avg_lead_time;

-- =============================================
-- 12. CUSTOMER DEMOGRAPHICS ANALYSIS
-- =============================================
SELECT 
    customer_demographics,
    COUNT(*) AS order_count,
    AVG(lead_time) AS avg_lead_time,
    AVG(order_quantities) AS avg_order_quantity
FROM supply_chain
GROUP BY customer_demographics
ORDER BY avg_lead_time DESC;

-- =============================================
-- 13. DEFECT RATE ANALYSIS
-- =============================================
-- High defect rate items
SELECT 
    sku,
    product_type,
    defect_rates,
    lead_time,
    inspection_results
FROM supply_chain
WHERE defect_rates > 2.0
ORDER BY defect_rates DESC;

-- =============================================
-- 14. PRODUCTION VOLUME BOTTLENECKS
-- =============================================
SELECT 
    CASE 
        WHEN production_volumes < 300 THEN 'Low Volume'
        WHEN production_volumes BETWEEN 300 AND 700 THEN 'Medium Volume'
        WHEN production_volumes > 700 THEN 'High Volume'
    END AS volume_category,
    COUNT(*) AS record_count,
    AVG(lead_time) AS avg_lead_time,
    AVG(defect_rates) AS avg_defect_rate
FROM supply_chain
GROUP BY volume_category
ORDER BY avg_lead_time DESC;

-- =============================================
-- 15. INTERACTION EFFECTS: PRODUCT TYPE × LOCATION
-- =============================================
SELECT 
    product_type,
    location,
    COUNT(*) AS record_count,
    AVG(lead_time) AS avg_lead_time,
    AVG(defect_rates) AS avg_defect_rate
FROM supply_chain
GROUP BY product_type, location
ORDER BY product_type, avg_lead_time DESC;

-- =============================================
-- 16. PREPARATION FOR CLUSTERING ANALYSIS
-- =============================================
-- Create a view with normalized features for clustering
CREATE OR REPLACE VIEW clustering_features AS
SELECT 
    sku,
    (defect_rates - (SELECT AVG(defect_rates) FROM supply_chain)) 
    / (SELECT STDDEV(defect_rates) FROM supply_chain) AS defect_rate_z,
    
    (production_volumes - (SELECT AVG(production_volumes) FROM supply_chain)) 
    / (SELECT STDDEV(production_volumes) FROM supply_chain) AS production_volume_z,
    
    (price - (SELECT AVG(price) FROM supply_chain)) 
    / (SELECT STDDEV(price) FROM supply_chain) AS price_z,
    
    (shipping_costs - (SELECT AVG(shipping_costs) FROM supply_chain)) 
    / (SELECT STDDEV(shipping_costs) FROM supply_chain) AS shipping_cost_z,
    
    lead_time
FROM supply_chain;

-- =============================================
-- 17. BOTTLENECK IDENTIFICATION QUERY
-- =============================================
SELECT 
    sku,
    product_type,
    location,
    lead_time,
    defect_rates,
    production_volumes,
    CASE 
        WHEN defect_rates > 2.0 AND production_volumes > 700 THEN 'High Defect & High Volume'
        WHEN defect_rates > 2.0 THEN 'High Defect'
        WHEN production_volumes > 700 THEN 'High Volume'
        WHEN price > 60 AND lead_time > 20 THEN 'Premium & Delayed'
        ELSE 'Normal'
    END AS bottleneck_category
FROM supply_chain
ORDER BY lead_time DESC;

-- =============================================
-- 18. MONTHLY/PERIODIC PERFORMANCE SNAPSHOT
-- =============================================
-- (Assuming a date column existed - this is hypothetical)
-- SELECT 
--     DATE_FORMAT(order_date, '%Y-%m') AS month,
--     product_type,
--     COUNT(*) AS total_orders,
--     AVG(lead_time) AS avg_lead_time,
--     AVG(defect_rates) AS avg_defect_rate
-- FROM supply_chain
-- GROUP BY DATE_FORMAT(order_date, '%Y-%m'), product_type
-- ORDER BY month DESC, avg_lead_time DESC;

-- =============================================
-- 19. RECOMMENDATION-READY SUMMARY
-- =============================================
SELECT 
    'Top 5 Longest Lead Time SKUs' AS insight_type,
    GROUP_CONCAT(sku ORDER BY lead_time DESC LIMIT 5) AS details
FROM supply_chain

UNION ALL

SELECT 
    'Locations Needing Improvement' AS insight_type,
    GROUP_CONCAT(location ORDER BY avg_lead DESC) AS details
FROM (
    SELECT location, AVG(lead_time) AS avg_lead
    FROM supply_chain
    GROUP BY location
    HAVING AVG(lead_time) > 18
) AS problematic_locations

UNION ALL

SELECT 
    'High Defect Suppliers' AS insight_type,
    GROUP_CONCAT(supplier_name ORDER BY avg_defect DESC) AS details
FROM (
    SELECT supplier_name, AVG(defect_rates) AS avg_defect
    FROM supply_chain
    GROUP BY supplier_name
    HAVING AVG(defect_rates) > 2.5
) AS high_defect_suppliers;

-- =============================================
-- 20. EXPORT FOR MACHINE LEARNING
-- =============================================
-- Create a flat table ready for Python/ML processing
CREATE OR REPLACE VIEW ml_ready_data AS
SELECT 
    sc.sku,
    sc.product_type,
    sc.price,
    sc.defect_rates,
    sc.production_volumes,
    sc.manufacturing_costs,
    sc.shipping_costs,
    sc.lead_time,
    pte.product_type_lead_time_mean,
    le.location_lead_time_mean,
    se.sku_lead_time_mean,
    CASE sc.transportation_modes 
        WHEN 'Air' THEN 1 
        WHEN 'Road' THEN 2 
        WHEN 'Sea' THEN 3 
        WHEN 'Rail' THEN 4 
        ELSE 0 
    END AS transport_mode_code,
    CASE sc.inspection_results 
        WHEN 'Pass' THEN 1 
        WHEN 'Fail' THEN 0 
        ELSE 0.5 
    END AS inspection_score
FROM supply_chain sc
LEFT JOIN product_type_encoding pte ON sc.product_type = pte.product_type
LEFT JOIN location_encoding le ON sc.location = le.location
LEFT JOIN sku_encoding se ON sc.sku = se.sku;

-- =============================================
-- END OF SQL SCRIPT
-- =============================================