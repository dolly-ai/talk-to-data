-- PostgreSQL version of database schema
-- Use this for Render.com deployment

-- Create sales table
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    quantity INT,
    price DECIMAL(10, 2),
    sale_date DATE,
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    city VARCHAR(50),
    country VARCHAR(50),
    total_purchases INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample sales data
INSERT INTO sales (product_name, category, quantity, price, sale_date, region) VALUES
('Laptop Pro', 'Electronics', 5, 1200.00, '2024-01-15', 'North'),
('Wireless Mouse', 'Electronics', 20, 25.00, '2024-01-16', 'South'),
('Office Chair', 'Furniture', 10, 250.00, '2024-01-17', 'East'),
('Desk Lamp', 'Furniture', 15, 45.00, '2024-01-18', 'West'),
('USB Cable', 'Electronics', 50, 8.00, '2024-01-19', 'North'),
('Monitor 27"', 'Electronics', 8, 350.00, '2024-01-20', 'South'),
('Standing Desk', 'Furniture', 6, 480.00, '2024-01-21', 'East'),
('Keyboard Mechanical', 'Electronics', 12, 89.00, '2024-01-22', 'West'),
('Bookshelf', 'Furniture', 7, 120.00, '2024-01-23', 'North'),
('Webcam HD', 'Electronics', 18, 75.00, '2024-01-24', 'South'),
('Coffee Table', 'Furniture', 9, 180.00, '2024-01-25', 'East'),
('Headphones', 'Electronics', 25, 65.00, '2024-01-26', 'West'),
('File Cabinet', 'Furniture', 5, 200.00, '2024-01-27', 'North'),
('Tablet 10"', 'Electronics', 11, 299.00, '2024-01-28', 'South'),
('Ergonomic Chair', 'Furniture', 8, 320.00, '2024-01-29', 'East'),
('Smartphone', 'Electronics', 15, 699.00, '2024-02-01', 'West'),
('Dining Table', 'Furniture', 4, 550.00, '2024-02-02', 'North'),
('Printer', 'Electronics', 9, 180.00, '2024-02-03', 'South'),
('Sofa 3-Seater', 'Furniture', 3, 850.00, '2024-02-04', 'East'),
('Smart Watch', 'Electronics', 22, 250.00, '2024-02-05', 'West');

-- Insert sample customers data
INSERT INTO customers (name, email, city, country, total_purchases) VALUES
('John Doe', 'john.doe@email.com', 'New York', 'USA', 5),
('Jane Smith', 'jane.smith@email.com', 'London', 'UK', 8),
('Ahmed Hassan', 'ahmed.hassan@email.com', 'Dubai', 'UAE', 3),
('Maria Garcia', 'maria.garcia@email.com', 'Madrid', 'Spain', 12),
('Chen Wei', 'chen.wei@email.com', 'Shanghai', 'China', 7),
('Sophie Martin', 'sophie.martin@email.com', 'Paris', 'France', 4),
('Raj Kumar', 'raj.kumar@email.com', 'Mumbai', 'India', 9),
('Emma Wilson', 'emma.wilson@email.com', 'Sydney', 'Australia', 6),
('Carlos Silva', 'carlos.silva@email.com', 'São Paulo', 'Brazil', 11),
('Yuki Tanaka', 'yuki.tanaka@email.com', 'Tokyo', 'Japan', 5);

-- Verify data
SELECT 'Sales table:' as info, COUNT(*) as count FROM sales
UNION ALL
SELECT 'Customers table:' as info, COUNT(*) as count FROM customers;
