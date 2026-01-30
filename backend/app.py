from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from groq import Groq
import json
import pandas as pd
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

# Detect database type
DB_TYPE = os.getenv('DB_TYPE', 'mysql')  # mysql or postgresql

# Database configuration
if DB_TYPE == 'postgresql':
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'talk_to_data'),
        'port': os.getenv('DB_PORT', '5432')
    }
else:
    import pymysql
    
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'talk_to_data'),
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }

# Groq API Client
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def get_db_connection():
    """Create database connection"""
    if DB_TYPE == 'postgresql':
        return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    else:
        return pymysql.connect(**DB_CONFIG)

def get_table_schema():
    """Get database schema information"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        if DB_TYPE == 'postgresql':
            # PostgreSQL query
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            schema_info = {}
            for table in tables:
                table_name = table['table_name']
                cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                """)
                columns = cursor.fetchall()
                schema_info[table_name] = [
                    {'Field': col['column_name'], 'Type': col['data_type']} 
                    for col in columns
                ]
        else:
            # MySQL query
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            schema_info = {}
            for table in tables:
                table_name = list(table.values())[0]
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                schema_info[table_name] = columns
        
        cursor.close()
        return schema_info
    finally:
        connection.close()

def natural_language_to_sql(question, schema_info):
    """Convert natural language question to SQL using Groq"""
    
    schema_text = "Database Schema:\n"
    for table_name, columns in schema_info.items():
        schema_text += f"\nTable: {table_name}\n"
        for col in columns:
            schema_text += f"  - {col['Field']} ({col['Type']})\n"
    
    db_dialect = "PostgreSQL" if DB_TYPE == 'postgresql' else "MySQL"
    
    prompt = f"""{schema_text}

Database Type: {db_dialect}

User Question: {question}

Convert the user's question into a valid {db_dialect} query. Return ONLY the SQL query without any explanation or markdown formatting. The query should be safe and use proper {db_dialect} syntax."""

    # Using Groq with llama-3.3-70b-versatile
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1000,
    )
    
    sql_query = chat_completion.choices[0].message.content.strip()
    # Remove markdown code blocks if present
    sql_query = re.sub(r'```sql\n?|\n?```', '', sql_query).strip()
    
    return sql_query

def execute_sql_query(sql_query):
    """Execute SQL query and return results"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(sql_query)
        
        if DB_TYPE == 'postgresql':
            results = cursor.fetchall()
            # Convert RealDictRow to regular dict
            results = [dict(row) for row in results]
        else:
            results = cursor.fetchall()
        
        cursor.close()
        return results
    finally:
        connection.close()

def generate_insights(question, sql_query, results):
    """Generate insights from query results using Groq"""
    
    # Convert results to readable format
    results_text = json.dumps(results, indent=2, default=str)[:2000]  # Limit size
    
    prompt = f"""User Question: {question}

SQL Query Executed: {sql_query}

Query Results (first 2000 chars): {results_text}

Provide a clear, concise summary of the key findings from this data. Focus on:
1. Direct answer to the user's question
2. Notable patterns or insights
3. Any important numbers or trends

Keep it under 150 words."""

    # Using Groq for fast insights
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=500,
    )
    
    return chat_completion.choices[0].message.content.strip()

def prepare_chart_data(results):
    """Prepare data for frontend visualization"""
    if not results:
        return None
    
    # Convert to pandas DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Detect chart type based on data structure
    chart_config = {
        "type": "table",
        "data": results
    }
    
    if len(df.columns) == 2:
        # Likely suitable for bar/line chart
        col_types = df.dtypes
        if col_types.iloc[1] in ['int64', 'float64']:
            chart_config = {
                "type": "bar",
                "labels": df.iloc[:, 0].astype(str).tolist(),
                "values": df.iloc[:, 1].tolist(),
                "label": df.columns[1]
            }
    
    return chart_config

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "db_type": DB_TYPE})

@app.route('/api/query', methods=['POST'])
def process_query():
    """Main endpoint to process natural language queries"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Get database schema
        schema_info = get_table_schema()
        
        # Convert natural language to SQL
        sql_query = natural_language_to_sql(question, schema_info)
        
        # Execute SQL query
        results = execute_sql_query(sql_query)
        
        # Generate insights
        insights = generate_insights(question, sql_query, results)
        
        # Prepare chart data
        chart_data = prepare_chart_data(results)
        
        return jsonify({
            "question": question,
            "sql_query": sql_query,
            "results": results,
            "insights": insights,
            "chart_data": chart_data,
            "row_count": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tables', methods=['GET'])
def get_tables():
    """Get list of tables in database"""
    try:
        schema_info = get_table_schema()
        return jsonify({"tables": list(schema_info.keys())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
