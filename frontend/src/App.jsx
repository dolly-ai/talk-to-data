import { useState } from 'react'
import axios from 'axios'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import './App.css'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

const API_URL = import.meta.env.VITE_API_URL || '/api'

function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)

  const exampleQuestions = [
    'What are the top 5 products by revenue?',
    'Show me sales by region',
    'Which category has the highest sales?',
    'How many customers are from each country?',
    'What is the average price per category?'
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await axios.post(`${API_URL}/query`, { question })
      setResponse(res.data)
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const renderChart = () => {
    if (!response?.chart_data) return null

    const { chart_data } = response

    if (chart_data.type === 'bar') {
      const chartData = {
        labels: chart_data.labels,
        datasets: [
          {
            label: chart_data.label,
            data: chart_data.values,
            backgroundColor: 'rgba(99, 102, 241, 0.8)',
            borderColor: 'rgba(99, 102, 241, 1)',
            borderWidth: 1,
          },
        ],
      }

      const options = {
        responsive: true,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: 'Query Results',
          },
        },
      }

      return (
        <div className="chart-container">
          <Bar data={chartData} options={options} />
        </div>
      )
    }

    return null
  }

  const renderTable = () => {
    if (!response?.results || response.results.length === 0) return null

    const columns = Object.keys(response.results[0])

    return (
      <div className="table-container">
        <h3>Results ({response.row_count} rows)</h3>
        <table>
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {response.results.slice(0, 50).map((row, idx) => (
              <tr key={idx}>
                {columns.map((col) => (
                  <td key={col}>{String(row[col])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {response.row_count > 50 && (
          <p className="table-note">Showing first 50 of {response.row_count} rows</p>
        )}
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>💬 Talk to Data</h1>
        <p>Ask questions about your data in natural language</p>
      </header>

      <main>
        <div className="query-section">
          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about your data..."
                disabled={loading}
              />
              <button type="submit" disabled={loading || !question.trim()}>
                {loading ? 'Processing...' : 'Ask'}
              </button>
            </div>
          </form>

          <div className="examples">
            <p>Try these examples:</p>
            <div className="example-buttons">
              {exampleQuestions.map((q, idx) => (
                <button
                  key={idx}
                  className="example-btn"
                  onClick={() => setQuestion(q)}
                  disabled={loading}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && (
          <div className="error-box">
            <strong>Error:</strong> {error}
          </div>
        )}

        {response && (
          <div className="results-section">
            <div className="sql-query">
              <h3>Generated SQL:</h3>
              <pre>{response.sql_query}</pre>
            </div>

            {response.insights && (
              <div className="insights">
                <h3>Key Insights:</h3>
                <p>{response.insights}</p>
              </div>
            )}

            {renderChart()}
            {renderTable()}
          </div>
        )}
      </main>

      <footer>
        <p>Powered by Claude AI & MySQL</p>
      </footer>
    </div>
  )
}

export default App
