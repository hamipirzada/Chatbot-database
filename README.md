# DNS Chatbot - Intelligent Database Analytics

An intelligent chatbot interface for retail/inventory database analytics powered by Large Language Models. Transform natural language queries into SQL insights with advanced analytics capabilities including ABC-XYZ analysis, forecasting, and interactive visualizations.

## 🚀 Features

### 🤖 Natural Language Processing
- **Intelligent Query Understanding**: Converts natural language questions into optimized SQL queries
- **Context-Aware Responses**: Maintains conversation history for better understanding
- **Multi-Intent Recognition**: Handles complex queries with multiple conditions and filters

### 📊 Advanced Analytics
- **ABC Analysis**: Pareto analysis for inventory classification (A: 80%, B: 15%, C: 5%)
- **XYZ Analysis**: Demand variability analysis using coefficient of variation
- **ABC-XYZ Combined**: Comprehensive inventory categorization matrix
- **Time Series Forecasting**: ARIMA and Linear Regression models for predictions
- **Sales Analytics**: Revenue, quantity, and performance metrics

### 📈 Interactive Visualizations
- **Multiple Chart Types**: Bar, line, pie, scatter, histogram, and heatmap charts
- **Dynamic Customization**: Real-time chart type switching and axis selection
- **Plotly Integration**: Interactive, responsive visualizations
- **Export Capabilities**: PDF and Excel export with embedded charts

### 🔧 Smart Query Processing
- **Auto-Correction**: Intelligent SQL error detection and correction
- **Case-Insensitive Matching**: Robust text comparison handling
- **Query Optimization**: Performance-optimized database queries
- **Retry Logic**: Automatic query refinement for better results

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Database**: PostgreSQL
- **AI/ML**: Groq API (Llama 3.3-70B)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Statsmodels

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Groq API key for LLM integration

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/hamipirzada/Chatbot-database.git
cd Chatbot-database
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:

```env
# Database Configuration
PGDATABASE=your_database_name
PGUSER=your_username
PGPASSWORD=your_password
PGHOST=localhost
PGPORT=5432

# AI Configuration
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Database Setup
Ensure your PostgreSQL database contains the required tables:
- `pos_order` - Point of sale orders
- `pos_order_line` - Order line items
- `pos_payment` - Payment records
- `product_template` - Product information
- `product_category` - Product categories
- `stock_warehouse` - Warehouse locations
- And more (see Database Schema section)

### 5. Run the Application
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

## 🗄️ Database Schema

The chatbot works with a comprehensive retail/inventory database schema including:

### Core Tables
- **Sales**: `pos_order`, `pos_order_line`, `pos_payment`
- **Products**: `product_product`, `product_template`, `product_category`
- **Inventory**: `stock_quant`, `stock_move`, `stock_valuation_layer`
- **Locations**: `stock_warehouse`, `stock_location`
- **Partners**: `res_partner` (customers/vendors)
- **Purchasing**: `purchase_order`, `purchase_order_line`

### Key Relationships
- Orders → Order Lines → Products
- Products → Categories → Warehouses
- Payments → Orders → Locations

## 💬 Usage Examples

### Basic Queries
```
"Show me total sales for 2024"
"What are the top 10 selling products?"
"Sales by warehouse for XYZ location"
```

### Advanced Analytics
```
"Perform ABC analysis on products by sales volume"
"Show XYZ analysis for demand variability"
"Forecast sales for the next 3 months"
"ABC-XYZ analysis for inventory optimization"
```

### Complex Filters
```
"Sales for ABC category in XYZ warehouse for 2024"
"Top vendors by revenue in the electronics category"
"Payment method analysis for cash vs card transactions"
```

## 🎯 Key Features in Detail

### ABC-XYZ Analysis
- **ABC Classification**: Based on revenue/quantity contribution
- **XYZ Classification**: Based on demand predictability (CV thresholds)
- **Combined Matrix**: Strategic inventory management insights
- **Visual Reports**: Interactive charts and summary tables

### Forecasting Engine
- **ARIMA Models**: Advanced time series forecasting
- **Linear Regression**: Trend-based predictions
- **Automatic Method Selection**: Best-fit model determination
- **Confidence Intervals**: Prediction uncertainty visualization

### Intelligent Query Processing
- **Intent Recognition**: Understands query purpose and entities
- **Entity Extraction**: Identifies products, locations, time periods
- **Query Optimization**: Performance-tuned SQL generation
- **Error Recovery**: Automatic correction and retry mechanisms

## 📊 Export & Reporting

### Data Export
- **Excel Export**: Formatted spreadsheets with data
- **PDF Reports**: Professional reports with charts and analysis
- **Interactive Downloads**: One-click export functionality

### Visualization Options
- **Real-time Charts**: Instant visualization updates
- **Multiple Formats**: Bar, line, pie, scatter, histogram
- **Customizable Views**: User-selectable axes and groupings
- **Export Charts**: Save visualizations in various formats

## 🔧 Configuration

### Visualization Settings
- Chart type selection (sidebar)
- Axis customization
- Color scheme options
- Export format preferences

### AI Model Settings
- Model selection (Groq API)
- Temperature and token limits
- Timeout configurations
- Retry logic parameters

## 🚨 Troubleshooting

### Common Issues

**Database Connection Errors**
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Verify connection parameters in .env
PGHOST=localhost
PGPORT=5432
```

**API Rate Limits**
```bash
# Monitor Groq API usage
# Implement request throttling if needed
```

**Memory Issues with Large Datasets**
```bash
# Optimize queries with LIMIT clauses
# Use pagination for large result sets
```

### Performance Optimization
- Enable database indexing on frequently queried columns
- Use connection pooling for high-traffic scenarios
- Implement caching for repeated queries
- Monitor and optimize slow SQL queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for public methods
- Write unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq**: For providing the LLM API
- **Streamlit**: For the amazing web framework
- **PostgreSQL**: For robust database support
- **Plotly**: For interactive visualizations

## 📞 Support

For support, questions, or feature requests:
- Create an issue on [GitHub](https://github.com/hamipirzada/Chatbot-database/issues)

## 🔮 Roadmap

- [ ] Multi-database support (MySQL, SQLite)
- [ ] Real-time dashboard features
- [ ] Advanced ML models integration
- [ ] Multi-language support
- [ ] API endpoint creation
- [ ] Docker containerization
- [ ] Cloud deployment guides

---

**Made with ❤️ for intelligent data analytics**
