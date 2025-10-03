<img width="3555" height="1754" alt="top_industries" src="https://github.com/user-attachments/assets/06e16d6c-3a10-458a-94d6-a844e7e3a3d5" />AI-Powered Lead Generation Tool 
Overview
This project is an “AI-powered lead generation tool” designed to help businesses identify high-value companies efficiently. By combining rule-based scoring and a machine learning model, the tool analyzes company data (from LinkedIn, Crunchbase, or other sources) and identifies potential high-value leads.  
The tool is also deployed as an interactive web app using Streamlit, making it accessible online without requiring Python installation.  
Features
1. Lead Scoring
- Uses both rule-based criteria and Random Forest ML model.
- Scores companies based on:
  - Presence of a website
  - Company size
  - Industry relevance
  - Social engagement (followers)
  - Funding and Crunchbase profile availability
  - Company age

2. Visualizations
- Top industries: among high-value leads  
- Company size distribution
- Followers distribution (log scale)  
- Feature importance chart from ML model  

3. Interactive Dashboard (Streamlit)
- Upload any company CSV and score leads dynamically.  
- Filter leads by:
  - Industry
  - Company size
  - Number of followers  
- View high-value leads in a data table.
- Download high-value leads as a CSV.  

4. Automation
- Pre-trained ML model is included (`lead_model.pkl`) for instant predictions.
- Automatically calculates lead probability for each company.
How It Works

1. Upload Data
   - Upload a CSV file containing company information (LinkedIn data or similar).  

2. Feature Engineering
   - Converts company size to numeric values.
   - Calculates industry relevance, followers, website presence, and other metrics.  

3. Lead Scoring
   - Rule-based scoring identifies potential high-value leads.  
   - If ML model is available, predicts lead score and probability.  

4. Visual Insights
   - Charts and tables provide actionable insights into high-value leads.  

5. Export
   - Download high-value leads CSV for direct business use.
LinkedIn-company-info CSV Columns
Your CSV file should contain (some optional, but recommended):
- name – Company name  
- Website – Company website  
- company_size/ employees_in_linkedin – Number of employees  
- industries – Industry or business category  
- followers – Social media engagement  
- crunchbase_url – Optional Crunchbase profile  
- founded – Year founded  

A Linkedln company info CSV is included as `LinkedIn-company-info.csv`.
Tech Stack
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib
  - streamlit
- Streamlit Cloud deployment for web access

Screen Shot of Visualizations

Top industries
<img width="3555" height="1754" alt="top_industries" src="https://github.com/user-attachments/assets/4629f878-349e-4638-b1de-8c426c3b07e9" />

How to Use
1. Clone the repository:
```bash
git clone https://github.com/Anushree2005-AI/AI-lead-generation-tool.git
cd ai-leadgen-tool

