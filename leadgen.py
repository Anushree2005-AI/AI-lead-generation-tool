import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

st.title("Lead Generation Dashboard")
st.write("Upload your company data to find high-value prospects")

# Try loading the ML model
model = None
try:
    model = joblib.load("lead_model.pkl")
    st.success("Model loaded!")
except:
    st.warning("No trained model found - using basic scoring")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} companies")
    
    with st.expander("Preview data"):
        st.dataframe(df.head())
    
    st.subheader("Processing...")
    
    # Check if company has a website
    df["HasWebsite"] = df.get("Website", pd.Series([None]*len(df))).notnull().astype(int)
    
    # Convert company size to numbers
    def get_company_size(size):
        if pd.isna(size): 
            return 0
        
        size = str(size).lower().replace("employees", "").strip()
        
        # Handle ranges like "50-100"
        if "-" in size:
            try:
                parts = size.split("-")
                low = int(parts[0].replace(',', ''))
                high = int(parts[1].replace(',', ''))
                return (low + high) // 2
            except: 
                return 0
        
        # Handle "1000+" format
        if "+" in size:
            try: 
                return int(size.replace("+", "").replace(',', ''))
            except: 
                return 0
        
        # Plain numbers
        try: 
            return int(size.replace(',', ''))
        except: 
            return 0
    
    if 'company_size' in df.columns:
        df["CompanySizeNum"] = df["company_size"].apply(get_company_size)
    elif 'employees_in_linkedin' in df.columns:
        df["CompanySizeNum"] = df["employees_in_linkedin"].apply(get_company_size)
    else:
        df["CompanySizeNum"] = 0
    
    # Get follower count
    if 'followers' in df.columns:
        df["FollowersNum"] = pd.to_numeric(df["followers"], errors='coerce').fillna(0)
    else:
        df["FollowersNum"] = 0
    
    # Count words in description
    if 'about' in df.columns:
        df["DescLength"] = df["about"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    else:
        df["DescLength"] = 0
    
    # Check if industry is valuable
    valuable_industries = [
        "software", "finance", "healthcare", "ai", "technology",
        "artificial intelligence", "financial services", "fintech",
        "saas", "cloud", "data", "analytics", "biotechnology",
        "pharmaceutical", "medical", "venture capital", "investment",
        "consulting", "professional services", "it services"
    ]
    
    if 'industries' in df.columns:
        df["IndustryScore"] = df["industries"].apply(
            lambda x: 1 if pd.notna(x) and any(ind in str(x).lower() for ind in valuable_industries) else 0
        )
    else:
        df["IndustryScore"] = 0
    
    # Check Crunchbase presence
    if 'crunchbase_url' in df.columns:
        df["HasCrunchbase"] = df["crunchbase_url"].notnull().astype(int)
    else:
        df["HasCrunchbase"] = 0
    
    # Calculate company age
    if 'founded' in df.columns:
        def get_age(year):
            try:
                year_str = str(year).strip()
                if year_str.isdigit() and len(year_str) == 4:
                    return 2025 - int(year_str)
                return 0
            except:
                return 0
        df["CompanyAge"] = df["founded"].apply(get_age)
    else:
        df["CompanyAge"] = 0
    
    # Check funding status
    if 'funding' in df.columns:
        df["HasFunding"] = df["funding"].notnull().astype(int)
    else:
        df["HasFunding"] = 0
    
    # Simple scoring rules
    df["LeadScore"] = (
        (df["HasWebsite"] == 1) &
        (df["CompanySizeNum"] >= 50) &
        (df["IndustryScore"] == 1) &
        (df["FollowersNum"] >= 100)
    ).astype(int)
    
    st.success("Done!")
    
    # Show some stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Company Size", f"{df['CompanySizeNum'].mean():.0f}")
    c2.metric("Have Website", f"{df['HasWebsite'].sum()}")
    c3.metric("Target Industries", f"{df['IndustryScore'].sum()}")
    c4.metric("Avg Followers", f"{df['FollowersNum'].mean():.0f}")
    
    st.subheader("Scoring Leads")
    
    features = [
        "HasWebsite", "CompanySizeNum", "FollowersNum", "DescLength",
        "IndustryScore", "HasCrunchbase", "CompanyAge", "HasFunding"
    ]
    
    # Use ML model if available
    if model:
        try:
            df["PredictedLeadScore"] = model.predict(df[features])
            df["LeadProbability"] = model.predict_proba(df[features])[:, 1]
            st.success("Predictions complete!")
        except Exception as err:
            st.error(f"Model error: {err}")
            df["PredictedLeadScore"] = df["LeadScore"]
            df["LeadProbability"] = df["LeadScore"].astype(float)
    else:
        df["PredictedLeadScore"] = df["LeadScore"]
        df["LeadProbability"] = df["LeadScore"].astype(float)
    
    st.subheader("Results")
    
    prospects = df[df["PredictedLeadScore"] == 1].copy()
    
    if len(prospects) > 0:
        if "LeadProbability" in prospects.columns:
            prospects = prospects.sort_values("LeadProbability", ascending=False)
        
        percent = len(prospects)/len(df)*100
        st.info(f"Found {len(prospects)} high-value leads ({percent:.1f}% of total)")
        
        cols_to_show = ['name', 'industries', 'Website', 'company_size', 'followers', 'LeadProbability']
        available = [c for c in cols_to_show if c in prospects.columns]
        
        st.dataframe(prospects[available], width='stretch')
        
        csv_data = prospects.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download leads",
            csv_data,
            "high_value_leads.csv",
            "text/csv"
        )
    else:
        st.warning("No leads match the criteria")
    
    st.subheader("Charts")
    
    t1, t2, t3, t4 = st.tabs(["Industries", "Company Size", "Followers", "Scores"])
    
    with t1:
        if len(prospects) > 0 and 'industries' in prospects.columns:
            st.write("Top industries")
            top_10 = prospects['industries'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_10.values, y=top_10.index, hue=top_10.index, palette="viridis", ax=ax, legend=False)
            ax.set_xlabel("Companies")
            ax.set_ylabel("Industry")
            st.pyplot(fig)
        else:
            st.info("Not enough data")
    
    with t2:
        st.write("Employee count distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['CompanySizeNum'], bins=30, color='skyblue', edgecolor='black', ax=ax)
        ax.set_xlabel("Employees")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with t3:
        st.write("Follower distribution")
        has_followers = df[df['FollowersNum'] > 0]['FollowersNum']
        
        if len(has_followers) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(has_followers, bins=30, color='salmon', edgecolor='black', ax=ax)
            ax.set_yscale('log')
            ax.set_xlabel("Followers")
            ax.set_ylabel("Count (log)")
            st.pyplot(fig)
        else:
            st.info("No follower data")
    
    with t4:
        st.write("Lead quality breakdown")
        counts = df['PredictedLeadScore'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(counts.index, counts.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Quality")
        ax.set_ylabel("Count")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Standard', 'High Value'])
        
        for i, v in enumerate(counts.values):
            ax.text(i, v + max(counts.values)*0.02, str(v), ha='center', fontweight='bold')
        
        st.pyplot(fig)
    
    st.subheader("Export")
    all_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download all scored data",
        all_data,
        "all_leads_scored.csv",
        "text/csv"
    )

else:
    st.info("Upload a CSV to begin")
    
    st.markdown("""
    **Your CSV should include:**
    - Company name
    - Industry/sector
    - Website URL
    - Employee count
    - LinkedIn followers
    - Company description
    - Crunchbase URL (optional)
    - Founded year (optional)
    - Funding details (optional)
    """)