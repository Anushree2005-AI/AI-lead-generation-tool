# =============================
# AI-Powered Lead Generation Tool (Fixed Visualization)
# =============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import warnings
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for display
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
warnings.filterwarnings('ignore')

# Enable interactive mode for matplotlib
plt.ion()

# =============================
# Step 1: Load LinkedIn Company Data
# =============================

try:
    df = pd.read_csv("LinkedIn-company-info.csv")
    print(f"Loaded {len(df)} companies from CSV")
    print(f"\nColumns found: {df.columns.tolist()}")
    
except FileNotFoundError:
    print("ERROR: LinkedIn-company-info.csv not found!")
    exit()

# =============================
# Step 2: Feature Engineering
# =============================


print("FEATURE ENGINEERING")


# 1. Has Website
if 'Website' in df.columns:
    df["HasWebsite"] = df["Website"].notnull().astype(int)
else:
    df["HasWebsite"] = 0

# 2. Company Size -> Numeric
def size_to_num(size):
    if pd.isna(size):
        return 0
    size = str(size).lower().replace("employees", "").strip()
    
    # Handle ranges like "51-200"
    if "-" in size:
        try:
            low, high = size.split("-")
            return (int(low.replace(',', '').strip()) + int(high.replace(',', '').strip())) // 2
        except:
            return 0
    
    # Handle "10000+" format
    if "+" in size:
        try:
            return int(size.replace("+", "").replace(',', '').strip())
        except:
            return 0
    
    # Handle plain numbers
    try:
        return int(size.replace(',', '').strip())
    except:
        return 0

if 'company_size' in df.columns:
    df["CompanySizeNum"] = df["company_size"].apply(size_to_num)
elif 'employees_in_linkedin' in df.columns:
    df["CompanySizeNum"] = df["employees_in_linkedin"].apply(size_to_num)
else:
    df["CompanySizeNum"] = 0

print(f"CompanySizeNum: Average company size = {df['CompanySizeNum'].mean():.0f} employees")

# 3. Followers
if 'followers' in df.columns:
    df["FollowersNum"] = pd.to_numeric(df["followers"], errors='coerce').fillna(0)
else:
    df["FollowersNum"] = 0

# 4. Description Length
if 'about' in df.columns:
    df["DescLength"] = df["about"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
else:
    df["DescLength"] = 0

# 5. Industry Relevance
high_value_industries = [
    "software", "finance", "healthcare", "ai", "technology", 
    "artificial intelligence", "financial services", "fintech",
    "saas", "cloud", "data", "analytics", "biotechnology",
    "pharmaceutical", "medical", "venture capital", "investment",
    "consulting", "professional services", "it services"
]

if 'industries' in df.columns:
    df["IndustryScore"] = df["industries"].apply(
        lambda x: 1 if pd.notna(x) and any(ind in str(x).lower() for ind in high_value_industries) else 0
    )
else:
    df["IndustryScore"] = 0

# 6. Crunchbase
if 'crunchbase_url' in df.columns:
    df["HasCrunchbase"] = df["crunchbase_url"].notnull().astype(int)
else:
    df["HasCrunchbase"] = 0

# 7. Company Age
if 'founded' in df.columns:
    current_year = 2025
    def parse_year(x):
        try:
            x = str(x).strip()
            if x.isdigit() and len(x) == 4:
                return current_year - int(x)
            return 0
        except:
            return 0
    df["CompanyAge"] = df["founded"].apply(parse_year)
else:
    df["CompanyAge"] = 0

# 8. Funding
if 'funding' in df.columns:
    df["HasFunding"] = df["funding"].notnull().astype(int)
else:
    df["HasFunding"] = 0

print("\nFeature engineering complete!")

# =============================
# Step 3: Lead Scoring
# =============================
print("LEAD SCORING")

df["LeadScore"] = (
    (df["HasWebsite"] == 1) &
    (df["CompanySizeNum"] >= 50) &
    (df["IndustryScore"] == 1) &
    (df["FollowersNum"] >= 100)
).astype(int)

print(f"Initial leads identified: {df['LeadScore'].sum()} out of {len(df)} companies")

# =============================
# Step 4: ML Model (if possible)
# =============================

feature_columns = ["HasWebsite", "CompanySizeNum", "FollowersNum", "DescLength", 
                   "IndustryScore", "HasCrunchbase", "CompanyAge", "HasFunding"]

X = df[feature_columns]
y = df["LeadScore"]

if len(df) < 20 or y.sum() == 0 or y.sum() == len(y):
    print("\nNot enough variation for ML model → using rule-based scoring only")
    df["PredictedLeadScore"] = df["LeadScore"]
    df["LeadProbability"] = df["LeadScore"].astype(float)
else:
    print("\nTraining machine learning model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\nModel Accuracy:")
    print(f"Train: {model.score(X_train, y_train):.2%} | Test: {model.score(X_test, y_test):.2%}")
    print("\nClassification Report:\n", classification_report(y_test, preds, zero_division=0))
    joblib.dump(model, "lead_model.pkl")
    df["PredictedLeadScore"] = model.predict(X)
    df["LeadProbability"] = model.predict_proba(X)[:, 1]

# =============================
# Step 5: Results
# =============================
print("RESULTS")

display_cols = ['name', 'industries', 'Website', 'company_size', 'followers']
available_display_cols = [col for col in display_cols if col in df.columns]

print("\nTOP HIGH-VALUE LEADS:")
high_value = df[df["PredictedLeadScore"] == 1].copy()
if len(high_value) > 0:
    if "LeadProbability" in high_value.columns:
        high_value = high_value.sort_values("LeadProbability", ascending=False)
    print(high_value[available_display_cols].head(20).to_string(index=False))
else:
    print("No high-value leads found")

# Save outputs
df.to_csv("scored_leads.csv", index=False)
if len(high_value) > 0:
    high_value.to_csv("high_value_leads.csv", index=False)

# =============================
# Step 6: Visualizations (FIXED)
# =============================

print("GENERATING VISUALIZATIONS")

# Create figure 1: Top Industries for High-Value Leads
if len(high_value) > 0 and 'industries' in high_value.columns:
    fig1 = plt.figure(figsize=(12, 6))
    top_industries = high_value['industries'].value_counts().head(10)
    ax1 = sns.barplot(x=top_industries.values, y=top_industries.index, palette="viridis")
    plt.title("Top 10 Industries Among High-Value Leads", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Companies", fontsize=12)
    plt.ylabel("Industry", fontsize=12)
    plt.tight_layout()
    plt.savefig("top_industries.png", dpi=300, bbox_inches='tight')
    plt.draw()
    plt.pause(0.1)

# Create figure 2: Company Size Distribution
fig2 = plt.figure(figsize=(12, 6))
ax2 = sns.histplot(df['CompanySizeNum'], bins=30, color='skyblue', edgecolor='black')
plt.title("Company Size Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Number of Employees", fontsize=12)
plt.ylabel("Number of Companies", fontsize=12)
plt.tight_layout()
plt.savefig("company_size_distribution.png", dpi=300, bbox_inches='tight')
plt.draw()
plt.pause(0.1)

# Create figure 3: Followers Distribution
fig3 = plt.figure(figsize=(12, 6))
followers_nonzero = df[df['FollowersNum'] > 0]['FollowersNum']
if len(followers_nonzero) > 0:
    ax3 = sns.histplot(followers_nonzero, bins=30, color='salmon', edgecolor='black')
    plt.yscale('log')
    plt.title("Followers Distribution (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Followers", fontsize=12)
    plt.ylabel("Number of Companies (log scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig("followers_distribution.png", dpi=300, bbox_inches='tight')
    plt.draw()
    plt.pause(0.1)

# Create figure 4: Lead Score Distribution
fig4 = plt.figure(figsize=(10, 6))
lead_counts = df['PredictedLeadScore'].value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4']
plt.bar(lead_counts.index, lead_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title("Lead Score Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Lead Score (0=Low, 1=High)", fontsize=12)
plt.ylabel("Number of Companies", fontsize=12)
plt.xticks([0, 1], ['Low Value', 'High Value'])
for i, v in enumerate(lead_counts.values):
    plt.text(i, v + max(lead_counts.values)*0.02, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("lead_score_distribution.png", dpi=300, bbox_inches='tight')
plt.draw()
plt.pause(0.1)
# Keep plots open
plt.ioff()
plt.show()