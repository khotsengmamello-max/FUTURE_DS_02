import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Gold color palette
GOLD = '#f5a623'
DARK_GOLD = '#c98a1a'

# ============================================
# 1. LOAD AND CLEAN DATA
# ============================================
print("Loading Telco Customer Churn data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# Data cleaning
print("\n--- Data Cleaning ---")

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Convert TotalCharges to numeric (some are empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing TotalCharges
df = df.dropna(subset=['TotalCharges'])

# Convert SeniorCitizen to categorical
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# Create churn binary column (1 for churned, 0 for not)
df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Save cleaned data
df.to_csv('Telco_Customer_Churn_CLEANED.csv', index=False)
print(" Cleaned data saved to 'Telco_Customer_Churn_CLEANED.csv'")

# ============================================
# 2. OVERALL CHURN RATE
# ============================================
print("\n" + "="*50)
print("OVERALL CHURN RATE")
print("="*50)

churn_rate = df['Churn_Binary'].mean() * 100
print(f"Overall Churn Rate: {churn_rate:.2f}%")
print(f"Customers who churned: {df['Churn_Binary'].sum():,}")
print(f"Customers who stayed: {(len(df) - df['Churn_Binary'].sum()):,}")
print(f"Total customers: {len(df):,}")

# ============================================
# 3. CHURN BY CATEGORY
# ============================================
print("\n" + "="*50)
print("CHURN BY CATEGORY")
print("="*50)

# Churn by Contract Type
print("\n1. Churn by Contract Type:")
contract_churn = df.groupby('Contract')['Churn_Binary'].mean() * 100
print(contract_churn)

# Churn by Payment Method
print("\n2. Churn by Payment Method:")
payment_churn = df.groupby('PaymentMethod')['Churn_Binary'].mean() * 100
print(payment_churn)

# Churn by Senior Citizen
print("\n3. Churn by Senior Citizen:")
senior_churn = df.groupby('SeniorCitizen')['Churn_Binary'].mean() * 100
print(senior_churn)

# Churn by Internet Service
print("\n4. Churn by Internet Service:")
internet_churn = df.groupby('InternetService')['Churn_Binary'].mean() * 100
print(internet_churn)

# ============================================
# 4. TENURE ANALYSIS (Customer Lifetime)
# ============================================
print("\n" + "="*50)
print("TENURE ANALYSIS (Customer Lifetime)")
print("="*50)

print(f"Average tenure of all customers: {df['tenure'].mean():.1f} months")
print(f"Average tenure of churned customers: {df[df['Churn_Binary']==1]['tenure'].mean():.1f} months")
print(f"Average tenure of retained customers: {df[df['Churn_Binary']==0]['tenure'].mean():.1f} months")

# Tenure distribution
print("\nTenure distribution (months):")
print(df['tenure'].describe())

# ============================================
# 5. MONTHLY CHARGES ANALYSIS
# ============================================
print("\n" + "="*50)
print("MONTHLY CHARGES ANALYSIS")
print("="*50)

print(f"Average Monthly Charges: ${df['MonthlyCharges'].mean():.2f}")
print(f"Average Monthly Charges (Churned): ${df[df['Churn_Binary']==1]['MonthlyCharges'].mean():.2f}")
print(f"Average Monthly Charges (Retained): ${df[df['Churn_Binary']==0]['MonthlyCharges'].mean():.2f}")

# ============================================
# 6. TOTAL CHARGES ANALYSIS
# ============================================
print("\n" + "="*50)
print("TOTAL CHARGES ANALYSIS")
print("="*50)

print(f"Average Total Charges: ${df['TotalCharges'].mean():.2f}")
print(f"Average Total Charges (Churned): ${df[df['Churn_Binary']==1]['TotalCharges'].mean():.2f}")
print(f"Average Total Charges (Retained): ${df[df['Churn_Binary']==0]['TotalCharges'].mean():.2f}")

# ============================================
# 7. GENERATE CHARTS
# ============================================
print("\n--- Generating Charts ---")

# Chart 1: Churn Rate Pie Chart
plt.figure(figsize=(8, 8))
churn_counts = df['Churn'].value_counts()
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
        colors=[GOLD, '#2a2a2e'], startangle=90, explode=(0.05, 0))
plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold', color=GOLD)
plt.tight_layout()
plt.savefig('1_churn_pie.png', dpi=150)
plt.close()

# Chart 2: Churn by Contract Type
plt.figure(figsize=(10, 6))
contract_churn.plot(kind='bar', color=GOLD, edgecolor=DARK_GOLD)
plt.title('Churn Rate by Contract Type', fontsize=16, fontweight='bold', color=GOLD)
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('2_churn_by_contract.png', dpi=150)
plt.close()

# Chart 3: Churn by Payment Method
plt.figure(figsize=(12, 6))
payment_churn.plot(kind='bar', color=GOLD, edgecolor=DARK_GOLD)
plt.title('Churn Rate by Payment Method', fontsize=16, fontweight='bold', color=GOLD)
plt.xlabel('Payment Method')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('3_churn_by_payment.png', dpi=150)
plt.close()

# Chart 4: Tenure Distribution (Churned vs Retained)
plt.figure(figsize=(12, 6))
df[df['Churn_Binary']==0]['tenure'].hist(alpha=0.7, bins=20, label='Retained', color=GOLD, edgecolor=DARK_GOLD)
df[df['Churn_Binary']==1]['tenure'].hist(alpha=0.7, bins=20, label='Churned', color='#2a2a2e', edgecolor=GOLD)
plt.title('Tenure Distribution: Churned vs Retained Customers', fontsize=16, fontweight='bold', color=GOLD)
plt.xlabel('Tenure (months)')
plt.ylabel('Number of Customers')
plt.legend()
plt.tight_layout()
plt.savefig('4_tenure_distribution.png', dpi=150)
plt.close()

# Chart 5: Monthly Charges Distribution
plt.figure(figsize=(12, 6))
df[df['Churn_Binary']==0]['MonthlyCharges'].hist(alpha=0.7, bins=30, label='Retained', color=GOLD, edgecolor=DARK_GOLD)
df[df['Churn_Binary']==1]['MonthlyCharges'].hist(alpha=0.7, bins=30, label='Churned', color='#2a2a2e', edgecolor=GOLD)
plt.title('Monthly Charges Distribution: Churned vs Retained', fontsize=16, fontweight='bold', color=GOLD)
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Number of Customers')
plt.legend()
plt.tight_layout()
plt.savefig('5_monthly_charges.png', dpi=150)
plt.close()

# Chart 6: Churn by Senior Citizen
plt.figure(figsize=(8, 6))
senior_churn.plot(kind='bar', color=GOLD, edgecolor=DARK_GOLD)
plt.title('Churn Rate by Senior Citizen Status', fontsize=16, fontweight='bold', color=GOLD)
plt.xlabel('Senior Citizen')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('6_churn_by_senior.png', dpi=150)
plt.close()

# ============================================
# 8. KEY INSIGHTS & RECOMMENDATIONS
# ============================================
print("\n" + "="*50)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*50)

print("\n KEY INSIGHTS:")
print(f"1. Overall churn rate is {churn_rate:.1f}% - this is significant and needs attention")
print("2. Customers with month-to-month contracts have the highest churn rate")
print("3. Electronic check payment method has the highest churn rate")
print("4. Customers churn most within the first 12 months")
print("5. Higher monthly charges correlate with higher churn rates")

print("\n ACTIONABLE RECOMMENDATIONS:")
print("1. Offer incentives for customers to switch from month-to-month to annual contracts")
print("2. Improve onboarding experience for new customers (first 12 months are critical)")
print("3. Investigate why Electronic Check payment method has high churn - offer discounts for auto-pay")
print("4. Create loyalty programs for customers who stay beyond 12 months")
print("5. Review pricing strategy - customers with higher monthly charges are more likely to churn")

# Save summary to CSV
summary = pd.DataFrame({
    'Metric': ['Overall Churn Rate', 'Total Customers', 'Churned Customers', 'Retained Customers',
               'Avg Tenure (All)', 'Avg Tenure (Churned)', 'Avg Tenure (Retained)',
               'Avg Monthly Charges (All)', 'Avg Monthly Charges (Churned)', 'Avg Monthly Charges (Retained)'],
    'Value': [f"{churn_rate:.2f}%", f"{len(df):,}", f"{df['Churn_Binary'].sum():,}", f"{len(df) - df['Churn_Binary'].sum():,}",
              f"{df['tenure'].mean():.1f} months", f"{df[df['Churn_Binary']==1]['tenure'].mean():.1f} months",
              f"{df[df['Churn_Binary']==0]['tenure'].mean():.1f} months",
              f"${df['MonthlyCharges'].mean():.2f}", f"${df[df['Churn_Binary']==1]['MonthlyCharges'].mean():.2f}",
              f"${df[df['Churn_Binary']==0]['MonthlyCharges'].mean():.2f}"]
})
summary.to_csv('task2_summary.csv', index=False)

print("\n Task 2 Complete! Summary saved to 'task2_summary.csv'")
print("\n Generated files:")
print("  - Telco_Customer_Churn_CLEANED.csv")
print("  - 1_churn_pie.png")
print("  - 2_churn_by_contract.png")
print("  - 3_churn_by_payment.png")
print("  - 4_tenure_distribution.png")
print("  - 5_monthly_charges.png")
print("  - 6_churn_by_senior.png")
print("  - task2_summary.csv")