#!/usr/bin/env python3
"""
Analysis of the projects_3point4k.csv dataset
Focus: Software vs Non-Software, then Software Projects (domains & build types)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
print("Loading data...")
df = pd.read_csv('projects_3point4k.csv')

print(f"\n{'='*70}")
print("DATASET OVERVIEW")
print(f"{'='*70}")
print(f"Total repositories: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# ==========================================
# PART 1: SOFTWARE VS NON-SOFTWARE ANALYSIS
# ==========================================

print(f"\n{'='*70}")
print("PART 1: SOFTWARE VS NON-SOFTWARE PROJECTS")
print(f"{'='*70}")

# Repo Type Analysis
repo_type_counts = df['repo_type'].value_counts()
print("\nRepository Types:")
for repo_type, count in repo_type_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {repo_type}: {count:,} ({pct:.1f}%)")

# Filter data
df_software = df[df['repo_type'] == 'software_project'].copy()
df_non_software = df[df['repo_type'] == 'non_software_project'].copy()

print(f"\nSoftware Projects: {len(df_software):,} ({len(df_software)/len(df)*100:.1f}%)")
print(f"Non-Software Projects: {len(df_non_software):,} ({len(df_non_software)/len(df)*100:.1f}%)")

# ==========================================
# PART 2: SOFTWARE PROJECTS ONLY
# ==========================================

print(f"\n{'='*70}")
print("PART 2: SOFTWARE PROJECTS ANALYSIS")
print(f"{'='*70}")

# Build System Analysis
print(f"\n{'='*70}")
print("BUILD SYSTEM DISTRIBUTION (Software Projects)")
print(f"{'='*70}")
build_system_counts = df_software['build_system_label'].value_counts()
print("\nBuild Systems:")
for build_sys, count in build_system_counts.items():
    if pd.notna(build_sys):
        pct = (count / len(df_software)) * 100
        print(f"  {build_sys}: {count:,} ({pct:.1f}%)")

# Build Complexity Analysis
print(f"\n{'='*70}")
print("BUILD COMPLEXITY DISTRIBUTION (Software Projects)")
print(f"{'='*70}")
complexity_counts = df_software['build_complexity'].value_counts()
print("\nBuild Complexity:")
for complexity, count in complexity_counts.items():
    if pd.notna(complexity):
        pct = (count / len(df_software)) * 100
        print(f"  {complexity}: {count:,} ({pct:.1f}%)")

# Domain Analysis (Top 20)
print(f"\n{'='*70}")
print("DOMAIN CLASSIFICATION (Software Projects - Top 20)")
print(f"{'='*70}")
domain_counts = df_software['domain'].value_counts().head(20)
print("\nPrimary Domains:")
for domain, count in domain_counts.items():
    if pd.notna(domain):
        pct = (count / len(df_software)) * 100
        print(f"  {domain}: {count:,} ({pct:.1f}%)")

# ==========================================
# VISUALIZATIONS
# ==========================================

print(f"\n{'='*70}")
print("GENERATING VISUALIZATIONS...")
print(f"{'='*70}")

# 1. Software vs Non-Software Overview
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Repo Type Pie Chart
repo_type_counts = df['repo_type'].value_counts()
axes[0].pie(repo_type_counts.values, labels=repo_type_counts.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Repository Type Distribution', fontsize=14, fontweight='bold')

# Build System Distribution (Software Projects)
build_system_counts = df_software['build_system_label'].value_counts()
axes[1].barh(range(len(build_system_counts)), build_system_counts.values, color='coral')
axes[1].set_yticks(range(len(build_system_counts)))
axes[1].set_yticklabels(build_system_counts.index)
axes[1].set_xlabel('Count')
axes[1].set_title('Build System Distribution (Software Projects)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_1_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualization_1_overview.png")

# 2. Build Types and Complexity (Software Projects)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Build Complexity Distribution
complexity_counts = df_software['build_complexity'].value_counts()
axes[0].bar(range(len(complexity_counts)), complexity_counts.values, color='lightgreen', edgecolor='black')
axes[0].set_xticks(range(len(complexity_counts)))
axes[0].set_xticklabels(complexity_counts.index, rotation=45, ha='right')
axes[0].set_ylabel('Count')
axes[0].set_title('Build Complexity Distribution (Software Projects)', fontsize=14, fontweight='bold')

# Component Count Distribution
component_counts = df_software['build_complexity_component_count'].dropna()
axes[1].hist(component_counts, bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Component Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Component Count (Software Projects)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_2_build_types.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualization_2_build_types.png")

# 3. Domain Analysis (Software Projects)
# Get top 21 domains (skip first one, show next 20)
top_domains = df_software['domain'].value_counts().head(21).index[1:]  # Skip first domain
domain_build = pd.crosstab(df_software['domain'], df_software['build_system_label'])
domain_build_top = domain_build.loc[top_domains]

# First 10 domains (2-11)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

domain_build_first10 = domain_build_top.iloc[:10]
domain_build_first10.plot(kind='barh', stacked=True, ax=axes[0], colormap='Set3')
axes[0].set_xlabel('Count')
axes[0].set_title('Build Systems by Domain (Domains 2-11)', fontsize=14, fontweight='bold')
axes[0].legend(title='Build System', bbox_to_anchor=(1.05, 1), loc='upper left')

# Next 10 domains (12-21)
domain_build_next10 = domain_build_top.iloc[10:20]
domain_build_next10.plot(kind='barh', stacked=True, ax=axes[1], colormap='Set3')
axes[1].set_xlabel('Count')
axes[1].set_title('Build Systems by Domain (Domains 12-21)', fontsize=14, fontweight='bold')
axes[1].legend(title='Build System', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('visualization_3_domains.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualization_3_domains.png")

# 4. Build Tools Analysis
fig, ax = plt.subplots(figsize=(10, 8))

# Extract build tools
build_tools_list = []
for tools in df_software['build_system_build_tools'].dropna():
    if pd.notna(tools) and tools:
        tools_clean = tools.replace('|', ',').split(',')
        build_tools_list.extend([t.strip() for t in tools_clean if t.strip()])

tool_counts = pd.Series(build_tools_list).value_counts().head(20)
ax.barh(range(len(tool_counts)), tool_counts.values, color='purple')
ax.set_yticks(range(len(tool_counts)))
ax.set_yticklabels(tool_counts.index)
ax.set_xlabel('Count')
ax.set_title('Most Used Build Tools (Software Projects)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_4_build_tools.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualization_4_build_tools.png")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE!")
print(f"{'='*70}")
print("\nGenerated files:")
print("  - visualization_1_overview.png")
print("  - visualization_2_build_types.png")
print("  - visualization_3_domains.png")
print("  - visualization_4_build_tools.png")
print("\nAll visualizations saved successfully!")

