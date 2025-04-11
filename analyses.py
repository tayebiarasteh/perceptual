"""
analyses.py
Created on April 8, 2025.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""
import pdb
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM
import seaborn as sns
from itertools import combinations
from scipy.stats import f_oneway, ttest_rel, ttest_ind, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from contextlib import redirect_stdout
import matplotlib.image as mpimg



import warnings
warnings.filterwarnings('ignore')


# basedir = 'xxxxxx'



def zeroshot_turing():
    os.makedirs('zero_shot', exist_ok=True)

    with open("./zero_shot/turing_zeroshot.txt", "w") as f:
        with redirect_stdout(f):

            print('\t\tResults for Zero-shot case\n\n')

            df_zero = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyzeroshot.csv'))

            df_zero = df_zero.dropna().copy()

            numeric_cols = df_zero.columns.drop('Listener')

            # Compute column-wise mean and std (per pathology group)
            column_stats = df_zero.agg(['mean', 'std']).round(0).astype(int).T
            column_stats = column_stats.rename(columns={'mean': 'Mean (%)', 'std': 'SD (%)'})

            # Compute row-wise mean and std (per listener), excluding non-numeric columns
            df_zero['Mean (%)'] = df_zero.drop(columns=['Listener']).mean(axis=1).round(0).astype(int)
            df_zero['SD (%)'] = df_zero.drop(columns=['Listener']).std(axis=1).round(0).astype(int)

            overall_mean = int(df_zero[numeric_cols].values.flatten().mean().round(0))
            overall_std = int(df_zero[numeric_cols].values.flatten().std().round(0))

            print("zero-shot per pathology group:")
            print(column_stats)

            print("\nzero-shot per listener:")
            print(df_zero[['Listener', 'Mean (%)', 'SD (%)']])

            print(f"\nOverall Zero-shot Accuracy: {overall_mean} ± {overall_std}%")

            print(f"\nCLP Zero-shot Accuracy: {column_stats['Mean (%)'].loc['CLP']} ± {column_stats['SD (%)'].loc['CLP']}%")
            print(f"Control adults Zero-shot Accuracy: {column_stats['Mean (%)'].loc['Control adults']} ± {column_stats['SD (%)'].loc['Control adults']}%")
            print(f"Control children Zero-shot Accuracy: {column_stats['Mean (%)'].loc['Control children']} ± {column_stats['SD (%)'].loc['Control children']}%")
            print(f"Dysarthria Zero-shot Accuracy: {column_stats['Mean (%)'].loc['Dysarthria']} ± {column_stats['SD (%)'].loc['Dysarthria']}%")
            print(f"Dysglossia Zero-shot Accuracy: {column_stats['Mean (%)'].loc['Dysglossia']} ± {column_stats['SD (%)'].loc['Dysglossia']}%")
            print(f"Dysphonia Zero-shot Accuracy: {column_stats['Mean (%)'].loc['Dysphonia']} ± {column_stats['SD (%)'].loc['Dysphonia']}%\n")

            ##########################################################################################################
            # -------- Boxplot Section -------- #
            df_melted = df_zero.melt(id_vars='Listener', value_vars=numeric_cols, var_name='Pathology', value_name='Accuracy')

            # Plot with customized aesthetics
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_melted, x='Pathology', y='Accuracy', showfliers=True)

            # Font and style adjustments
            # plt.title("Turing test accuracy for zero-shot analysis", fontsize=18)
            plt.ylabel("Accuracy [%]", fontsize=16)
            plt.xlabel("Speech pathology group", fontsize=16)
            plt.xticks(rotation=30, fontsize=13)
            plt.yticks(fontsize=13)
            sns.despine()
            plt.tight_layout()

            # Save and show
            plt.savefig("./zero_shot/turing_box_zeroshot.png", dpi=300)
            # plt.show()

            ##########################################################################################################
            print('\n####################################################################################################\n')

            # -------- Repeated-Measures ANOVA -------- #
            aovrm = AnovaRM(df_melted, 'Accuracy', 'Listener', within=['Pathology'])
            anova_results = aovrm.fit()

            # Print ANOVA summary
            print("\nRepeated-measures ANOVA result:")
            print(anova_results.summary())
            print('If significant: There is a statistically significant main effect of pathology group on zero-shot Turing test accuracy. '
                  '\n In plain terms, human listeners were significantly better at identifying original vs anonymized speech for some pathologies compared to others.')

            # Extract ANOVA table
            anova_table = anova_results.anova_table

            # Get F-statistic and p-value
            f_value = anova_table['F Value'].iloc[0]
            p_value = anova_table['Pr > F'].iloc[0]

            # Get degrees of freedom
            df1 = int(anova_table['Num DF'].iloc[0])  # numerator df
            df2 = int(anova_table['Den DF'].iloc[0])  # denominator df

            # Print results
            print("\nRepeated-measures ANOVA result:")
            print(f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.3f}")

            # Smart interpretation
            if p_value < 0.05:
                interpretation = "This indicates a statistically significant effect of pathology group on Turing test accuracy."
            else:
                interpretation = "This indicates that there is no statistically significant difference in accuracy across pathology groups."

            print("\nInterpretation:")
            print(f"A repeated-measures ANOVA revealed a main effect of pathology group on discrimination accuracy, "
                  f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.3f}. \n{interpretation}")

            print('\nThe F-statistic compares variance between groups (df₁) to variance within groups (df₂).')
            print('Larger F values suggest more separation between group means.')
            print('The p-value tells you whether that separation is statistically significant.')

            ####################################################################################################
            # post-hoc t-tests with FDR
            print('\n####################################################################################################\n')
            print('\npost-hoc t-tests with FDR\n')

            # Prepare 6x6 matrix for FDR-corrected p-values
            groups = df_melted['Pathology'].unique()
            fdr_matrix = pd.DataFrame(np.ones((len(groups), len(groups))), index=groups, columns=groups)

            # Collect pairwise raw p-values
            pairwise_results_fdr = []
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    if i < j:
                        listeners = df_melted['Listener'].unique()
                        vals1 = df_melted[df_melted['Pathology'] == g1].set_index('Listener').loc[listeners]['Accuracy']
                        vals2 = df_melted[df_melted['Pathology'] == g2].set_index('Listener').loc[listeners]['Accuracy']
                        t_stat, p_val = ttest_rel(vals1, vals2)
                        pairwise_results_fdr.append((g1, g2, p_val))

            # FDR correction
            raw_pvals_fdr = [res[2] for res in pairwise_results_fdr]
            _, corrected_pvals_fdr, _, _ = multipletests(raw_pvals_fdr, method='fdr_bh')

            # Populate matrix and collect significant results
            sig_comparisons_fdr = []
            for (g1, g2, _), corr_p in zip(pairwise_results_fdr, corrected_pvals_fdr):
                fdr_matrix.loc[g1, g2] = corr_p
                fdr_matrix.loc[g2, g1] = corr_p
                if corr_p < 0.05:
                    sig_comparisons_fdr.append((g1, g2, round(corr_p, 3)))

            # Round matrix for presentation
            fdr_matrix_rounded = fdr_matrix.round(3)
            print("\nFDR-Corrected Pairwise P-values (6x6):")
            print(fdr_matrix_rounded)

            # Generate natural language summary
            if sig_comparisons_fdr:
                sig_sentences = [f"{a} vs {b} (p = {p})" for a, b, p in sig_comparisons_fdr]
                summary_sentence_fdr = "Post-hoc pairwise comparisons using FDR correction revealed statistically significant differences between: \n " + "; ".join(sig_sentences) + "."
            else:
                summary_sentence_fdr = "Post-hoc pairwise comparisons using FDR correction revealed no statistically significant differences."

            print("\n" + summary_sentence_fdr)

            fdr_csv_path = "./zero_shot/turing_posthoc_pvalues_zeroshot.csv"
            fdr_matrix_rounded.to_csv(fdr_csv_path)

            ####################################################################################################
            # Native vs Non-Native Listener Comparison
            print('\n####################################################################################################\n')
            print("\nListener language proficiency effect (native vs non-native):\n")

            # Define native and non-native listeners
            native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']
            non_native_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

            # Melt the dataframe to long form
            df_long = df_zero.melt(id_vars='Listener', value_vars=numeric_cols, var_name='Pathology', value_name='Accuracy')

            # Label listener types
            df_long['Language Group'] = df_long['Listener'].apply(lambda x: 'Native' if x in native_listeners else 'Non-native')

            # Split groups
            native_all = df_long[df_long['Language Group'] == 'Native']['Accuracy']
            non_native_all = df_long[df_long['Language Group'] == 'Non-native']['Accuracy']

            # Stats
            native_avg_all = round(native_all.mean())
            native_sd_all = round(native_all.std())
            non_native_avg_all = round(non_native_all.mean())
            non_native_sd_all = round(non_native_all.std())

            # Independent t-test
            t_stat_all, p_val_all = ttest_ind(native_all, non_native_all)

            print(f"Overall \nNative: {native_avg_all} ± {native_sd_all}% vs. Non-native: {non_native_avg_all} ± {non_native_sd_all}% (t = {t_stat_all:.2f}, p = {p_val_all:.4f})\n")

            for pathology in numeric_cols:
                native_scores = df_zero[df_zero['Listener'].isin(native_listeners)][pathology]
                non_native_scores = df_zero[df_zero['Listener'].isin(non_native_listeners)][pathology]

                native_avg = round(native_scores.mean())
                native_sd = round(native_scores.std())
                non_native_avg = round(non_native_scores.mean())
                non_native_sd = round(non_native_scores.std())

                t_stat, p_val = ttest_ind(native_scores, non_native_scores)

                print(f"{pathology}:")
                print(f"  Native: {native_avg} ± {native_sd}% vs. Non-native: {non_native_avg} ± {non_native_sd}% (t = {t_stat:.2f}, p = {p_val:.3f})\n")



def fewishot_turing():
    os.makedirs('few_shot', exist_ok=True)

    with open("./few_shot/turing_fewshot.txt", "w") as f:
        with redirect_stdout(f):

            print('\t\tResults for Few-shot case\n\n')

            df_zero = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyfewshot.csv'))

            df_zero = df_zero.dropna().copy()

            numeric_cols = df_zero.columns.drop('Listener')

            # Compute column-wise mean and std (per pathology group)
            column_stats = df_zero.agg(['mean', 'std']).round(0).astype(int).T
            column_stats = column_stats.rename(columns={'mean': 'Mean (%)', 'std': 'SD (%)'})

            # Compute row-wise mean and std (per listener), excluding non-numeric columns
            df_zero['Mean (%)'] = df_zero.drop(columns=['Listener']).mean(axis=1).round(0).astype(int)
            df_zero['SD (%)'] = df_zero.drop(columns=['Listener']).std(axis=1).round(0).astype(int)

            overall_mean = int(df_zero[numeric_cols].values.flatten().mean().round(0))
            overall_std = int(df_zero[numeric_cols].values.flatten().std().round(0))

            print("Few-shot per pathology group:")
            print(column_stats)

            print("\nFew-shot per listener:")
            print(df_zero[['Listener', 'Mean (%)', 'SD (%)']])

            print(f"\nOverall Few-shot Accuracy: {overall_mean} ± {overall_std}%")

            print(f"\nCLP Few-shot Accuracy: {column_stats['Mean (%)'].loc['CLP']} ± {column_stats['SD (%)'].loc['CLP']}%")
            print(f"Control adults Few-shot Accuracy: {column_stats['Mean (%)'].loc['Control adults']} ± {column_stats['SD (%)'].loc['Control adults']}%")
            print(f"Control children Few-shot Accuracy: {column_stats['Mean (%)'].loc['Control children']} ± {column_stats['SD (%)'].loc['Control children']}%")
            print(f"Dysarthria Few-shot Accuracy: {column_stats['Mean (%)'].loc['Dysarthria']} ± {column_stats['SD (%)'].loc['Dysarthria']}%")
            print(f"Dysglossia Few-shot Accuracy: {column_stats['Mean (%)'].loc['Dysglossia']} ± {column_stats['SD (%)'].loc['Dysglossia']}%")
            print(f"Dysphonia Few-shot Accuracy: {column_stats['Mean (%)'].loc['Dysphonia']} ± {column_stats['SD (%)'].loc['Dysphonia']}%\n")

            ##########################################################################################################
            # -------- Boxplot Section -------- #
            df_melted = df_zero.melt(id_vars='Listener', value_vars=numeric_cols, var_name='Pathology', value_name='Accuracy')

            # Plot with customized aesthetics
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_melted, x='Pathology', y='Accuracy', showfliers=True)

            # Font and style adjustments
            # plt.title("Turing test accuracy for few-shot analysis", fontsize=18)
            plt.ylabel("Accuracy [%]", fontsize=16)
            plt.xlabel("Speech pathology group", fontsize=16)
            plt.xticks(rotation=30, fontsize=13)
            plt.yticks(fontsize=13)
            sns.despine()
            plt.tight_layout()

            # Save and show
            plt.savefig("./few_shot/turing_box_fewshot.png", dpi=300)
            # plt.show()

            ##########################################################################################################
            print('\n####################################################################################################\n')

            # -------- Repeated-Measures ANOVA -------- #
            aovrm = AnovaRM(df_melted, 'Accuracy', 'Listener', within=['Pathology'])
            anova_results = aovrm.fit()

            # Print ANOVA summary
            print("\nRepeated-measures ANOVA result:")
            print(anova_results.summary())
            print('If significant: There is a statistically significant main effect of pathology group on Few-shot Turing test accuracy. '
                  '\n In plain terms, human listeners were significantly better at identifying original vs anonymized speech for some pathologies compared to others.')

            # Extract ANOVA table
            anova_table = anova_results.anova_table

            # Get F-statistic and p-value
            f_value = anova_table['F Value'].iloc[0]
            p_value = anova_table['Pr > F'].iloc[0]

            # Get degrees of freedom
            df1 = int(anova_table['Num DF'].iloc[0])  # numerator df
            df2 = int(anova_table['Den DF'].iloc[0])  # denominator df

            # Print results
            print("\nRepeated-measures ANOVA result:")
            print(f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.3f}")

            # Smart interpretation
            if p_value < 0.05:
                interpretation = "This indicates a statistically significant effect of pathology group on Turing test accuracy."
            else:
                interpretation = "This indicates that there is no statistically significant difference in accuracy across pathology groups."

            print("\nInterpretation:")
            print(f"A repeated-measures ANOVA revealed a main effect of pathology group on discrimination accuracy, "
                  f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.3f}. \n{interpretation}")

            print('\nThe F-statistic compares variance between groups (df₁) to variance within groups (df₂).')
            print('Larger F values suggest more separation between group means.')
            print('The p-value tells you whether that separation is statistically significant.')

            ####################################################################################################
            print('\n####################################################################################################\n')
            # post-hoc t-tests with FDR
            print('\npost-hoc t-tests with FDR\n')

            # Prepare 6x6 matrix for FDR-corrected p-values
            groups = df_melted['Pathology'].unique()
            fdr_matrix = pd.DataFrame(np.ones((len(groups), len(groups))), index=groups, columns=groups)

            # Collect pairwise raw p-values
            pairwise_results_fdr = []
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    if i < j:
                        listeners = df_melted['Listener'].unique()
                        vals1 = df_melted[df_melted['Pathology'] == g1].set_index('Listener').loc[listeners]['Accuracy']
                        vals2 = df_melted[df_melted['Pathology'] == g2].set_index('Listener').loc[listeners]['Accuracy']
                        t_stat, p_val = ttest_rel(vals1, vals2)
                        pairwise_results_fdr.append((g1, g2, p_val))

            # FDR correction
            raw_pvals_fdr = [res[2] for res in pairwise_results_fdr]
            _, corrected_pvals_fdr, _, _ = multipletests(raw_pvals_fdr, method='fdr_bh')

            # Populate matrix and collect significant results
            sig_comparisons_fdr = []
            for (g1, g2, _), corr_p in zip(pairwise_results_fdr, corrected_pvals_fdr):
                fdr_matrix.loc[g1, g2] = corr_p
                fdr_matrix.loc[g2, g1] = corr_p
                if corr_p < 0.05:
                    sig_comparisons_fdr.append((g1, g2, round(corr_p, 3)))

            # Round matrix for presentation
            fdr_matrix_rounded = fdr_matrix.round(3)
            print("\nFDR-Corrected Pairwise P-values (6x6):")
            print(fdr_matrix_rounded)

            # Generate natural language summary
            if sig_comparisons_fdr:
                sig_sentences = [f"{a} vs {b} (p = {p})" for a, b, p in sig_comparisons_fdr]
                summary_sentence_fdr = "Post-hoc pairwise comparisons using FDR correction revealed statistically significant differences between: \n " + "; ".join(sig_sentences) + "."
            else:
                summary_sentence_fdr = "Post-hoc pairwise comparisons using FDR correction revealed no statistically significant differences."

            print("\n" + summary_sentence_fdr)

            fdr_csv_path = "./few_shot/turing_posthoc_pvalues_fewshot.csv"
            fdr_matrix_rounded.to_csv(fdr_csv_path)
            ####################################################################################################
            # Native vs Non-Native Listener Comparison
            print('\n####################################################################################################\n')
            print("\nListener language proficiency effect (native vs non-native):\n")

            # Define native and non-native listeners
            native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']
            non_native_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

            # Melt the dataframe to long form
            df_long = df_zero.melt(id_vars='Listener', value_vars=numeric_cols, var_name='Pathology', value_name='Accuracy')

            # Label listener types
            df_long['Language Group'] = df_long['Listener'].apply(lambda x: 'Native' if x in native_listeners else 'Non-native')

            # Split groups
            native_all = df_long[df_long['Language Group'] == 'Native']['Accuracy']
            non_native_all = df_long[df_long['Language Group'] == 'Non-native']['Accuracy']

            # Stats
            native_avg_all = round(native_all.mean())
            native_sd_all = round(native_all.std())
            non_native_avg_all = round(non_native_all.mean())
            non_native_sd_all = round(non_native_all.std())

            # Independent t-test
            t_stat_all, p_val_all = ttest_ind(native_all, non_native_all)

            print(f"Overall \nNative: {native_avg_all} ± {native_sd_all}% vs. Non-native: {non_native_avg_all} ± {non_native_sd_all}% (t = {t_stat_all:.2f}, p = {p_val_all:.3f})\n")

            for pathology in numeric_cols:
                native_scores = df_zero[df_zero['Listener'].isin(native_listeners)][pathology]
                non_native_scores = df_zero[df_zero['Listener'].isin(non_native_listeners)][pathology]

                native_avg = round(native_scores.mean())
                native_sd = round(native_scores.std())
                non_native_avg = round(non_native_scores.mean())
                non_native_sd = round(non_native_scores.std())

                t_stat, p_val = ttest_ind(native_scores, non_native_scores)

                print(f"{pathology}:")
                print(f"  Native: {native_avg} ± {native_sd}% vs. Non-native: {non_native_avg} ± {non_native_sd}% (t = {t_stat:.2f}, p = {p_val:.3f})\n")



def male_vs_female_turing_fewshot():
    os.makedirs('fairness', exist_ok=True)

    df_male = pd.read_csv(os.path.join(basedir, 'male - accuracyfewshot.csv')).dropna().set_index("Listener")
    df_female = pd.read_csv(os.path.join(basedir, 'female - accuracyfewshot.csv')).dropna().set_index("Listener")

    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']

    results = []

    # Group-wise paired t-tests with mean & std
    for group in patients + controls:
        t_stat, p_val = ttest_rel(df_male[group], df_female[group])
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": round(p_val, 3)
        })

    # Compute patient and control averages
    df_male["Patient Avg"] = df_male[patients].mean(axis=1)
    df_female["Patient Avg"] = df_female[patients].mean(axis=1)
    df_male["Control Avg"] = df_male[controls].mean(axis=1)
    df_female["Control Avg"] = df_female[controls].mean(axis=1)

    # T-tests for averages
    for group in ["Patient Avg", "Control Avg"]:
        t_stat, p_val = ttest_rel(df_male[group], df_female[group])
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": round(p_val, 3)
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("./fairness/gender_fairness_fewshot.csv", index=False)




def male_vs_female_turing_zeroshot():
    os.makedirs('fairness', exist_ok=True)

    df_male = pd.read_csv(os.path.join(basedir, 'male - accuracyzeroshot.csv')).dropna().set_index("Listener")
    df_female = pd.read_csv(os.path.join(basedir, 'female - accuracyzeroshot.csv')).dropna().set_index("Listener")

    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']

    results = []

    # Individual pathology/control groups
    for group in patients + controls:
        t_stat, p_val = ttest_rel(df_male[group], df_female[group])
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": round(p_val, 3)
        })

    # Add averaged groups (Patient Avg / Control Avg)
    df_male["Patient Avg"] = df_male[patients].mean(axis=1)
    df_female["Patient Avg"] = df_female[patients].mean(axis=1)
    df_male["Control Avg"] = df_male[controls].mean(axis=1)
    df_female["Control Avg"] = df_female[controls].mean(axis=1)

    for group in ["Patient Avg", "Control Avg"]:
        t_stat, p_val = ttest_rel(df_male[group], df_female[group])
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": round(p_val, 3)
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("./fairness/gender_fairness_zeroshot.csv", index=False)




def quality():
    os.makedirs('quality', exist_ok=True)

    df_raw = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - Quality Percentage.csv'), header=None)

    headers_1 = pd.Series(df_raw.iloc[0, 1:]).fillna(method='ffill').astype(str)
    headers_2 = pd.Series(df_raw.iloc[1, 1:]).fillna(method='ffill').astype(str)
    columns = [f"{p.strip()}_{t.strip()}" for p, t in zip(headers_1, headers_2)]
    columns.insert(0, "Listener")

    df_quality = df_raw.iloc[2:].copy()
    df_quality.columns = columns
    df_quality = df_quality.dropna(subset=["Listener"])

    # Melt into long format
    df_long = df_quality.melt(id_vars="Listener", var_name="Condition", value_name="Score")
    df_long[['Pathology', 'Type']] = df_long['Condition'].str.extract(r'(.+)_([A-Za-z]+)')
    df_long['Score'] = pd.to_numeric(df_long['Score'], errors='coerce')
    df_long = df_long.dropna(subset=['Score'])

    native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']
    nonnative_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

    results = []

    for group, label in [
        (df_long['Listener'].unique(), "All"),
        (nonnative_listeners, "Non-native"),
        (native_listeners, "Native")
    ]:
        for pathology in df_long['Pathology'].unique():
            sub = df_long[(df_long['Pathology'] == pathology) & (df_long['Listener'].isin(group))]
            orig = sub[sub['Type'] == 'Original'].set_index('Listener')['Score']
            anon = sub[sub['Type'] == 'Anonymized'].set_index('Listener')['Score']
            paired = pd.DataFrame({'Original': orig, 'Anonymized': anon}).dropna()

            if not paired.empty:
                t, p = ttest_rel(paired['Original'], paired['Anonymized'])
                m_orig, s_orig = paired['Original'].mean(), paired['Original'].std()
                m_anon, s_anon = paired['Anonymized'].mean(), paired['Anonymized'].std()
            else:
                m_orig = m_anon = s_orig = s_anon = p = np.nan

            results.append({
                'Pathology': pathology,
                'Listener Group': label,
                'Original Mean': round(m_orig),
                'Original SD': round(s_orig),
                'Anonymized Mean': round(m_anon),
                'Anonymized SD': round(s_anon),
                'p-value': p
            })

        # Overall comparison for group
        overall = df_long[df_long['Listener'].isin(group)].groupby(['Listener', 'Type'])['Score'].mean().unstack()
        t, p = ttest_rel(overall['Original'], overall['Anonymized'])
        m_orig, s_orig = overall['Original'].mean(), overall['Original'].std()
        m_anon, s_anon = overall['Anonymized'].mean(), overall['Anonymized'].std()

        results.append({
            'Pathology': 'Overall',
            'Listener Group': label,
            'Original Mean': round(m_orig),
            'Original SD': round(s_orig),
            'Anonymized Mean': round(m_anon),
            'Anonymized SD': round(s_anon),
            'p-value': p
        })

    df_stats = pd.DataFrame(results)

    # -------------------------------
    # FDR correction per group
    # -------------------------------
    corrected_rows = []
    for group in ['All', 'Non-native', 'Native']:
        df_sub = df_stats[(df_stats['Listener Group'] == group) & (df_stats['Pathology'] != 'Overall')].copy()
        pvals = df_sub['p-value'].dropna().values
        _, fdr_corrected, _, _ = multipletests(pvals, method='fdr_bh')
        df_sub.loc[:, 'p-value'] = np.round(fdr_corrected, 3)
        corrected_rows.append(df_sub)

    # Append 'Overall' rows (not corrected)
    df_overall = df_stats[df_stats['Pathology'] == 'Overall']
    df_final = pd.concat(corrected_rows + [df_overall], ignore_index=True)
    df_final['p-value'] = df_final['p-value'].round(3)

    # -------------------------------
    # Format as mean ± SD
    # -------------------------------
    df_final['Original'] = df_final['Original Mean'].astype("Int64").astype(str) + ' ± ' + df_final[
        'Original SD'].astype("Int64").astype(str)
    df_final['Anonymized'] = df_final['Anonymized Mean'].astype("Int64").astype(str) + ' ± ' + df_final[
        'Anonymized SD'].astype("Int64").astype(str)

    df_final = df_final[['Pathology', 'Listener Group', 'Original', 'Anonymized', 'p-value']]

    df_final.to_csv("./quality/quality_values_org_vs_anonym.csv", index=False)


    #####################################################################################################
    # bar chart figure

    bar_data = []

    for group in ['All', 'Non-native', 'Native']:
        df_group = df_final[df_final['Listener Group'] == group].copy()

        for idx, row in df_group.iterrows():
            try:
                orig_mean, orig_std = [x.strip() for x in row['Original'].split('±')]
                anon_mean, anon_std = [x.strip() for x in row['Anonymized'].split('±')]

                orig_mean = float(orig_mean)
                orig_std = float(orig_std)

                anon_mean = float(anon_mean)
                anon_std = float(anon_std)
            except:
                continue

            bar_data.append({
                'Group': group,
                'Pathology': row['Pathology'],
                'Type': 'Original',
                'Mean': orig_mean,
                'Std': orig_std,
                'p-value': row['p-value']
            })
            bar_data.append({
                'Group': group,
                'Pathology': row['Pathology'],
                'Type': 'Anonymized',
                'Mean': anon_mean,
                'Std': anon_std,
                'p-value': row['p-value']
            })

    df_bar = pd.DataFrame(bar_data)

    # --- Plot layout settings ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, sharey=True)
    groups = ['All', 'Non-native', 'Native']
    titles = ['a) All listeners', 'b) Non-native listeners', 'c) Native listeners']

    for i, grp in enumerate(groups):
        ax = axes[i]
        df_plot = df_bar[df_bar['Group'] == grp]
        order = df_plot['Pathology'].unique().tolist()

        # Manual bar plotting for control over error bars
        for j, pathology in enumerate(order):
            for k, bar_type in enumerate(['Original', 'Anonymized']):
                bar = df_plot[(df_plot['Pathology'] == pathology) & (df_plot['Type'] == bar_type)]
                if not bar.empty:
                    mean = bar['Mean'].values[0]
                    std = bar['Std'].values[0]
                    x_pos = j + (k - 0.5) * 0.2
                    color = sns.color_palette('Set2')[k]
                    ax.bar(x_pos, mean, yerr=std, width=0.18, label=bar_type if j == 0 else "",
                           color=color, capsize=5, edgecolor='black')

        # Title, labels, ticks
        ax.set_title(titles[i], loc='left', fontsize=18)
        ax.set_ylabel('Perceived quality normalized [%]', fontsize=16)
        ax.set_ylim(0, 110)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, fontsize=14)
        if i == 2:
            ax.set_xlabel('Speech pathology group', fontsize=16)
        else:
            ax.set_xlabel('')
        ax.legend(loc="lower right", fontsize=14, title_fontsize=15)

        # Despine top and right edges
        sns.despine(ax=ax, top=True, right=True)

        # Annotate p-values above bars
        for j, pathology in enumerate(order):
            subset = df_plot[(df_plot['Group'] == grp) & (df_plot['Pathology'] == pathology)]
            if subset['p-value'].nunique() == 1:
                pval = subset['p-value'].iloc[0]
                if pd.notna(pval):
                    ptext = f"p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
                    bar_max = subset['Mean'].max()
                    std_max = subset['Std'].max()
                    ax.text(j, bar_max + std_max + -3, ptext, ha='left', fontsize=13)

    plt.tight_layout()
    plt.savefig("./quality/quality_bars.png", dpi=300)

    #####################################################################################################
    # post hoc

    headers_1 = pd.Series(df_raw.iloc[0, 1:]).fillna(method='ffill').astype(str)
    headers_2 = pd.Series(df_raw.iloc[1, 1:]).fillna(method='ffill').astype(str)
    columns = [f"{p.strip()}_{t.strip()}" for p, t in zip(headers_1, headers_2)]
    columns.insert(0, "Listener")

    df_quality = df_raw.iloc[2:].copy()
    df_quality.columns = columns
    df_quality = df_quality.dropna(subset=["Listener"])

    df_long = df_quality.melt(id_vars="Listener", var_name="Condition", value_name="Score")
    df_long[['Pathology', 'Type']] = df_long['Condition'].str.extract(r'(.+)_([A-Za-z]+)')
    df_long['Score'] = pd.to_numeric(df_long['Score'], errors='coerce')
    df_long = df_long.dropna(subset=['Score'])

    # Compute degradation scores (original - anonymized)
    pivot_df = df_long.pivot_table(index=['Listener', 'Pathology'], columns='Type', values='Score').reset_index()
    pivot_df['Degradation'] = pivot_df['Original'] - pivot_df['Anonymized']

    # One-way ANOVA across pathology groups
    groups = [group['Degradation'].dropna().values for name, group in pivot_df.groupby('Pathology')]
    f_stat, p_val_anova = f_oneway(*groups)
    df1 = len(pivot_df['Pathology'].unique()) - 1
    df2 = len(pivot_df) - len(pivot_df['Pathology'].unique())

    print(f"Degrees of freedom: df1 = {df1}, df2 = {df2}")

    anova_df = pd.DataFrame({
        "Metric": ["F-statistic", "p-value", "df1 (between groups)", "df2 (within groups)"],
        "Value": [round(f_stat, 2), round(p_val_anova, 3), df1, df2]
    })

    anova_csv_path = "./quality/quality_degradation_anova.csv"
    anova_df.to_csv(anova_csv_path, index=False)

    print(f"ANOVA result: F = {f_stat:.2f}, p = {p_val_anova:.3f}")

    # Pairwise post-hoc comparisons
    pathologies = sorted(pivot_df['Pathology'].unique())
    pairwise_results = []
    for i in range(len(pathologies)):
        for j in range(i + 1, len(pathologies)):
            g1 = pathologies[i]
            g2 = pathologies[j]
            vals1 = pivot_df[pivot_df['Pathology'] == g1]['Degradation']
            vals2 = pivot_df[pivot_df['Pathology'] == g2]['Degradation']
            t_stat, p_val = ttest_rel(vals1, vals2)
            pairwise_results.append((g1, g2, p_val))

    # FDR correction
    raw_pvals = [res[2] for res in pairwise_results]
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

    # Build 6x6 matrix
    pval_matrix = pd.DataFrame(np.ones((len(pathologies), len(pathologies))), index=pathologies, columns=pathologies)

    for (group1, group2, _), corr_p in zip(pairwise_results, corrected_pvals):
        pval_matrix.loc[group1, group2] = corr_p
        pval_matrix.loc[group2, group1] = corr_p

    pval_matrix = pval_matrix.round(3)

    pval_matrix.to_csv("./quality/quality_degradation_posthoc_pvalues.csv")
    #####################################################################################################
    # Native vs. non native

    native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']
    nonnative_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

    pivot_df['Listener'] = pivot_df['Listener'].str.strip()

    existing_nonnative = [l for l in nonnative_listeners if l in pivot_df['Listener'].unique()]
    existing_native = [l for l in native_listeners if l in pivot_df['Listener'].unique()]

    # Aggregate degradation per listener per pathology
    agg_df = pivot_df.groupby(['Listener', 'Pathology'])['Degradation'].mean().reset_index()

    # Collect comparison results
    results = []

    # Per-pathology comparisons
    for pathology in agg_df['Pathology'].unique():
        subset = agg_df[agg_df['Pathology'] == pathology]

        native_vals = subset[subset['Listener'].isin(existing_native)]['Degradation']
        nonnative_vals = subset[subset['Listener'].isin(existing_nonnative)]['Degradation']

        t_stat, p_val = ttest_ind(native_vals, nonnative_vals)
        native_mean = round(native_vals.mean())
        native_std = round(native_vals.std())
        nonnative_mean = round(nonnative_vals.mean())
        nonnative_std = round(nonnative_vals.std())

        results.append({
            "Pathology": pathology,
            "Native": f"{native_mean} ± {native_std}",
            "Non-native": f"{nonnative_mean} ± {nonnative_std}",
            "p-value": p_val  # raw for now
        })

    # Overall comparison across all groups
    native_all = agg_df[agg_df['Listener'].isin(existing_native)]['Degradation']
    nonnative_all = agg_df[agg_df['Listener'].isin(existing_nonnative)]['Degradation']

    t_stat_all, p_val_all = ttest_ind(native_all, nonnative_all)
    native_all_mean = round(native_all.mean())
    native_all_std = round(native_all.std())
    nonnative_all_mean = round(nonnative_all.mean())
    nonnative_all_std = round(nonnative_all.std())

    results.append({
        "Pathology": "Overall",
        "Native": f"{native_all_mean} ± {native_all_std}",
        "Non-native": f"{nonnative_all_mean} ± {nonnative_all_std}",
        "p-value": p_val_all  # uncorrected
    })

    # Apply FDR correction to the first 6 pathology p-values
    results_pathologies = results[:-1]
    results_overall = results[-1]

    raw_pvals = [r['p-value'] for r in results_pathologies]
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

    # Replace p-values with corrected ones (rounded to 3 decimals)
    for i, corr_p in enumerate(corrected_pvals):
        results_pathologies[i]['p-value'] = round(corr_p, 3)

    results_overall['p-value'] = round(results_overall['p-value'], 3)

    df_final = pd.DataFrame(results_pathologies + [results_overall])

    df_final.to_csv("./quality/quality_degradation_native_vs_nonnative.csv", index=False)



def correlation():

    df_zeroshot = pd.read_csv(os.path.join(basedir, 'zeroshottemp.csv'))
    df_fewshot = pd.read_csv(os.path.join(basedir, 'fewshottemp.csv'))


    # Extract numeric parts
    df_zeroshot_numeric = df_zeroshot.select_dtypes(include='number')
    df_fewshot_numeric = df_fewshot.select_dtypes(include='number')

    # -------------------------------
    # 2. Manually define EER and Gain data
    # -------------------------------

    # Zero-shot EER values
    anonymized_zeroshot = pd.DataFrame({
        "Dysarthria": [36.59],
        "Dysglossia": [34.26],
        "Dysphonia": [38.86],
        "CLP": [32.19],
        "Overall patients": [30.24]
    })

    gain_zeroshot = pd.DataFrame({
        "Dysarthria": [34.79],
        "Dysglossia": [32.48],
        "Dysphonia": [36.67],
        "CLP": [25.18],
        "Overall patients": [27.28]
    })

    # -------------------------------
    # 3. Combine into labeled DataFrames
    # -------------------------------

    df_combined = pd.concat([
        df_zeroshot_numeric,
        anonymized_zeroshot,
        gain_zeroshot
    ], ignore_index=True)

    df_combined.index = ['AccuracyZeroshot', 'Anonymized', 'Gain']

    df_combined_fewshot = pd.concat([
        df_fewshot_numeric,
        anonymized_fewshot,
        gain_fewshot
    ], ignore_index=True)

    df_combined_fewshot.index = ['AccuracyFewshot', 'AnonymizedFewshot', 'GainFewshot']

    # -------------------------------
    # 4. Compute Pearson r & p-values
    # -------------------------------

    r1, p1 = pearsonr(df_combined.loc['Anonymized'], df_combined.loc['AccuracyZeroshot'])
    r2, p2 = pearsonr(df_combined.loc['Gain'], df_combined.loc['AccuracyZeroshot'])
    r3, p3 = pearsonr(df_combined_fewshot.loc['AnonymizedFewshot'], df_combined_fewshot.loc['AccuracyFewshot'])
    r4, p4 = pearsonr(df_combined_fewshot.loc['GainFewshot'], df_combined_fewshot.loc['AccuracyFewshot'])

    # Summary table
    summary_df = pd.DataFrame({
        "Metric Comparison": [
            "EER vs Zero-shot Accuracy",
            "EER Gain vs Zero-shot Accuracy",
            "EER vs Few-shot Accuracy",
            "EER Gain vs Few-shot Accuracy"
        ],
        "Pearson’s r": [round(r1, 3), round(r2, 3), round(r3, 3), round(r4, 3)],
        "p-value": [round(p1, 3), round(p2, 3), round(p3, 3), round(p4, 3)]
    })

    # Save summary CSV
    summary_path = "./pearson_summary_all_with_gain.csv"
    summary_df.to_csv(summary_path, index=False)

    # -------------------------------
    # 5. Scatter plot (only EER vs Accuracy)
    # -------------------------------

    plt.figure(figsize=(8, 6))

    # Zero-shot vs EER
    plt.scatter(df_combined.loc['AccuracyZeroshot'], df_combined.loc['Anonymized'],
                label='Zero-shot vs EER', marker='o')
    pdb.set_trace()

    # Few-shot vs EER
    plt.scatter(df_combined_fewshot.loc['AccuracyFewshot'], df_combined_fewshot.loc['AnonymizedFewshot'],
                label='Few-shot vs EER', marker='^')

    plt.title("Accuracy vs EER (Zero-shot & Few-shot)")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("EER (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = "./filtered_scatter_plot.png"
    plt.savefig(plot_path)
    plt.close()

    print("Saved summary table:", summary_path)
    print("Saved scatter plot:", plot_path)



def scatterplot():
    os.makedirs('correlation', exist_ok=True)

    # Subplot 1: EER vs Turing Accuracy
    a0 = [36.59, 34.26, 38.86, 32.19, 30.24]  # EER [fixed]
    b0_circle = [91.75, 96.75, 88.625, 84.5, 90.40625] # [change] zeroshot
    b0_triangle = [91.75, 96.625, 91.125, 88.375, 91.96875] # [change] fewshot



    a1 = [94.86, 98.86, 98.38, 96.37, 96.07]  # AUC anonym [fixed]
    b1_circle = [55.875, 63.625, 60.625, 64.75, 61.21875] # [change] quality anonym

    a2 = [97.33, 97.73, 99.12, 96.44, 97.05]  # AUC org [fixed]
    b2_circle = [82.75, 90, 81.625, 82.625, 84.25] # [change] quality org



    fig, axs = plt.subplots(1, 3, figsize=(18, 8))

    # --- First Subplot ---
    axs[0].scatter(b0_circle, a0, color='blue', marker='o', label='Zero-shot Turing', s=120)
    axs[0].scatter(b0_triangle, a0, color='orange', marker='^', label='Few-shot Turing', s=120)
    axs[0].set_title("a) EER vs. Turing accuracy", fontsize=20, loc='left')
    axs[0].set_xlabel("Turing accuracy [%]", fontsize=18)
    axs[0].set_ylabel("EER [%]", fontsize=18)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].legend(fontsize=16)
    axs[0].set_ylim(30, 41)
    axs[0].set_xlim(84, 98)

    # --- Second Subplot ---
    axs[1].scatter(b1_circle, a1, color='red', marker='s', label='Anonymized', s=120)  # Red squares
    axs[1].set_title("b) Quality anonymized", fontsize=20, loc='left')
    axs[1].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[1].set_ylabel("AUC [%]", fontsize=18)
    axs[1].tick_params(axis='both', labelsize=16)
    # axs[1].legend(fontsize=16)
    axs[1].set_ylim(94, 99)
    axs[1].set_xlim(54, 65)

    # --- Third Subplot ---
    axs[2].scatter(b2_circle, a2, color='black', marker='*', label='Original', s=120)  # Red squares
    axs[2].set_title("c) Quality original", fontsize=20, loc='left')
    axs[2].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[2].set_ylabel("AUC [%]", fontsize=18)
    axs[2].tick_params(axis='both', labelsize=16)
    # axs[2].legend(fontsize=16)
    axs[2].set_ylim(94, 100)
    axs[2].set_xlim(80, 91)


    plt.tight_layout()
    plt.savefig("./correlation/scatter_plot_correlation.png")




def boxplot_appender():
    img1 = mpimg.imread('./zero_shot/turing_box_zeroshot.png')
    img2 = mpimg.imread('./few_shot/turing_box_fewshot.png')

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))  # Optimal screen fit

    # Display each image in a subplot
    axs[0].imshow(img1)
    axs[0].axis('off')  # Hide axes
    axs[0].set_title('a) Turing test accuracy for single-shot analysis', fontsize=20, loc='left')

    axs[1].imshow(img2)
    axs[1].axis('off')  # Hide axes
    axs[1].set_title('b) Turing test accuracy for few-shot analysis', fontsize=20, loc='left')

    # Adjust layout and show the figure
    plt.tight_layout(pad=3.0)
    plt.savefig("./zero_shot/boxplot.png")




if __name__ == '__main__':
    # zeroshot_turing()
    # fewishot_turing()
    # boxplot_appender()
    # male_vs_female_turing_fewshot()
    # male_vs_female_turing_zeroshot()
    # quality()
    scatterplot()
