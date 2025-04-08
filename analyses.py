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
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from contextlib import redirect_stdout
from scipy.stats import ttest_ind



import warnings
warnings.filterwarnings('ignore')


basedir = 'xxxxxx'



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
            plt.title("Turing test accuracy for zero-shot analysis", fontsize=18)
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
            print(f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.4f}")

            # Smart interpretation
            if p_value < 0.05:
                interpretation = "This indicates a statistically significant effect of pathology group on Turing test accuracy."
            else:
                interpretation = "This indicates that there is no statistically significant difference in accuracy across pathology groups."

            print("\nInterpretation:")
            print(f"A repeated-measures ANOVA revealed a main effect of pathology group on discrimination accuracy, "
                  f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.4f}. \n{interpretation}")

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
                    sig_comparisons_fdr.append((g1, g2, round(corr_p, 4)))

            # Round matrix for presentation
            fdr_matrix_rounded = fdr_matrix.round(4)
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
            native_listeners = ['EN', 'MS', 'TN', 'LB']
            non_native_listeners = ['SA', 'TA', 'PP', 'MP']

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
                print(f"  Native: {native_avg} ± {native_sd}% vs. Non-native: {non_native_avg} ± {non_native_sd}% (t = {t_stat:.2f}, p = {p_val:.4f})\n")



def fewishot_turing():
    os.makedirs('few_shot', exist_ok=True)

    with open("./few_shot/turing_fewshot.txt", "w") as f:
        with redirect_stdout(f):

            print('\t\tResults for Few-shot case\n\n')

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
            plt.title("Turing test accuracy for few-shot analysis", fontsize=18)
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
            print(f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.4f}")

            # Smart interpretation
            if p_value < 0.05:
                interpretation = "This indicates a statistically significant effect of pathology group on Turing test accuracy."
            else:
                interpretation = "This indicates that there is no statistically significant difference in accuracy across pathology groups."

            print("\nInterpretation:")
            print(f"A repeated-measures ANOVA revealed a main effect of pathology group on discrimination accuracy, "
                  f"F({df1}, {df2}) = {f_value:.2f}, p = {p_value:.4f}. \n{interpretation}")

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
                    sig_comparisons_fdr.append((g1, g2, round(corr_p, 4)))

            # Round matrix for presentation
            fdr_matrix_rounded = fdr_matrix.round(4)
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
            native_listeners = ['EN', 'MS', 'TN', 'LB']
            non_native_listeners = ['SA', 'TA', 'PP', 'MP']

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
                print(f"  Native: {native_avg} ± {native_sd}% vs. Non-native: {non_native_avg} ± {non_native_sd}% (t = {t_stat:.2f}, p = {p_val:.4f})\n")



def male_vs_female_turing_fewshot():
    os.makedirs('few_shot', exist_ok=True)

    male_df = pd.read_csv(os.path.join(basedir, 'male - accuracyfewshot.csv'))
    female_df = pd.read_csv(os.path.join(basedir, 'female - accuracyfewshot.csv'))

    pathologies = ['CLP', 'Control adults', 'Control children', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']
    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']

    def mean_sd(arr):
        arr = arr[~np.isnan(arr)]
        return round(arr.mean()), round(arr.std())

    results = []

    # ---- Per-pathology comparisons ----
    for pathology in pathologies:
        male_scores = male_df[pathology].dropna()
        female_scores = female_df[pathology].dropna()

        t_stat, p_val = ttest_ind(male_scores, female_scores)
        m_mean, m_sd = mean_sd(male_scores)
        f_mean, f_sd = mean_sd(female_scores)

        results.append({
            'Group': pathology,
            'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
            'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
            'Raw p-value': p_val
        })

    # ---- Controls ----
    male_ctrl = male_df[controls].values.flatten()
    female_ctrl = female_df[controls].values.flatten()
    t_ctrl, p_ctrl = ttest_ind(male_ctrl[~np.isnan(male_ctrl)], female_ctrl[~np.isnan(female_ctrl)])
    m_mean, m_sd = mean_sd(male_ctrl)
    f_mean, f_sd = mean_sd(female_ctrl)
    results.append({
        'Group': 'Controls',
        'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
        'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
        'Raw p-value': p_ctrl
    })

    # ---- Patients ----
    male_pat = male_df[patients].values.flatten()
    female_pat = female_df[patients].values.flatten()
    t_pat, p_pat = ttest_ind(male_pat[~np.isnan(male_pat)], female_pat[~np.isnan(female_pat)])
    m_mean, m_sd = mean_sd(male_pat)
    f_mean, f_sd = mean_sd(female_pat)
    results.append({
        'Group': 'Patients',
        'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
        'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
        'Raw p-value': p_pat
    })

    # ---- Overall ----
    male_all = male_df[pathologies].values.flatten()
    female_all = female_df[pathologies].values.flatten()
    t_all, p_all = ttest_ind(male_all[~np.isnan(male_all)], female_all[~np.isnan(female_all)])
    m_mean, m_sd = mean_sd(male_all)
    f_mean, f_sd = mean_sd(female_all)
    results.append({
        'Group': 'Overall',
        'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
        'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
        'Raw p-value': p_all
    })

    # ---- Apply FDR correction ----
    raw_pvals = [row['Raw p-value'] for row in results]
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

    # Add corrected values
    for i, p_corr in enumerate(corrected_pvals):
        results[i]['FDR-corrected p-value'] = round(p_corr, 4)
        results[i]['Raw p-value'] = round(results[i]['Raw p-value'], 4)

    gender_comparison_df = pd.DataFrame(results)
    gender_comparison_df.to_csv("./few_shot/gender_fewshot.csv", index=False)



def male_vs_female_turing_zeroshot():
    os.makedirs('zero_shot', exist_ok=True)

    male_df = pd.read_csv(os.path.join(basedir, 'male - accuracyzeroshot.csv'))
    female_df = pd.read_csv(os.path.join(basedir, 'female - accuracyzeroshot.csv'))

    pathologies = ['CLP', 'Control adults', 'Control children', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']
    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']

    def mean_sd(arr):
        arr = arr[~np.isnan(arr)]
        return round(arr.mean()), round(arr.std())

    results = []

    # ---- Per-pathology comparisons ----
    for pathology in pathologies:
        male_scores = male_df[pathology].dropna()
        female_scores = female_df[pathology].dropna()

        t_stat, p_val = ttest_ind(male_scores, female_scores)
        m_mean, m_sd = mean_sd(male_scores)
        f_mean, f_sd = mean_sd(female_scores)

        results.append({
            'Group': pathology,
            'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
            'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
            'Raw p-value': p_val
        })

    # ---- Controls ----
    male_ctrl = male_df[controls].values.flatten()
    female_ctrl = female_df[controls].values.flatten()
    t_ctrl, p_ctrl = ttest_ind(male_ctrl[~np.isnan(male_ctrl)], female_ctrl[~np.isnan(female_ctrl)])
    m_mean, m_sd = mean_sd(male_ctrl)
    f_mean, f_sd = mean_sd(female_ctrl)
    results.append({
        'Group': 'Controls',
        'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
        'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
        'Raw p-value': p_ctrl
    })

    # ---- Patients ----
    male_pat = male_df[patients].values.flatten()
    female_pat = female_df[patients].values.flatten()
    t_pat, p_pat = ttest_ind(male_pat[~np.isnan(male_pat)], female_pat[~np.isnan(female_pat)])
    m_mean, m_sd = mean_sd(male_pat)
    f_mean, f_sd = mean_sd(female_pat)
    results.append({
        'Group': 'Patients',
        'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
        'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
        'Raw p-value': p_pat
    })

    # ---- Overall ----
    male_all = male_df[pathologies].values.flatten()
    female_all = female_df[pathologies].values.flatten()
    t_all, p_all = ttest_ind(male_all[~np.isnan(male_all)], female_all[~np.isnan(female_all)])
    m_mean, m_sd = mean_sd(male_all)
    f_mean, f_sd = mean_sd(female_all)
    results.append({
        'Group': 'Overall',
        'Male Accuracy (Mean ± SD)': f"{m_mean} ± {m_sd}",
        'Female Accuracy (Mean ± SD)': f"{f_mean} ± {f_sd}",
        'Raw p-value': p_all
    })

    # ---- Apply FDR correction ----
    raw_pvals = [row['Raw p-value'] for row in results]
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

    # Add corrected values
    for i, p_corr in enumerate(corrected_pvals):
        results[i]['FDR-corrected p-value'] = round(p_corr, 4)
        results[i]['Raw p-value'] = round(results[i]['Raw p-value'], 4)

    gender_comparison_df = pd.DataFrame(results)
    gender_comparison_df.to_csv("./zero_shot/gender_zeroshot.csv", index=False)




if __name__ == '__main__':
    # zeroshot_turing()
    # fewishot_turing()
    male_vs_female_turing_fewshot()
    # male_vs_female_turing_zeroshot()
