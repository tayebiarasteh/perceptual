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
from scipy.stats import f_oneway, ttest_rel, ttest_ind, pearsonr, spearmanr, mannwhitneyu, shapiro, wilcoxon
from statsmodels.stats.multitest import multipletests
from contextlib import redirect_stdout
import matplotlib.image as mpimg
from PIL import Image



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
                        # t_stat, p_val = wilcoxon(vals1, vals2)
                        pairwise_results_fdr.append((g1, g2, p_val))
                        # print(shapiro(vals1), shapiro(vals1))

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
            # t_stat_all, p_val_all = ttest_ind(native_all, non_native_all)
            t_stat_all, p_val_all = mannwhitneyu(native_all, non_native_all)
            print('native shapiro:', shapiro(native_all))
            print('non-native shapiro:', shapiro(non_native_all))

            print(f"Overall \nNative: {native_avg_all} ± {native_sd_all}% vs. Non-native: {non_native_avg_all} ± {non_native_sd_all}% (t = {t_stat_all:.2f}, p = {p_val_all:.4f})\n")

            # for pathology in numeric_cols:
            #     native_scores = df_zero[df_zero['Listener'].isin(native_listeners)][pathology]
            #     non_native_scores = df_zero[df_zero['Listener'].isin(non_native_listeners)][pathology]
            #
            #     native_avg = round(native_scores.mean())
            #     native_sd = round(native_scores.std())
            #     non_native_avg = round(non_native_scores.mean())
            #     non_native_sd = round(non_native_scores.std())
            #
            #     t_stat, p_val = ttest_ind(native_scores, non_native_scores)
            #
            #     print(f"{pathology}:")
            #     print(f"  Native: {native_avg} ± {native_sd}% vs. Non-native: {non_native_avg} ± {non_native_sd}% (t = {t_stat:.2f}, p = {p_val:.3f})\n")



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
            # print("\nFDR-Corrected Pairwise P-values (6x6):")
            # print(fdr_matrix_rounded)

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

            # Independent test
            # t_stat_all, p_val_all = ttest_ind(native_all, non_native_all)
            t_stat_all, p_val_all = mannwhitneyu(native_all, non_native_all)
            print('native shapiro:', shapiro(native_all))
            print('non-native shapiro:', shapiro(non_native_all))

            print(f"Overall \nNative: {native_avg_all} ± {native_sd_all}% vs. Non-native: {non_native_avg_all} ± {non_native_sd_all}% (t = {t_stat_all:.2f}, p = {p_val_all:.4f})\n")
            print(f"Overall \nNative: {native_avg_all} ± {native_sd_all}% vs. Non-native: {non_native_avg_all} ± {non_native_sd_all}% (t = {t_stat_all:.2f}, p = {p_val_all:.4f})\n")

            # for pathology in numeric_cols:
            #     native_scores = df_zero[df_zero['Listener'].isin(native_listeners)][pathology]
            #     non_native_scores = df_zero[df_zero['Listener'].isin(non_native_listeners)][pathology]
            #
            #     native_avg = round(native_scores.mean())
            #     native_sd = round(native_scores.std())
            #     non_native_avg = round(non_native_scores.mean())
            #     non_native_sd = round(non_native_scores.std())
            #
            #     t_stat, p_val = ttest_ind(native_scores, non_native_scores)
            #
            #     print(f"{pathology}:")
            #     print(f"  Native: {native_avg} ± {native_sd}% vs. Non-native: {non_native_avg} ± {non_native_sd}% (t = {t_stat:.2f}, p = {p_val:.3f})\n")



def male_vs_female_turing_fewshot():
    os.makedirs('fairness', exist_ok=True)

    df_male = pd.read_csv(os.path.join(basedir, 'male - accuracyfewshot.csv')).dropna().set_index("Listener")
    df_female = pd.read_csv(os.path.join(basedir, 'female - accuracyfewshot.csv')).dropna().set_index("Listener")

    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']
    all_groups = patients + controls

    normality_results = []

    # Run Shapiro-Wilk test for each group
    for group in all_groups:
        male_vals = df_male[group]
        female_vals = df_female[group]

        # Shapiro-Wilk test
        stat_male, p_male = shapiro(male_vals)
        stat_female, p_female = shapiro(female_vals)

        normality_results.append({
            "Group": group,
            "Male Shapiro-W": round(stat_male, 3),
            "Male p-value": round(p_male, 4),
            "Female Shapiro-W": round(stat_female, 3),
            "Female p-value": round(p_female, 4)
        })

    df_normality = pd.DataFrame(normality_results)
    df_normality.to_csv("./fairness/normality_gender_fewshot.csv", index=False)

    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']

    results = []

    # Individual groups (Mann-Whitney U test)
    for group in patients + controls:
        u_stat, p_val = mannwhitneyu(df_male[group], df_female[group], alternative='two-sided')
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": p_val
        })

    # Add patient and control averages
    df_male["Patient Avg"] = df_male[patients].mean(axis=1)
    df_female["Patient Avg"] = df_female[patients].mean(axis=1)
    df_male["Control Avg"] = df_male[controls].mean(axis=1)
    df_female["Control Avg"] = df_female[controls].mean(axis=1)

    # Group averages (Mann-Whitney U test)
    for group in ["Patient Avg", "Control Avg"]:
        u_stat, p_val = mannwhitneyu(df_male[group], df_female[group], alternative='two-sided')
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": p_val
        })

    df_results = pd.DataFrame(results)

    # FDR correction (only for individual groups, not aggregated ones)
    raw_pvals = df_results.loc[df_results['Group'].isin(patients + controls), 'p-value']
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

    # Update corrected p-values
    df_results.loc[df_results['Group'].isin(patients + controls), 'p-value'] = corrected_pvals
    df_results['p-value'] = df_results['p-value'].round(3)

    df_results.to_csv("./fairness/gender_fairness_fewshot.csv", index=False)



def male_vs_female_turing_zeroshot():
    os.makedirs('fairness', exist_ok=True)

    df_male = pd.read_csv(os.path.join(basedir, 'male - accuracyzeroshot.csv')).dropna().set_index("Listener")
    df_female = pd.read_csv(os.path.join(basedir, 'female - accuracyzeroshot.csv')).dropna().set_index("Listener")

    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']
    all_groups = patients + controls

    normality_results = []

    # Run Shapiro-Wilk test for each group
    for group in all_groups:
        male_vals = df_male[group]
        female_vals = df_female[group]

        # Shapiro-Wilk test
        stat_male, p_male = shapiro(male_vals)
        stat_female, p_female = shapiro(female_vals)

        normality_results.append({
            "Group": group,
            "Male Shapiro-W": round(stat_male, 3),
            "Male p-value": round(p_male, 4),
            "Female Shapiro-W": round(stat_female, 3),
            "Female p-value": round(p_female, 4)
        })

    df_normality = pd.DataFrame(normality_results)
    df_normality.to_csv("./fairness/normality_gender_zeroshot.csv", index=False)

    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    controls = ['Control adults', 'Control children']

    results = []

    # Individual groups (Mann-Whitney U test)
    for group in patients + controls:
        u_stat, p_val = mannwhitneyu(df_male[group], df_female[group], alternative='two-sided')
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": p_val
        })

    # Add patient and control averages
    df_male["Patient Avg"] = df_male[patients].mean(axis=1)
    df_female["Patient Avg"] = df_female[patients].mean(axis=1)
    df_male["Control Avg"] = df_male[controls].mean(axis=1)
    df_female["Control Avg"] = df_female[controls].mean(axis=1)

    # Group averages (Mann-Whitney U test)
    for group in ["Patient Avg", "Control Avg"]:
        u_stat, p_val = mannwhitneyu(df_male[group], df_female[group], alternative='two-sided')
        results.append({
            "Group": group,
            "Male Mean": round(df_male[group].mean(), 0),
            "Male Std": round(df_male[group].std(), 0),
            "Female Mean": round(df_female[group].mean(), 0),
            "Female Std": round(df_female[group].std(), 0),
            "p-value": p_val
        })

    df_results = pd.DataFrame(results)

    # FDR correction (only for individual groups, not aggregated ones)
    raw_pvals = df_results.loc[df_results['Group'].isin(patients + controls), 'p-value']
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

    # Update corrected p-values
    df_results.loc[df_results['Group'].isin(patients + controls), 'p-value'] = corrected_pvals
    df_results['p-value'] = df_results['p-value'].round(3)

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

    # Post-hoc comparisons for original and anonymized separately
    for score_type in ['Original', 'Anonymized']:
        pathologies = sorted(pivot_df['Pathology'].unique())
        pairwise_results = []
        for i in range(len(pathologies)):
            for j in range(i + 1, len(pathologies)):
                g1 = pathologies[i]
                g2 = pathologies[j]
                vals1 = pivot_df[pivot_df['Pathology'] == g1][score_type]
                vals2 = pivot_df[pivot_df['Pathology'] == g2][score_type]
                t_stat, p_val = ttest_rel(vals1, vals2)
                pairwise_results.append((g1, g2, p_val))

        raw_pvals = [res[2] for res in pairwise_results]
        _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

        # Build matrix
        pval_matrix = pd.DataFrame(np.ones((len(pathologies), len(pathologies))), index=pathologies,
                                   columns=pathologies)
        for (group1, group2, _), corr_p in zip(pairwise_results, corrected_pvals):
            pval_matrix.loc[group1, group2] = corr_p
            pval_matrix.loc[group2, group1] = corr_p
        pval_matrix = pval_matrix.round(3)

        # Save to file
        suffix = 'original' if score_type == 'Original' else 'anonymized'
        pval_matrix.to_csv(f"./quality/quality_posthoc_{suffix}.csv")

    #####################################################################################################
    # Native vs. non native

    native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']
    nonnative_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']
    pivot_df['Listener'] = pivot_df['Listener'].str.strip()

    existing_nonnative = [l for l in nonnative_listeners if l in pivot_df['Listener'].unique()]
    existing_native = [l for l in native_listeners if l in pivot_df['Listener'].unique()]

    os.makedirs('./quality', exist_ok=True)

    summary_rows = []

    for score_type in ['Original', 'Anonymized']:
        native_all = pivot_df[pivot_df['Listener'].isin(existing_native)][score_type]
        nonnative_all = pivot_df[pivot_df['Listener'].isin(existing_nonnative)][score_type]

        t_stat_all, p_val_all = ttest_ind(native_all, nonnative_all)

        native_all_mean = round(native_all.mean())
        native_all_std = round(native_all.std())
        nonnative_all_mean = round(nonnative_all.mean())
        nonnative_all_std = round(nonnative_all.std())

        summary_rows.append({
            "Score Type": score_type,
            "Native": f"{native_all_mean} ± {native_all_std}",
            "Non-native": f"{nonnative_all_mean} ± {nonnative_all_std}",
            "p-value": round(p_val_all, 3)
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("./quality/quality_native_vs_nonnative.csv", index=False)


def correlation():
    os.makedirs('correlation', exist_ok=True)

    with open("./correlation/correlation.txt", "w") as f:
        with redirect_stdout(f):

            a0 = [36.59, 34.26, 38.86, 32.19, 30.24]  # EER [fixed]
            a1 = [94.86, 98.86, 98.38, 96.37, 96.07]  # AUC anonym [fixed]
            a2 = [97.33, 97.73, 99.12, 96.44, 97.05]  # AUC org [fixed]

            df_fewshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyfewshot.csv'))
            df_zeroshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyzeroshot.csv'))

            # Drop Control adults and Control children columns
            drop_cols = ["Control adults", "Control children"]
            df_fewshot = df_fewshot.drop(columns=drop_cols, errors='ignore')
            df_zeroshot = df_zeroshot.drop(columns=drop_cols, errors='ignore')

            # Add patient average columns
            df_fewshot["Patient Average"] = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)
            df_zeroshot["Patient Average"] = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)

            # Compute column-wise averages
            fewshot = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()
            zeroshot = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()

            r_zero, p_zero = pearsonr(a0, zeroshot)
            print(f"All listeners: Pearson's r (EER vs Zero-shot): r = {r_zero:.3f}, p = {p_zero:.3f}")
            r_few, p_few = pearsonr(a0, fewshot)
            print(f"All listeners: Pearson's r (EER vs Few-shot): r = {r_few:.3f}, p = {p_few:.3f}")

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

            # Compute group-wise means
            grouped = df_long.groupby(['Pathology', 'Type'])['Score'].mean().unstack()

            # Compute overall average over the 4 patient groups
            patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
            grouped = grouped.loc[patients]
            grouped.loc['Patient Average'] = grouped.mean()

            anonym = grouped['Anonymized'].tolist()
            orig = grouped['Original'].tolist()

            r_anonym, p_anonym = pearsonr(a1, anonym)
            print(f"All listeners: Pearson's r (AUC vs Anonymized): r = {r_anonym:.3f}, p = {p_anonym:.3f}")
            r_orig, p_orig = pearsonr(a2, orig)
            print(f"All listeners: Pearson's r (AUC vs Original): r = {r_orig:.3f}, p = {p_orig:.3f}")


            ########## non native ##############
            df_fewshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyfewshot.csv'))
            df_zeroshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyzeroshot.csv'))

            # List of non-native listeners
            non_native_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

            # Filter for non-native listeners only
            df_fewshot = df_fewshot[df_fewshot['Listener'].isin(non_native_listeners)]
            df_zeroshot = df_zeroshot[df_zeroshot['Listener'].isin(non_native_listeners)]

            # Drop Control adults and Control children columns
            drop_cols = ["Control adults", "Control children"]
            df_fewshot = df_fewshot.drop(columns=drop_cols, errors='ignore')
            df_zeroshot = df_zeroshot.drop(columns=drop_cols, errors='ignore')

            # Add patient average columns
            df_fewshot["Patient Average"] = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)
            df_zeroshot["Patient Average"] = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)

            fewshot = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()
            zeroshot = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()

            r_zero, p_zero = pearsonr(a0, zeroshot)
            print('\n############# Non-native ################')
            print(f"Non-native listeners: Pearson's r (EER vs Zero-shot): r = {r_zero:.3f}, p = {p_zero:.3f}")
            r_few, p_few = pearsonr(a0, fewshot)
            print(f"Non-native listeners: Pearson's r (EER vs Few-shot): r = {r_few:.3f}, p = {p_few:.3f}")

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

            # List of non-native listeners
            non_native_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

            # Filter for non-native listeners only
            df_long = df_long[df_long['Listener'].isin(non_native_listeners)]

            # Compute group-wise means
            grouped = df_long.groupby(['Pathology', 'Type'])['Score'].mean().unstack()

            # Compute overall average over the 4 patient groups
            patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
            grouped = grouped.loc[patients]
            grouped.loc['Patient Average'] = grouped.mean()

            anonym = grouped['Anonymized'].tolist()
            orig = grouped['Original'].tolist()

            r_anonym, p_anonym = pearsonr(a1, anonym)
            print(f"Non-native listeners: Pearson's r (AUC vs Anonymized): r = {r_anonym:.3f}, p = {p_anonym:.3f}")
            r_orig, p_orig = pearsonr(a2, orig)
            print(f"Non-native listeners: Pearson's r (AUC vs Original): r = {r_orig:.3f}, p = {p_orig:.3f}")




            ########## Native ##############
            df_fewshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyfewshot.csv'))
            df_zeroshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyzeroshot.csv'))

            # List of Native listeners
            native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']

            # Filter for Native listeners only
            df_fewshot = df_fewshot[df_fewshot['Listener'].isin(native_listeners)]
            df_zeroshot = df_zeroshot[df_zeroshot['Listener'].isin(native_listeners)]

            # Drop Control adults and Control children columns
            drop_cols = ["Control adults", "Control children"]
            df_fewshot = df_fewshot.drop(columns=drop_cols, errors='ignore')
            df_zeroshot = df_zeroshot.drop(columns=drop_cols, errors='ignore')

            # Add patient average columns
            df_fewshot["Patient Average"] = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)
            df_zeroshot["Patient Average"] = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)

            fewshot = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()
            zeroshot = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()

            r_zero, p_zero = pearsonr(a0, zeroshot)
            print('\n############# Native ################')
            print(f"Native listeners: Pearson's r (EER vs Zero-shot): r = {r_zero:.3f}, p = {p_zero:.3f}")
            r_few, p_few = pearsonr(a0, fewshot)
            print(f"Native listeners: Pearson's r (EER vs Few-shot): r = {r_few:.3f}, p = {p_few:.3f}")

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

            # List of native listeners
            native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']
            df_long = df_long[df_long['Listener'].isin(native_listeners)]

            # Compute group-wise means
            grouped = df_long.groupby(['Pathology', 'Type'])['Score'].mean().unstack()

            # Compute overall average over the 4 patient groups
            patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
            grouped = grouped.loc[patients]
            grouped.loc['Patient Average'] = grouped.mean()

            anonym = grouped['Anonymized'].tolist()
            orig = grouped['Original'].tolist()

            r_anonym, p_anonym = pearsonr(a1, anonym)
            print(f"Native listeners: Pearson's r (AUC vs Anonymized): r = {r_anonym:.3f}, p = {p_anonym:.3f}")
            r_orig, p_orig = pearsonr(a2, orig)
            print(f"Native listeners: Pearson's r (AUC vs Original): r = {r_orig:.3f}, p = {p_orig:.3f}")



def scatterplot():
    os.makedirs('correlation', exist_ok=True)

    ######################### for all ##############################

    # Subplot 1: EER vs Turing Accuracy
    a0 = [36.59, 34.26, 38.86, 32.19, 30.24]  # EER [fixed]

    df_fewshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyfewshot.csv'))
    df_zeroshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyzeroshot.csv'))

    # Drop Control adults and Control children columns
    drop_cols = ["Control adults", "Control children"]
    df_fewshot = df_fewshot.drop(columns=drop_cols, errors='ignore')
    df_zeroshot = df_zeroshot.drop(columns=drop_cols, errors='ignore')

    # Add patient average columns
    df_fewshot["Patient Average"] = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)
    df_zeroshot["Patient Average"] = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)

    # Compute column-wise averages
    b0_triangle = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()
    b0_circle = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()


    # subplot 2
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

    # Compute group-wise means
    grouped = df_long.groupby(['Pathology', 'Type'])['Score'].mean().unstack()

    # Compute overall average over the 4 patient groups
    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    grouped = grouped.loc[patients]
    grouped.loc['Patient Average'] = grouped.mean()

    b1_circle = grouped['Anonymized'].tolist()
    b2_circle = grouped['Original'].tolist()

    a1 = [94.86, 98.86, 98.38, 96.37, 96.07]  # AUC anonym [fixed]
    a2 = [97.33, 97.73, 99.12, 96.44, 97.05]  # AUC org [fixed]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # --- First Subplot ---
    axs[0].scatter(b0_circle, a0, color='blue', marker='o', label='Zero-shot Turing', s=120)
    axs[0].scatter(b0_triangle, a0, color='orange', marker='^', label='Few-shot Turing', s=120)
    axs[0].set_title("EER vs. Turing accuracy", fontsize=20, loc='center')
    axs[0].set_xlabel("Turing accuracy [%]", fontsize=18)
    axs[0].set_ylabel("EER [%]", fontsize=18)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].legend(fontsize=16)
    # axs[0].set_ylim(int(min(a0)), int(max(a0) +3))
    axs[0].set_ylim(29.7, 42)
    # axs[0].set_xlim(int(min(b0_circle)-3), 100)
    axs[0].set_xlim(79.01, 100)

    # --- Second Subplot ---
    axs[1].scatter(b1_circle, a1, color='red', marker='s', label='Anonymized', s=120)  # Red squares
    axs[1].set_title("Quality anonymized", fontsize=20, loc='center')
    axs[1].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[1].set_ylabel("AUC [%]", fontsize=18)
    axs[1].tick_params(axis='both', labelsize=16)
    # axs[1].legend(fontsize=16)
    # axs[1].set_ylim(int(min(a1)-1), 100)
    axs[1].set_ylim(93.5, 100)
    # axs[1].set_xlim(int(min(b1_circle)-1), int(max(b1_circle) +2))
    axs[1].set_xlim(49.1, 70)

    # --- Third Subplot ---
    axs[2].scatter(b2_circle, a2, color='black', marker='*', label='Original', s=120)  # Red squares
    axs[2].set_title("Quality original", fontsize=20, loc='center')
    axs[2].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[2].set_ylabel("AUC [%]", fontsize=18)
    axs[2].tick_params(axis='both', labelsize=16)
    # axs[2].legend(fontsize=16)
    # axs[2].set_ylim(int(min(a2)-1), 100)
    axs[2].set_ylim(95.49, 100)
    # axs[2].set_xlim(int(min(b2_circle)-1), int(max(b2_circle) +2))
    axs[2].set_xlim(74.7, 95)


    plt.tight_layout()
    plt.savefig("./correlation/scatter_plot_correlation.png")

    ######################### for non native ##############################

    # Subplot 1: EER vs Turing Accuracy
    a0 = [36.59, 34.26, 38.86, 32.19, 30.24]  # EER [fixed]

    df_fewshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyfewshot.csv'))
    df_zeroshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyzeroshot.csv'))

    # List of non-native listeners
    non_native_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

    # Filter for non-native listeners only
    df_fewshot = df_fewshot[df_fewshot['Listener'].isin(non_native_listeners)]
    df_zeroshot = df_zeroshot[df_zeroshot['Listener'].isin(non_native_listeners)]

    # Drop Control adults and Control children columns
    drop_cols = ["Control adults", "Control children"]
    df_fewshot = df_fewshot.drop(columns=drop_cols, errors='ignore')
    df_zeroshot = df_zeroshot.drop(columns=drop_cols, errors='ignore')

    # Add patient average columns
    df_fewshot["Patient Average"] = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)
    df_zeroshot["Patient Average"] = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)

    # Compute column-wise averages
    b0_triangle = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()
    b0_circle = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()

    # subplot 2
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

    # List of non-native listeners
    non_native_listeners = ['SA', 'TA', 'HH', 'MP', 'ML']

    # Filter for non-native listeners only
    df_long = df_long[df_long['Listener'].isin(non_native_listeners)]

    # Compute group-wise means
    grouped = df_long.groupby(['Pathology', 'Type'])['Score'].mean().unstack()

    # Compute overall average over the 4 patient groups
    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    grouped = grouped.loc[patients]
    grouped.loc['Patient Average'] = grouped.mean()

    b1_circle = grouped['Anonymized'].tolist()
    b2_circle = grouped['Original'].tolist()

    a1 = [94.86, 98.86, 98.38, 96.37, 96.07]  # AUC anonym [fixed]
    a2 = [97.33, 97.73, 99.12, 96.44, 97.05]  # AUC org [fixed]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # --- First Subplot ---
    axs[0].scatter(b0_circle, a0, color='blue', marker='o', label='Zero-shot Turing', s=120)
    axs[0].scatter(b0_triangle, a0, color='orange', marker='^', label='Few-shot Turing', s=120)
    axs[0].set_title("EER vs. Turing accuracy", fontsize=20, loc='center')
    axs[0].set_xlabel("Turing accuracy [%]", fontsize=18)
    axs[0].set_ylabel("EER [%]", fontsize=18)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].legend(fontsize=16)
    # axs[0].set_ylim(int(min(a0)), int(max(a0) +3))
    axs[0].set_ylim(29.7, 42)
    # axs[0].set_xlim(int(min(b0_circle)-3), 100)
    axs[0].set_xlim(79.01, 100)

    # --- Second Subplot ---
    axs[1].scatter(b1_circle, a1, color='red', marker='s', label='Anonymized', s=120)  # Red squares
    axs[1].set_title("Quality anonymized", fontsize=20, loc='center')
    axs[1].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[1].set_ylabel("AUC [%]", fontsize=18)
    axs[1].tick_params(axis='both', labelsize=16)
    # axs[1].legend(fontsize=16)
    # axs[1].set_ylim(int(min(a1)-1), 100)
    axs[1].set_ylim(93.5, 100)
    # axs[1].set_xlim(int(min(b1_circle)-1), int(max(b1_circle) +2))
    axs[1].set_xlim(49.1, 70)

    # --- Third Subplot ---
    axs[2].scatter(b2_circle, a2, color='black', marker='*', label='Original', s=120)  # Red squares
    axs[2].set_title("Quality original", fontsize=20, loc='center')
    axs[2].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[2].set_ylabel("AUC [%]", fontsize=18)
    axs[2].tick_params(axis='both', labelsize=16)
    # axs[2].legend(fontsize=16)
    # axs[2].set_ylim(int(min(a2)-1), 100)
    axs[2].set_ylim(95.49, 100)
    # axs[2].set_xlim(int(min(b2_circle)-1), int(max(b2_circle) +2))
    axs[2].set_xlim(74.7, 95)


    plt.tight_layout()
    plt.savefig("./correlation/scatter_plot_correlation_nonnative.png")

    ######################### for native ##############################

    # Subplot 1: EER vs Turing Accuracy
    a0 = [36.59, 34.26, 38.86, 32.19, 30.24]  # EER [fixed]

    df_fewshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyfewshot.csv'))
    df_zeroshot = pd.read_csv(os.path.join(basedir, 'Perceptual speech anonym - accuracyzeroshot.csv'))

    # List of native listeners
    native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']

    # Filter for non-native listeners only
    df_fewshot = df_fewshot[df_fewshot['Listener'].isin(native_listeners)]
    df_zeroshot = df_zeroshot[df_zeroshot['Listener'].isin(native_listeners)]

    # Drop Control adults and Control children columns
    drop_cols = ["Control adults", "Control children"]
    df_fewshot = df_fewshot.drop(columns=drop_cols, errors='ignore')
    df_zeroshot = df_zeroshot.drop(columns=drop_cols, errors='ignore')

    # Add patient average columns
    df_fewshot["Patient Average"] = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)
    df_zeroshot["Patient Average"] = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia"]].mean(axis=1)

    # Compute column-wise averages
    b0_triangle = df_fewshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()
    b0_circle = df_zeroshot[["CLP", "Dysarthria", "Dysglossia", "Dysphonia", "Patient Average"]].mean().tolist()

    # subplot 2
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

    # List of native listeners
    native_listeners = ['EN', 'MS', 'TN', 'LB', 'TG']

    # Filter for non-native listeners only
    df_long = df_long[df_long['Listener'].isin(native_listeners)]

    # Compute group-wise means
    grouped = df_long.groupby(['Pathology', 'Type'])['Score'].mean().unstack()

    # Compute overall average over the 4 patient groups
    patients = ['CLP', 'Dysarthria', 'Dysglossia', 'Dysphonia']
    grouped = grouped.loc[patients]
    grouped.loc['Patient Average'] = grouped.mean()

    b1_circle = grouped['Anonymized'].tolist()
    b2_circle = grouped['Original'].tolist()

    a1 = [94.86, 98.86, 98.38, 96.37, 96.07]  # AUC anonym [fixed]
    a2 = [97.33, 97.73, 99.12, 96.44, 97.05]  # AUC org [fixed]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # --- First Subplot ---
    axs[0].scatter(b0_circle, a0, color='blue', marker='o', label='Zero-shot Turing', s=120)
    axs[0].scatter(b0_triangle, a0, color='orange', marker='^', label='Few-shot Turing', s=120)
    axs[0].set_title("EER vs. Turing accuracy", fontsize=20, loc='center')
    axs[0].set_xlabel("Turing accuracy [%]", fontsize=18)
    axs[0].set_ylabel("EER [%]", fontsize=18)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].legend(fontsize=16)
    # axs[0].set_ylim(int(min(a0)), int(max(a0) +3))
    axs[0].set_ylim(29.7, 42)
    # axs[0].set_xlim(int(min(b0_circle)-3), 100)
    axs[0].set_xlim(79.01, 100)

    # --- Second Subplot ---
    axs[1].scatter(b1_circle, a1, color='red', marker='s', label='Anonymized', s=120)  # Red squares
    axs[1].set_title("Quality anonymized", fontsize=20, loc='center')
    axs[1].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[1].set_ylabel("AUC [%]", fontsize=18)
    axs[1].tick_params(axis='both', labelsize=16)
    # axs[1].legend(fontsize=16)
    # axs[1].set_ylim(int(min(a1)-1), 100)
    axs[1].set_ylim(93.5, 100)
    # axs[1].set_xlim(int(min(b1_circle)-1), int(max(b1_circle) +2))
    axs[1].set_xlim(49.1, 70)

    # --- Third Subplot ---
    axs[2].scatter(b2_circle, a2, color='black', marker='*', label='Original', s=120)  # Red squares
    axs[2].set_title("Quality original", fontsize=20, loc='center')
    axs[2].set_xlabel("Perceived quality normalized [%]", fontsize=18)
    axs[2].set_ylabel("AUC [%]", fontsize=18)
    axs[2].tick_params(axis='both', labelsize=16)
    # axs[2].legend(fontsize=16)
    # axs[2].set_ylim(int(min(a2)-1), 100)
    axs[2].set_ylim(95.49, 100)
    # axs[2].set_xlim(int(min(b2_circle)-1), int(max(b2_circle) +2))
    axs[2].set_xlim(74.7, 95)


    plt.tight_layout()
    plt.savefig("./correlation/scatter_plot_correlation_native.png")

    ############################# Appender #############################

    img1_path = "./correlation/scatter_plot_correlation.png"
    img2_path = "./correlation/scatter_plot_correlation_nonnative.png"
    img3_path = "./correlation/scatter_plot_correlation_native.png"

    # Open images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img3 = Image.open(img3_path)

    # Create a vertical stack of subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 11))  # Slightly smaller height

    # Turn off axis display for each subplot
    for ax in axs:
        ax.axis('off')

    # Show each image in its respective subplot and add a title
    axs[0].imshow(img1)
    axs[0].set_title("a) All listeners", fontsize=14, loc='left')

    axs[1].imshow(img2)
    axs[1].set_title("b) Non-native listeners", fontsize=14, loc='left')

    axs[2].imshow(img3)
    axs[2].set_title("c) Native listeners", fontsize=14, loc='left')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("./correlation/final_scatter_plot_correlation.png")



def boxplot_appender():
    img1 = mpimg.imread('./zero_shot/turing_box_zeroshot.png')
    img2 = mpimg.imread('./few_shot/turing_box_fewshot.png')

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Optimal screen fit

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


