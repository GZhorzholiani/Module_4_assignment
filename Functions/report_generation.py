import matplotlib.pyplot as plt


def report_generation(start_date, end_date, dataframe, dataframe_name, save_path=None, text_file_path=None):
    # Filter the dataframe to include only rows within the specified date range
    df_period = dataframe[(dataframe["Date"] >= start_date) & (dataframe["Date"] <= end_date)]

    # Aggregate the data by summing up the values for each column in the filtered dataframe
    summary = df_period.agg({
        "HospitalizedPatients": "sum",
        "PatientsInIntensiveCare": "sum",
        "TotalHospitalizedPatients": "sum",
        "HomeConfinement": "sum",
        "CurrentPositiveCases": "sum",
        "NewPositiveCases": "sum",
        "Healed": "sum",
        "Dead": "sum",
        "TotalPositiveCases": "sum",
        "TestsExecuted": "sum"
    }).to_dict()

    # Create a figure with a 3x2 grid of subplots for visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle(f"COVID-19 Data {dataframe_name} Visualization ({start_date} - {end_date})",
                 fontsize=16, fontweight='bold')

    # Define categories and values for the first bar chart (Patient Counts)
    categories1 = ["Hospitalized", "ICU", "Home Confinement", "Total Hospitalized"]
    values1 = [summary["HospitalizedPatients"], summary["PatientsInIntensiveCare"],
               summary["HomeConfinement"], summary["TotalHospitalizedPatients"]]

    # Plot the first bar chart
    bars1 = axes[0, 0].bar(categories1, values1, color=['blue', 'red', 'purple', 'orange'])
    axes[0, 0].set_title("Total Patient Counts", fontsize=14, fontweight='bold', pad=25)
    axes[0, 0].set_ylabel("Count", fontsize=10)

    # Add value labels on top of each bar in the first chart
    for bar, value in zip(bars1, values1):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.08 * max(values1),
                        f"{value:,}", ha='center', fontsize=10, fontweight='bold', color='black')

    # Plot the line chart for the daily trend of total positive cases
    axes[0, 1].plot(df_period["Date"], df_period["TotalPositiveCases"], label="Total Cases", color="blue", marker="o")
    axes[0, 1].set_title("Daily Trend: Total Positive Cases", fontsize=14, fontweight='bold', pad=25)
    axes[0, 1].set_ylabel("Count", fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Define categories and values for the second bar chart (Recovered & Deaths)
    categories2 = ["Recovered", "Deaths"]
    values2 = [summary["Healed"], summary["Dead"]]

    # Plot the second bar chart
    bars2 = axes[1, 0].bar(categories2, values2, color=['green', 'red'])
    axes[1, 0].set_title("Recovered & Deaths", fontsize=14, fontweight='bold', pad=25)
    axes[1, 0].set_ylabel("Count", fontsize=10)

    # Add value labels on top of each bar in the second chart
    for bar, value in zip(bars2, values2):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.08 * max(values2),
                        f"{value:,}", ha='center', fontsize=10, fontweight='bold', color='black')

    # Plot the line chart for the daily trend of recovered and deaths
    axes[1, 1].plot(df_period["Date"], df_period["Healed"], label="Recovered", color="green", marker="o")
    axes[1, 1].plot(df_period["Date"], df_period["Dead"], label="Deaths", color="red", marker="o")
    axes[1, 1].set_title("Daily Trend: Recovered & Deaths", fontsize=14, fontweight='bold', pad=25)
    axes[1, 1].set_ylabel("Count", fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", rotation=45)

    # Define categories and values for the third bar chart (New Positive Cases & Total Positive Cases)
    categories3 = ["New Positive Cases", "Total Positive Cases"]
    values3 = [summary["NewPositiveCases"], summary["TotalPositiveCases"]]

    # Plot the third bar chart
    bars3 = axes[2, 0].bar(categories3, values3, color=['black', 'brown'])
    axes[2, 0].set_title("New Positive Cases & Total Positive Cases", fontsize=14, fontweight='bold', pad=25)
    axes[2, 0].set_ylabel("Count", fontsize=10)

    # Add value labels on top of each bar in the third chart
    for bar, value in zip(bars3, values3):
        axes[2, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.08 * max(values3),
                        f"{value:,}", ha='center', fontsize=10, fontweight='bold', color='black')

    # Plot the line chart for the daily trend of new positive cases
    axes[2, 1].plot(df_period["Date"], df_period["NewPositiveCases"], label="New Cases", color="black", marker="o")
    axes[2, 1].set_title("Daily Trend: New Cases", fontsize=14, fontweight='bold', pad=25)
    axes[2, 1].set_ylabel("Count", fontsize=10)
    axes[2, 1].legend()
    axes[2, 1].tick_params(axis="x", rotation=45)

    # Print summary statistics
    summary_text = f"Date Range: {start_date} - {end_date}\n"
    for key, value in summary.items():
        summary_text += f"{key.replace('_', ' ').title()}: {int(value):,}\n"

    # Write the summary to a text file
    if text_file_path:
        with open(text_file_path, "w") as f:
            f.write(summary_text)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the graph if a path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
    else:
        plt.show()
