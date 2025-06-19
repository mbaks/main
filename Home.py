# Home.py
import streamlit as st
import pandas as pd
import calendar # Import for calendar operations
import numpy as np # Import for numerical operations, especially with NaN
import seaborn as sns # Import for heatmap visualization
import matplotlib.pyplot as plt # Import for plotting

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

st.title("📂 Upload and Clean Trading History")
st.markdown("Upload your Excel or CSV trading history file to prepare it for analysis. This page will clean the data and make it ready for the dashboard pages.")

# Initialize session state for cleaned_df if it doesn't exist
# This is crucial for passing data between pages
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # Step 1: Load with header from row 7 (for CSV) or row 1 (for Excel with skiprows)
        # Assuming the CSV has headers on row 7, and Excel on row 1 with 6 rows to skip
        if uploaded_file.name.endswith('.csv'):
            # For CSV, read header from the 7th row (index 6), and skip rows 0-5
            df = pd.read_csv(uploaded_file, header=6, skiprows=range(6))
        else:
            # For Excel, header=0 means the first row, and skiprows=6 means skip first 6 rows.
            # This implies row 7 in Excel is now the header row.
            df = pd.read_excel(uploaded_file, header=0, skiprows=6)

        # Step 2: Truncate at 'Orders' column if it exists and is not the first column
        # The original code truncates up to the 'Orders' column, excluding it.
        # Let's ensure 'Orders' is not the very first column to avoid empty dataframe.
        if 'Orders' in df.columns and df.columns.get_loc('Orders') > 0:
            orders_index = df.columns.get_loc('Orders')
            df = df.iloc[:, :orders_index]
        elif 'Orders' in df.columns and df.columns.get_loc('Orders') == 0:
            st.warning(" 'Orders' column found at the very beginning. Skipping truncation based on 'Orders'.")


        # Step 3: Drop 'Unnamed' columns
        # This line removes columns whose names contain "Unnamed"
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
        # na=False handles potential NaN column names, though unlikely after read.

        # Step 4: Truncate rows after first empty 'Profit'
        # This identifies the first row where 'Profit' is NaN and keeps rows before it.
        if 'Profit' in df.columns:
            # Find the index of the first row where 'Profit' is NaN (empty)
            first_empty_profit_index = df[df['Profit'].isna()].index.min()
            if pd.notna(first_empty_profit_index): # Check if an empty profit was found
                # Keep rows from the beginning up to, but not including, the first empty profit row
                df = df.loc[:first_empty_profit_index - 1]
            else:
                st.info("No empty 'Profit' values found to truncate rows.")
        else:
            st.warning(" 'Profit' column not found for row truncation.")


        # Step 5: Rename columns for consistency
        # Renames specific columns to more descriptive names.
        df.rename(columns={
            'Time.1': 'Closing Time',
            'Price.1': 'Closing Price'
        }, inplace=True)

        # Step 6: Convert data types
        # - Commission, Profit, Swap → numeric (float)
        # - Time, Closing Time → datetime objects
        numeric_cols = ['Commission', 'Profit', 'Swap', 'Volume', 'Price', 'S/L', 'T/P', 'Closing Price']
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN (missing values)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        datetime_cols = ['Time', 'Closing Time']
        for col in datetime_cols:
            if col in df.columns:
                # Convert to datetime, coercing errors to NaT (Not a Time)
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # --- Additional cleaning/feature engineering based on typical trading data ---
        # Calculate 'Duration' if both 'Time' and 'Closing Time' exist
        if 'Time' in df.columns and 'Closing Time' in df.columns:
            df['Duration'] = (df['Closing Time'] - df['Time']).dt.total_seconds() / 60 # Duration in minutes
            df['Duration'] = df['Duration'].round(2) # Round to 2 decimal places

        # Calculate 'Net Profit' (Profit + Swap + Commission)
        # Ensure columns exist before attempting addition
        df['Net Profit'] = df['Profit'].fillna(0) + df['Swap'].fillna(0) + df['Commission'].fillna(0)
        df['Net Profit'] = df['Net Profit'].round(2)

        # Determine if a trade was profitable
        df['Is Profitable'] = df['Profit'] > 0

        # Store the cleaned DataFrame in session state for other pages to access
        st.session_state.cleaned_df = df

        # Step 7: Format preview for display (keeping original df for download)
        df_display = df.copy()
        if 'Profit' in df_display.columns:
            # Format 'Profit' with 'R' prefix and 2 decimal places for display
            df_display['Profit'] = df_display['Profit'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")
        if 'Net Profit' in df_display.columns:
            # Format 'Net Profit' similarly
            df_display['Net Profit'] = df_display['Net Profit'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")
        if 'Commission' in df_display.columns:
            # Format 'Commission' similarly
            df_display['Commission'] = df_display['Commission'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")
        if 'Swap' in df_display.columns:
            # Format 'Swap' similarly
            df_display['Swap'] = df_display['Swap'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")

        # Format datetime columns for display
        for col in ['Time', 'Closing Time']:
            if col in df_display.columns:
                df_display[col] = df_display[col].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("") # Use .fillna("") for NaT


        # Step 8: Show preview
        st.write("✅ Final Cleaned Data Preview:")
        st.dataframe(df_display)

        # Step 9: Download cleaned (raw) version
        # Converts the raw cleaned DataFrame (not the display-formatted one) to CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_Trading_History.csv",
            mime="text/csv"
        )

        # --- Monthly Profit Calendar with navigation ---
        st.markdown("---") # Separator for new section
        st.subheader("📅 Monthly Profit Calendar")

        # Use 'df' here instead of 'filtered'
        if "Closing Time" in df.columns and "Profit" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Closing Time"]):
            if not df["Closing Time"].empty and not df["Profit"].empty:
                # Calculate daily profit using the 'Closing Time'
                daily_profit_calendar = df.groupby(df["Closing Time"].dt.date)["Profit"].sum().reset_index()
                daily_profit_calendar.columns = ['Date', 'Profit']

                # Ensure 'Date' column in daily_profit_calendar is datetime
                daily_profit_calendar['Date'] = pd.to_datetime(daily_profit_calendar['Date'], errors='coerce')

                if daily_profit_calendar['Date'].isnull().all():
                    st.error("All dates in daily profit are invalid. Cannot generate calendar.")
                else:
                    # Drop rows where 'Date' became NaT during re-conversion
                    daily_profit_calendar.dropna(subset=['Date'], inplace=True)

                    if daily_profit_calendar.empty:
                        st.info("No valid daily profit data after date conversion for calendar.")
                    else: # This is the corrected flow: only proceed if daily_profit_calendar is NOT empty
                        # Get min and max year from the data
                        min_year = int(daily_profit_calendar['Date'].min().year)
                        max_year = int(daily_profit_calendar['Date'].max().year)

                        # Use current year if data is empty or outside valid range
                        default_year = int(pd.Timestamp.now().year)
                        if not daily_profit_calendar.empty:
                            if default_year < min_year or default_year > max_year:
                                default_year = min_year # Set to min_year if current year is out of range

                        # Month and Year selection in sidebar for consistency
                        st.sidebar.subheader("Calendar Navigation")

                        # Create year selectbox
                        selected_year = st.sidebar.selectbox(
                            "Select Year",
                            options=list(range(min_year, max_year + 1)),
                            index=list(range(min_year, max_year + 1)).index(default_year) if default_year in range(min_year, max_year + 1) else 0
                        )

                        # Create month selectbox
                        month_names = [calendar.month_name[i] for i in range(1, 13)]
                        selected_month_name = st.sidebar.selectbox(
                            "Select Month",
                            options=month_names,
                            index=pd.Timestamp.now().month - 1 # Default to current month
                        )
                        selected_month = month_names.index(selected_month_name) + 1 # Convert name to month number


                        # Filter daily profit for the selected month and year
                        monthly_profit_calendar = daily_profit_calendar[
                            (daily_profit_calendar['Date'].dt.year == selected_year) &
                            (daily_profit_calendar['Date'].dt.month == selected_month)
                        ]

                        # Create a full calendar grid for the selected month
                        cal = calendar.Calendar(firstweekday=0) # Monday is the first day of the week
                        month_days = cal.monthdatescalendar(selected_year, selected_month)

                        # Prepare data for heatmap
                        heatmap_data = np.full((len(month_days), 7), np.nan) # Initialize with NaN for empty cells
                        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                        week_labels = []

                        for week_idx, week in enumerate(month_days):
                            week_label_added_for_this_week = False # Reset for each week
                            for day_idx, date_obj in enumerate(week):
                                # Only process dates within the selected month for profit data
                                if date_obj.month == selected_month:
                                    day_profit = monthly_profit_calendar[monthly_profit_calendar['Date'] == pd.Timestamp(date_obj)]['Profit'].sum()
                                    if not pd.isna(day_profit):
                                        heatmap_data[week_idx, day_idx] = day_profit
                                    
                                    # Add week label based on the first day of the week in the selected month, but only once per valid week
                                    if not week_label_added_for_this_week:
                                        week_labels.append(f"Week {date_obj.isocalendar()[1]}") # ISO week number
                                        week_label_added_for_this_week = True
                                elif week_idx == 0 and date_obj.month < selected_month: # Handle days from previous month in first week
                                    if not week_label_added_for_this_week:
                                        week_labels.append(f"Week {date_obj.isocalendar()[1]}")
                                        week_label_added_for_this_week = True
                                elif week_idx == len(month_days) - 1 and date_obj.month > selected_month: # Handle days from next month in last week
                                    if not week_label_added_for_this_week:
                                        week_labels.append(f"Week {date_obj.isocalendar()[1]}")
                                        week_label_added_for_this_week = True
                            # If a week had no days from the selected month (e.g., all days from prev/next month), ensure a label is still added
                            if not week_label_added_for_this_week and week_idx >= len(week_labels):
                                # This might happen if all days in a row are not in the selected month
                                # We can append a generic week number or handle based on first day if outside month
                                # For simplicity, let's ensure it's added if it hasn't been by checking its index
                                if week_idx < len(week_labels):
                                    pass # Already added
                                else: # Fallback for weeks without any selected month days, though `monthdatescalendar` usually gives context
                                    week_labels.append(f"Week {week[0].isocalendar()[1]}")


                        fig_cal, ax_cal = plt.subplots(figsize=(12, min(len(month_days) * 1.5, 12))) # Adjust height based on number of weeks
                        
                        # Determine vmin and vmax for the diverging colormap
                        max_abs_profit = np.nanmax(np.abs(heatmap_data))
                        
                        sns.heatmap(heatmap_data,
                                    annot=True,
                                    fmt=".2f",
                                    cmap="RdYlGn",
                                    center=0,
                                    vmin=-max_abs_profit if not np.isnan(max_abs_profit) else None,
                                    vmax=max_abs_profit if not np.isnan(max_abs_profit) else None,
                                    cbar_kws={'label': 'Daily Profit (R)'},
                                    linewidths=.5,
                                    linecolor='black',
                                    ax=ax_cal,
                                    xticklabels=day_labels,
                                    yticklabels=week_labels if week_labels else False) # Use week labels or hide if empty

                        ax_cal.set_title(f"Daily Profit for {selected_month_name} {selected_year}", fontsize=16, pad=20)
                        ax_cal.set_xlabel("Day of Week", fontsize=12)
                        ax_cal.set_ylabel("Week", fontsize=12)

                        # Add day numbers to the cells
                        for week_idx, week in enumerate(month_days):
                            for day_idx, date_obj in enumerate(week):
                                # Only add day numbers for the actual month's days
                                if date_obj.month == selected_month:
                                    # Choose text color based on profit for readability on heatmap
                                    current_profit = heatmap_data[week_idx, day_idx]
                                    text_color = "black"
                                    if not pd.isna(current_profit):
                                        # Simple heuristic: if background is very light or very dark
                                        # You might need a more sophisticated color-contrast algorithm here
                                        if current_profit > (max_abs_profit * 0.5): # Very profitable, might be dark green
                                            text_color = "white"
                                        elif current_profit < -(max_abs_profit * 0.5): # Very losing, might be dark red
                                            text_color = "white"
                                            
                                    ax_cal.text(day_idx + 0.5, week_idx + 0.8, f"{date_obj.day}",
                                                ha='center', va='bottom', color=text_color, fontsize=10, weight='bold')

                        plt.tight_layout() # Adjust layout to prevent labels from overlapping
                        st.pyplot(fig_cal)
                        plt.close(fig_cal) # Close the figure to free up memory

            else:
                st.warning("Cannot plot 'Monthly Profit Calendar' as 'Closing Time' or 'Profit' columns are empty after filtering.")
        else:
            st.warning("Cannot plot 'Monthly Profit Calendar' as 'Closing Time' or 'Profit' column is missing or 'Closing Time' is not datetime type.")


    except Exception as e:
        # Display an error message if anything goes wrong during processing
        st.error(f"❌ Error during file processing: {e}")
        st.warning("Please ensure your file format matches the expected structure (header on row 7, 'Orders' column for truncation, 'Profit' column for row truncation).")
else:
    # Message to the user when no file is uploaded yet
    st.info("Please upload your trading history file to begin cleaning and analysis.")
