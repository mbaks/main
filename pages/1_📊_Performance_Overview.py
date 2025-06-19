# pages/1_📊_Performance_Overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
import calendar # Import for calendar operations
import numpy as np # Import for numerical operations, especially with NaN
import seaborn as sns # Import for heatmap visualization
import matplotlib.pyplot as plt # Import for plotting


st.set_page_config(layout="wide")

st.title("📊 Performance Overview")
st.markdown("Dive into the key performance indicators and visualize your overall trading results.")

# Custom CSS for styling the metric boxes
st.markdown("""
<style>
    div.stMetric {
        background-color: #f0f2f6; /* Light grey background */
        border: 1px solid #e0e0e0; /* Subtle border */
        border-radius: 10px; /* Rounded corners */
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Soft shadow */
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333; /* Default dark color */
    }
    .positive {
        color: #28a745 !important; /* Green for positive */
    }
    .negative {
        color: #dc3545 !important; /* Red for negative */
    }
    .neutral {
        color: #007bff !important; /* Blue for neutral/info, e.g., trade count */
    }
</style>
""", unsafe_allow_html=True)


# Check if data is available in session state
if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df.copy()

    if df.empty:
        st.warning("No data available for analysis. Please upload and clean a file on the 'Home' page.")
    else:
        st.write("---")
        st.header("Key Performance Indicators (KPIs)")

        # Calculate KPIs
        total_profit = df['Profit'].sum()
        total_commission = df['Commission'].sum()
        total_swap = df['Swap'].sum()
        total_net_profit = df['Net Profit'].sum()
        total_trades = len(df)
        profitable_trades = df['Is Profitable'].sum()
        loss_trades = total_trades - profitable_trades

        # Avoid division by zero for win rate
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

        avg_profit_per_trade = df['Profit'].mean() if total_trades > 0 else 0
        avg_net_profit_per_trade = df['Net Profit'].mean() if total_trades > 0 else 0

        # Helper function to format and color values
        def format_kpi_value(value, is_currency=True, is_percentage=False, color_based_on_sign=False):
            if is_currency:
                formatted_value = f"R {value:,.2f}"
            elif is_percentage:
                formatted_value = f"{value:,.2f}%"
            else:
                formatted_value = f"{value:,}"

            color_class = "neutral"
            if color_based_on_sign:
                if value > 0:
                    color_class = "positive"
                elif value < 0:
                    color_class = "negative"
            elif is_percentage: # Win rate specifically
                if value >= 50: # Example threshold for good win rate
                    color_class = "positive"
                else:
                    color_class = "negative"

            return f'<div class="metric-value {color_class}">{formatted_value}</div>'


        # Display KPIs using columns with custom styling
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f'<div class="metric-label">Total Gross Profit</div>{format_kpi_value(total_profit, color_based_on_sign=True)}', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Total Commission</div>{format_kpi_value(total_commission, color_based_on_sign=True)}', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-label">Total Swap</div>{format_kpi_value(total_swap, color_based_on_sign=True)}', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Total Net Profit</div>{format_kpi_value(total_net_profit, color_based_on_sign=True)}', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-label">Total Trades</div>{format_kpi_value(total_trades, is_currency=False)}', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Profitable Trades</div>{format_kpi_value(profitable_trades, is_currency=False)}', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-label">Losing Trades</div>{format_kpi_value(loss_trades, is_currency=False)}', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Win Rate</div>{format_kpi_value(win_rate, is_currency=False, is_percentage=True, color_based_on_sign=False)}', unsafe_allow_html=True)

        st.write("---")
        st.header("Profit Over Time")

        # Ensure 'Time' column is datetime and sort
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.sort_values('Time', inplace=True)

            # Cumulative Profit
            df['Cumulative Profit'] = df['Profit'].cumsum()
            df['Cumulative Net Profit'] = df['Net Profit'].cumsum()

            fig_cumulative = px.line(df, x='Time', y=['Cumulative Profit', 'Cumulative Net Profit'],
                                    title='Cumulative Profit Over Time',
                                    labels={'value': 'Profit (R)', 'Time': 'Date'},
                                    hover_data={'value': ':.2f', 'Time': '|%Y-%m-%d %H:%M:%S'},
                                    template="plotly_white")
            fig_cumulative.update_layout(hovermode="x unified")
            st.plotly_chart(fig_cumulative, use_container_width=True)

            # Daily Profit
            df['Date'] = df['Time'].dt.date
            daily_profit_overview = df.groupby('Date')[['Profit', 'Net Profit']].sum().reset_index()

            fig_daily = px.bar(daily_profit_overview, x='Date', y='Net Profit',
                                title='Daily Net Profit',
                                labels={'Net Profit': 'Net Profit (R)', 'Date': 'Date'},
                                template="plotly_white",
                                color='Net Profit',
                                color_continuous_scale=px.colors.diverging.RdYlGn) # Corrected: using diverging.RdYlGn
            st.plotly_chart(fig_daily, use_container_width=True)

        else:
            st.warning(" 'Time' column not found for time-series analysis.")

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


else:
    st.info("No trading data loaded. Please go to the 'Home' page to upload and clean your file.")

