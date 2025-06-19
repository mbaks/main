# pages/2_📈_Trade_Details.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📈 Trade Details")
st.markdown("Explore your individual trades. Use the filters to narrow down your analysis.")

# Check if data is available in session state
if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df.copy()

    if df.empty:
        st.warning("No data available for analysis. Please upload and clean a file on the 'Home' page.")
    else:
        st.write("---")
        st.header("All Trades")

        # --- Filters ---
        st.sidebar.header("Filter Trades")

        # Symbol filter
        if 'Symbol' in df.columns:
            unique_symbols = ['All'] + sorted(df['Symbol'].unique().tolist())
            selected_symbols = st.sidebar.multiselect(
                "Select Symbol(s)",
                options=unique_symbols,
                default='All'
            )
            if 'All' not in selected_symbols:
                df = df[df['Symbol'].isin(selected_symbols)]
        else:
            st.sidebar.info(" 'Symbol' column not found for filtering.")


        # Type (Buy/Sell) filter
        if 'Type' in df.columns:
            unique_types = ['All'] + sorted(df['Type'].unique().tolist())
            selected_types = st.sidebar.multiselect(
                "Select Type(s)",
                options=unique_types,
                default='All'
            )
            if 'All' not in selected_types:
                df = df[df['Type'].isin(selected_types)]
        else:
            st.sidebar.info(" 'Type' column not found for filtering.")

        # Profitability filter
        if 'Is Profitable' in df.columns:
            profit_status = st.sidebar.radio(
                "Filter by Profitability",
                options=["All", "Profitable", "Loss"],
                index=0
            )
            if profit_status == "Profitable":
                df = df[df['Is Profitable'] == True]
            elif profit_status == "Loss":
                df = df[df['Is Profitable'] == False]
        else:
            st.sidebar.info(" 'Is Profitable' column not found for filtering.")


        # Date Range filter
        if 'Time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Time']):
            min_date = df['Time'].min().date() if not df['Time'].empty else pd.to_datetime("2000-01-01").date()
            max_date = df['Time'].max().date() if not df['Time'].empty else pd.to_datetime("2000-01-01").date()

            date_range = st.sidebar.date_input(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date) if not df['Time'].empty else (pd.to_datetime("2000-01-01").date(), pd.to_datetime("2000-01-01").date())
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)]
        else:
            st.sidebar.info(" 'Time' column not found or is not a datetime type for date filtering.")


        st.sidebar.markdown("---")

        # --- Display Filtered Data ---
        if df.empty:
            st.warning("No trades match the selected filters.")
        else:
            st.write(f"Displaying {len(df)} trades out of {len(st.session_state.cleaned_df)} total trades.")

            # Option to hide/show columns
            all_cols = st.columns([0.8, 0.2])
            with all_cols[0]:
                st.subheader("Filtered Trades Table")
            with all_cols[1]:
                if st.checkbox("Show/Hide Columns", value=False):
                    columns_to_show = st.multiselect(
                        "Select columns to display",
                        options=st.session_state.cleaned_df.columns.tolist(),
                        default=st.session_state.cleaned_df.columns.tolist()
                    )
                    df_display = df[columns_to_show].copy()
                else:
                    df_display = df.copy()

            # Format specific columns for display
            if 'Profit' in df_display.columns:
                df_display['Profit'] = df_display['Profit'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")
            if 'Net Profit' in df_display.columns:
                df_display['Net Profit'] = df_display['Net Profit'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")
            if 'Commission' in df_display.columns:
                df_display['Commission'] = df_display['Commission'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")
            if 'Swap' in df_display.columns:
                df_display['Swap'] = df_display['Swap'].map(lambda x: f"R {x:,.2f}" if pd.notna(x) else "")
            for col in ['Time', 'Closing Time']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")


            st.dataframe(df_display, use_container_width=True)

            # Download filtered data
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered CSV",
                data=csv_data,
                file_name="filtered_Trading_History.csv",
                mime="text/csv"
            )

            st.write("---")
            st.header("Visualizations of Filtered Data")

            # --- Visual 1: Total Net Profit by Symbol (for filtered data) ---
            if 'Symbol' in df.columns and 'Net Profit' in df.columns:
                st.subheader("Net Profit by Symbol (Filtered)")
                filtered_symbol_profit = df.groupby('Symbol')['Net Profit'].sum().reset_index()
                fig_filtered_symbol_profit = px.bar(filtered_symbol_profit, x='Symbol', y='Net Profit',
                                                    title='Total Net Profit by Symbol (Filtered Data)',
                                                    labels={'Net Profit': 'Net Profit (R)'},
                                                    template="plotly_white",
                                                    color='Net Profit',
                                                    color_continuous_scale=px.colors.diverging.RdYlGn)
                st.plotly_chart(fig_filtered_symbol_profit, use_container_width=True)
            else:
                st.info(" 'Symbol' or 'Net Profit' columns not available for symbol analysis in filtered data.")


            # --- Visual 2: Trade Type Distribution (for filtered data) ---
            if 'Type' in df.columns:
                st.subheader("Trade Type Distribution (Filtered)")
                filtered_type_counts = df['Type'].value_counts().reset_index()
                filtered_type_counts.columns = ['Type', 'Count']
                fig_filtered_type_dist = px.pie(filtered_type_counts, values='Count', names='Type',
                                                title='Distribution of Trade Types (Filtered Data)',
                                                template="plotly_white")
                st.plotly_chart(fig_filtered_type_dist, use_container_width=True)
            else:
                st.info(" 'Type' column not available for trade type distribution in filtered data.")


            # --- Visual 3: Net Profit Distribution (for filtered data) ---
            if 'Net Profit' in df.columns:
                st.subheader("Net Profit Distribution (Filtered)")
                fig_filtered_profit_dist = px.histogram(df, x='Net Profit', nbins=50,
                                                        title='Distribution of Net Profit per Trade (Filtered Data)',
                                                        labels={'Net Profit': 'Net Profit (R)'},
                                                        template="plotly_white",
                                                        color_discrete_sequence=['skyblue'])
                fig_filtered_profit_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even", annotation_position="top right")
                st.plotly_chart(fig_filtered_profit_dist, use_container_width=True)
            else:
                st.info(" 'Net Profit' column not available for profit distribution in filtered data.")

            # --- Visual 4: Trade Duration Analysis (Turnaround Time) ---
            if 'Duration' in df.columns and 'Net Profit' in df.columns:
                st.subheader("Trade Turnaround Time (Duration) Analysis")
                
                # Histogram of trade durations
                fig_duration_hist = px.histogram(df, x='Duration', nbins=50,
                                                 title='Distribution of Trade Durations (Minutes)',
                                                 labels={'Duration': 'Trade Duration (Minutes)'},
                                                 template="plotly_white",
                                                 color_discrete_sequence=['orange'])
                st.plotly_chart(fig_duration_hist, use_container_width=True)

                # Scatter plot of Net Profit vs. Duration
                fig_duration_profit_scatter = px.scatter(df, x='Duration', y='Net Profit',
                                                        color='Is Profitable',
                                                        title='Net Profit vs. Trade Duration',
                                                        labels={'Duration': 'Trade Duration (Minutes)', 'Net Profit': 'Net Profit (R)'},
                                                        template="plotly_white",
                                                        hover_data=['Symbol', 'Type', 'Time', 'Closing Time'],
                                                        color_discrete_map={True: 'green', False: 'red'})
                st.plotly_chart(fig_duration_profit_scatter, use_container_width=True)
            else:
                st.info(" 'Duration' or 'Net Profit' columns not available for trade turnaround time analysis. Ensure 'Time' and 'Closing Time' are present and 'Duration' is calculated on the Home page.")

            # --- Visual 5: Volume Statistics ---
            if 'Volume' in df.columns and 'Net Profit' in df.columns:
                st.subheader("Volume Statistics")

                # Histogram of trade volumes
                fig_volume_hist = px.histogram(df, x='Volume', nbins=30,
                                               title='Distribution of Trade Volumes',
                                               labels={'Volume': 'Trade Volume'},
                                               template="plotly_white",
                                               color_discrete_sequence=['purple'])
                st.plotly_chart(fig_volume_hist, use_container_width=True)

                # Total Net Profit by Volume (might need grouping for discrete volumes)
                # To make this useful, let's group volumes into bins or take top N volumes
                # For simplicity, if volume is discrete and not too many unique values, a bar chart works.
                # Otherwise, a scatter plot of profit vs volume or a binned analysis is better.
                if df['Volume'].nunique() < 50: # If relatively few unique volumes
                    volume_profit_summary = df.groupby('Volume')['Net Profit'].sum().reset_index()
                    fig_volume_profit = px.bar(volume_profit_summary, x='Volume', y='Net Profit',
                                               title='Total Net Profit by Trade Volume',
                                               labels={'Volume': 'Trade Volume', 'Net Profit': 'Total Net Profit (R)'},
                                               template="plotly_white",
                                               color='Net Profit',
                                               color_continuous_scale=px.colors.diverging.RdYlGn)
                    st.plotly_chart(fig_volume_profit, use_container_width=True)
                else:
                    st.info("Too many unique volumes to show a bar chart of total profit by each volume. Consider a scatter plot or binned analysis.")
                    fig_volume_profit_scatter = px.scatter(df, x='Volume', y='Net Profit',
                                                            color='Is Profitable',
                                                            title='Net Profit vs. Trade Volume (Scatter)',
                                                            labels={'Volume': 'Trade Volume', 'Net Profit': 'Net Profit (R)'},
                                                            template="plotly_white",
                                                            hover_data=['Symbol', 'Type'],
                                                            color_discrete_map={True: 'green', False: 'red'})
                    st.plotly_chart(fig_volume_profit_scatter, use_container_width=True)

            else:
                st.info(" 'Volume' or 'Net Profit' columns not available for volume statistics.")


else:
    st.info("No trading data loaded. Please go to the 'Home' page to upload and clean your file.")
