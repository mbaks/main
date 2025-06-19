# pages/3_🔍_Advanced_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🔍 Advanced Analysis")
st.markdown("Delve deeper into your trading performance with breakdowns by various categories.")

# Check if data is available in session state
if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df.copy()

    if df.empty:
        st.warning("No data available for analysis. Please upload and clean a file on the 'Home' page.")
    else:
        st.write("---")

        # --- Analysis by Symbol ---
        st.header("Analysis by Symbol")
        if 'Symbol' in df.columns and 'Net Profit' in df.columns:
            symbol_performance = df.groupby('Symbol')['Net Profit'].agg(['sum', 'count', 'mean']).reset_index()
            symbol_performance.columns = ['Symbol', 'Total Net Profit (R)', 'Number of Trades', 'Average Net Profit (R)']
            symbol_performance['Total Net Profit (R)'] = symbol_performance['Total Net Profit (R)'].round(2)
            symbol_performance['Average Net Profit (R)'] = symbol_performance['Average Net Profit (R)'].round(2)
            symbol_performance = symbol_performance.sort_values('Total Net Profit (R)', ascending=False)

            st.subheader("Symbol Performance Table")
            st.dataframe(symbol_performance, use_container_width=True)

            # Bar chart for Total Net Profit by Symbol
            fig_symbol_profit = px.bar(symbol_performance, x='Symbol', y='Total Net Profit (R)',
                                       title='Total Net Profit by Symbol',
                                       labels={'Total Net Profit (R)': 'Total Net Profit (R)', 'Symbol': 'Trading Symbol'},
                                       template="plotly_white",
                                       color='Total Net Profit (R)',
                                       color_continuous_scale=px.colors.diverging.RdYlGn) # Use diverging for profit/loss
            st.plotly_chart(fig_symbol_profit, use_container_width=True)

            st.write("---")
        else:
            st.info(" 'Symbol' or 'Net Profit' columns not found for Symbol analysis.")


        # --- Analysis by Type (Buy/Sell) ---
        st.header("Analysis by Trade Type")
        if 'Type' in df.columns and 'Net Profit' in df.columns:
            type_performance = df.groupby('Type')['Net Profit'].agg(['sum', 'count', 'mean']).reset_index()
            type_performance.columns = ['Type', 'Total Net Profit (R)', 'Number of Trades', 'Average Net Profit (R)']
            type_performance['Total Net Profit (R)'] = type_performance['Total Net Profit (R)'].round(2)
            type_performance['Average Net Profit (R)'] = type_performance['Average Net Profit (R)'].round(2)
            type_performance = type_performance.sort_values('Total Net Profit (R)', ascending=False)

            st.subheader("Trade Type Performance Table")
            st.dataframe(type_performance, use_container_width=True)

            # Pie chart for distribution of trades by type
            fig_type_trades = px.pie(type_performance, values='Number of Trades', names='Type',
                                     title='Distribution of Trades by Type (Buy/Sell)',
                                     template="plotly_white")
            st.plotly_chart(fig_type_trades, use_container_width=True)

            # Bar chart for Total Net Profit by Type
            fig_type_profit = px.bar(type_performance, x='Type', y='Total Net Profit (R)',
                                     title='Total Net Profit by Trade Type',
                                     labels={'Total Net Profit (R)': 'Total Net Profit (R)', 'Type': 'Trade Type'},
                                     template="plotly_white",
                                     color='Total Net Profit (R)',
                                     color_continuous_scale=px.colors.diverging.RdYlGn)
            st.plotly_chart(fig_type_profit, use_container_width=True)

            st.write("---")
        else:
            st.info(" 'Type' or 'Net Profit' columns not found for Trade Type analysis.")


        # --- Profit/Loss Distribution ---
        st.header("Profit/Loss Distribution")
        if 'Net Profit' in df.columns:
            st.subheader("Net Profit Distribution (Histogram)")
            fig_profit_dist = px.histogram(df, x='Net Profit', nbins=50,
                                           title='Distribution of Net Profit per Trade',
                                           labels={'Net Profit': 'Net Profit (R)'},
                                           template="plotly_white",
                                           color_discrete_sequence=['skyblue']) # A single color for histogram
            fig_profit_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even", annotation_position="top right")
            st.plotly_chart(fig_profit_dist, use_container_width=True)
            st.write("---")
        else:
            st.info(" 'Net Profit' column not found for profit/loss distribution analysis.")


else:
    st.info("No trading data loaded. Please go to the 'Home' page to upload and clean your file.")

